"""Manual corrections of a finished run (PLAN.md 7.3). Fixed part of core, not a plug-in.

Corrections only change the current results: they are written straight into the result files of
the image and into the tables of the ID; a new run starts again from the automatic result.
Tools (coordinates in pixels of the full image):
    restore(x, y)   a rejected pore is counted (keeps its id)
    remove(x, y)    a counted pore is rejected (filter "manual"); a drawn pore is erased
    draw(points)    the outline becomes a new pore; it swallows every pore it touches, is clipped
                    to the sample and passes no filter
    area(points)    counted pores lying completely inside are rejected (drawn ones are erased)
    undo(), reset() the last / all corrections of the image are taken back

A pore is addressed by a point (within SEARCH_RADIUS_PX), as in Pore_Detection_V2. After every
action the value steps (LabelStep.changes_pores = False) and all evaluation modules run again and
everything is rewritten: image folder, images.json, Excel, id_values.json.

`Corrections` belongs to the launcher and lives as long as the viewer: it keeps the automatic
state and the list of corrections of every corrected image, so undo also works after browsing.
"""

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from . import geometry, io, plotting, registry, writer
from .config import Config
from .pipeline import (CORE_EXCEL, FRAME_COLUMNS, _bounding_box, _check_scale, _finalize, _merge_results,
                       excel_layout, frame_from_columns, write_id_plots)
from .types import ImageContext, LabeledPores

MANUAL = "manual"               # filter name of pores rejected by hand
SEARCH_RADIUS_PX = 3            # a click this close to a pore still hits it


@dataclass(frozen=True)
class Correction:
    action: str                                 # restore | remove | draw | area
    x: float
    y: float
    points: tuple[tuple[float, float], ...] = ()

    def to_dict(self) -> dict:
        data = {"action": self.action, "x": round(float(self.x), 2), "y": round(float(self.y), 2)}
        if self.points:
            data["points"] = [[round(float(px), 2), round(float(py), 2)] for px, py in self.points]
        return data

    @classmethod
    def outline(cls, action: str, points) -> "Correction":
        points = tuple((float(px), float(py)) for px, py in points)
        if len(points) < 3:
            raise ValueError("an outline needs at least three points")
        xs, ys = zip(*points)
        return cls(action, sum(xs) / len(xs), sum(ys) / len(ys), points)


@dataclass
class _Labels:
    """Counted and rejected pores of one image in full size, plus who was changed by hand."""

    labels: np.ndarray
    rejected_labels: np.ndarray
    rejected: dict[int, tuple[str, str]]        # pore_id -> (filter, reason)
    added: set[int] = field(default_factory=set)        # counted by hand (restored or drawn)
    removed: set[int] = field(default_factory=set)      # rejected by hand
    drawn: set[int] = field(default_factory=set)

    def copy(self) -> "_Labels":
        return _Labels(self.labels.copy(), self.rejected_labels.copy(), dict(self.rejected),
                       set(self.added), set(self.removed), set(self.drawn))


@dataclass(frozen=True)
class State:
    """What the viewer shows after an action (arrays in full image size)."""

    labels: np.ndarray
    rejected_labels: np.ndarray
    rejected: pd.DataFrame          # pore_id, filter, reason
    per_pore: pd.DataFrame          # as pores.json
    row: dict                       # as the row of images.json
    added: frozenset[int]
    removed: frozenset[int]
    corrections: int                # number of corrections of this image
    hit: bool                       # False: the last action hit no pore and changed nothing
    plots: dict[str, bytes] | None = None   # PNG per plot after a change; None = unchanged (files on disk)


class Corrections:
    """Corrections of one run; open a Session per image with session(). Files are written in the
    background, one job after the other; flush() waits for them (call close() at the end)."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        config = Config(yaml.safe_load((self.results_dir / writer.CONFIG_USED).read_text(encoding="utf-8")))
        self.config = config
        self.value_steps = [step for step in (registry.create("postprocessing.label_steps", e)
                                              for e in config.postprocessing.label_steps)
                            if not step.changes_pores]
        self.evaluators = [registry.create("evaluation", e) for e in config.evaluation]
        self.plots = [registry.create("plots", e) for e in config.plots]
        self.layout: dict[str, str | None] = dict(CORE_EXCEL)   # column -> sheet of the results Excel
        self._automatic: dict[str, tuple[bytes, bytes, dict]] = {}     # image -> automatic state (TIFF bytes)
        self._lists: dict[str, list[Correction]] = {}
        self.lock = threading.Lock()                # one action at a time
        self._writer = ThreadPoolExecutor(1, thread_name_prefix="corrections")
        self._pending: list[Future] = []
        self._pending_lock = threading.Lock()

    def session(self, image: str) -> "Session":
        self.flush()                                # the files of the image must be complete
        return Session(self, image)

    def flush(self) -> None:
        """Wait until all files are written; raises the first write error. Any thread."""
        with self._pending_lock:
            pending, self._pending = self._pending, []
        for future in pending:
            future.result()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._writer.shutdown()

    def _submit(self, job) -> None:
        with self._pending_lock:
            self._pending.append(self._writer.submit(job))


class Session:
    """Corrections of one image. Opening takes about as long as the viewer's own loading; an
    action well below a second for 100 MP (writing follows in the background)."""

    def __init__(self, store: Corrections, image: str):
        self.store, self.image = store, image
        self.folder = store.results_dir / image
        table = writer.read_table(store.results_dir / writer.IMAGES_JSON)
        self.row = table[table["image"] == image].iloc[0].to_dict()
        self.info, self.original = io.load_image(Path(self.row["path"]), _image_id(image, self.row["path"]))
        self.roi = _read(self.folder / writer.ROI_MASK_TIF, cv2.IMREAD_GRAYSCALE) > 0
        self.top, self.bottom, self.left, self.right = _bounding_box(self.roi)
        self.current = _Labels(_read(self.folder / writer.LABELS_TIF, cv2.IMREAD_UNCHANGED),
                               _read(self.folder / writer.REJECTED_LABELS_TIF, cv2.IMREAD_UNCHANGED),
                               _rejected_dict(writer.read_table(self.folder / writer.REJECTED_JSON)))
        saved_path = self.folder / writer.CORRECTIONS_JSON
        if saved_path.exists():                     # corrected earlier while the viewer was open
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            self.current.added, self.current.removed = set(saved["added"]), set(saved["removed"])
            self.current.drawn = set(saved["drawn"])
        self.state: State | None = None             # after the last action

    # --- tools ---

    def restore(self, x: float, y: float) -> State:
        return self._do(Correction("restore", float(x), float(y)))

    def remove(self, x: float, y: float) -> State:
        return self._do(Correction("remove", float(x), float(y)))

    def draw(self, points) -> State:
        return self._do(Correction.outline("draw", points))

    def area(self, points) -> State:
        return self._do(Correction.outline("area", points))

    def undo(self) -> State:
        corrections = self.store._lists.get(self.image, [])
        return self._rebuild(corrections[:-1]) if corrections else self._unchanged()

    def reset(self) -> State:
        return self._rebuild([]) if self.store._lists.get(self.image) else self._unchanged()

    @property
    def corrections(self) -> list[Correction]:
        return list(self.store._lists.get(self.image, []))

    # --- applying ---

    def _do(self, correction: Correction) -> State:
        with self.store.lock:
            self._remember_automatic()
            if not _apply(self.current, correction, self.roi):
                return self._unchanged()
            self.store._lists.setdefault(self.image, []).append(correction)
            return self._finish()

    def _rebuild(self, corrections: list[Correction]) -> State:
        """Start again from the automatic state and apply `corrections`."""
        with self.store.lock:
            labels, rejected_labels, rejected = self.store._automatic[self.image]
            self.current = _Labels(_decode(labels), _decode(rejected_labels), dict(rejected))
            self.store._lists[self.image] = [c for c in corrections if _apply(self.current, c, self.roi)]
            return self._finish()

    def _remember_automatic(self) -> None:
        """Before the first change the files on disk are the automatic state: keep their bytes."""
        if self.image not in self.store._automatic:
            self.store._automatic[self.image] = ((self.folder / writer.LABELS_TIF).read_bytes(),
                                                 (self.folder / writer.REJECTED_LABELS_TIF).read_bytes(),
                                                 dict(self.current.rejected))

    def _unchanged(self) -> State:
        if self.state is None:
            self.state = self._snapshot(self.row, writer.read_table(self.folder / writer.PORES_JSON))
        return replace(self.state, hit=False)

    def _finish(self) -> State:
        """Evaluate again (on a copy, so the viewer and the background writer never see later
        changes), then write everything in the background."""
        snapshot = self.current.copy()
        row, per_pore, images = self._evaluate(snapshot)
        picture = lambda: plotting.sample_picture(self.original, snapshot.labels, frame_from_columns(row))  # noqa: E731
        plots = plotting.render(self.store.plots, row, per_pore, picture)      # the viewer shows them at once
        self.row = row
        self.state = replace(self._snapshot(row, per_pore, snapshot), plots=plots)
        corrections = [c.to_dict() for c in self.store._lists.get(self.image, [])]
        rejected = self.state.rejected
        self.store._submit(lambda: self._write(snapshot, rejected, row, per_pore, images, plots, corrections))
        return self.state

    def _evaluate(self, state: "_Labels"):
        top, bottom, left, right = self.top, self.bottom, self.left, self.right
        labels = state.labels[top:bottom, left:right]
        pixels = geometry.pixels(labels)
        ids = np.flatnonzero(np.bincount(pixels[1], minlength=1)).astype(np.int32)
        ctx = ImageContext(replace(self.info, offset_px=(top, left)), self.original[top:bottom, left:right],
                           self.roi[top:bottom, left:right], _number(self.row.get("pixel_size_um")),
                           frame_from_columns(self.row))
        pores = LabeledPores(labels, len(ids), pd.DataFrame({"pore_id": ids}),
                             state.rejected_labels[top:bottom, left:right], _rejected_table(state.rejected))
        pores.__dict__["pixels"] = pixels           # fill the cache: no second pass over the image
        for step in self.store.value_steps:
            _check_scale(step, ctx)
            pores = step.process(ctx, pores)
        results = {}
        for step in self.store.evaluators:
            _check_scale(step, ctx)
            results[step.name] = step.evaluate(ctx, pores)
        per_image, per_pore, images = _merge_results(results)
        self.store.layout.update(excel_layout(self.store.evaluators, results))
        keep = {k: v for k, v in self.row.items() if k in ("image", "path", "pixel_size_um", "scale_label",
                                                             "scale_bar_px", "scale_box_px", *FRAME_COLUMNS)}
        row = {**keep, "manual_corrections": len(self.store._lists.get(self.image, [])), **per_image}
        return row, per_pore, images

    def _write(self, state: "_Labels", rejected: pd.DataFrame, row: dict, per_pore: pd.DataFrame,
               images: dict, plots: dict[str, bytes], corrections: list[dict]) -> None:
        """Background job: image folder, then images.json, Excel and ID values."""
        store = self.store
        writer.write_pores(self.folder, self.original, state.labels, state.rejected_labels, rejected, per_pore,
                           images, store.config.output.save_full_overlay)
        writer.write_plots(self.folder, plots)
        writer.write_corrections(self.folder, self.image, corrections, sorted(state.added), sorted(state.removed),
                                 sorted(state.drawn))
        table = writer.read_table(store.results_dir / writer.IMAGES_JSON)
        rows = [row if r["image"] == self.image else r for r in table.to_dict("records")]
        per_image = pd.DataFrame(rows, columns=list(dict.fromkeys([*table.columns, *row])))
        errors = writer.read_table(store.results_dir / writer.ERRORS_JSON)
        id_values = None
        if (store.results_dir / writer.ID_VALUES_JSON).exists():       # only after a complete run
            id_values = _finalize(store.evaluators, per_image, store.results_dir)
        writer.write_summary(store.results_dir, self.info.image_id, per_image, errors, id_values, store.layout)
        if id_values is not None:
            write_id_plots(store.plots, per_image, store.results_dir)

    def _snapshot(self, row: dict, per_pore: pd.DataFrame, state: "_Labels | None" = None) -> State:
        state = state or self.current.copy()
        return State(state.labels, state.rejected_labels, _rejected_table(state.rejected), per_pore, row,
                     frozenset(state.added), frozenset(state.removed),
                     len(self.store._lists.get(self.image, [])), True)


def _rejected_table(rejected: dict) -> pd.DataFrame:
    ids = sorted(rejected)
    return pd.DataFrame({"pore_id": np.array(ids, dtype=np.int32),
                         "filter": [rejected[i][0] for i in ids], "reason": [rejected[i][1] for i in ids]})


# --- the corrections themselves (on _Labels, in place) ---

def _apply(state: _Labels, c: Correction, roi: np.ndarray) -> bool:
    """Apply one correction; False if it hit nothing (then nothing changed)."""
    if c.action == "restore":
        pore = _label_at(state.rejected_labels, c.x, c.y)
        if pore == 0:
            return False
        _move(state.rejected_labels, state.labels, pore)
        state.rejected.pop(pore, None)
        if pore in state.removed:
            state.removed.discard(pore)             # back to the automatic state
        else:
            state.added.add(pore)
        return True

    if c.action == "remove":
        pore = _label_at(state.labels, c.x, c.y)
        if pore == 0:
            return False
        if pore in state.drawn:
            _erase(state, pore)
        else:
            _move(state.labels, state.rejected_labels, pore)
            state.rejected[pore] = (MANUAL, "removed by hand")
            state.added.discard(pore)
            state.removed.add(pore)
        return True

    window, region = _rasterize(c.points, state.labels.shape)
    if c.action == "draw":
        region &= roi[window]
        if not region.any():
            return False
        new = max(int(state.labels.max()), int(state.rejected_labels.max())) + 1
        for source in (state.labels, state.rejected_labels):
            for old in np.unique(source[window][region]):
                if old:
                    pixels = source == old          # may reach beyond the outline
                    source[pixels] = 0
                    state.labels[pixels] = new      # the drawing swallows touched pores completely
                    state.rejected.pop(int(old), None)
                    for group in (state.added, state.removed, state.drawn):
                        group.discard(int(old))
        state.labels[window][region] = new
        state.added.add(new)
        state.drawn.add(new)
        return True

    if c.action == "area":
        inside = state.labels[window][region]
        if not inside.any():
            return False
        total = np.bincount(geometry.pixels(state.labels)[1])
        counts = np.bincount(inside, minlength=len(total))
        hit = [int(p) for p in np.flatnonzero((counts == total) & (total > 0)) if p > 0]
        labels, rejected_labels = state.labels[window], state.rejected_labels[window]   # views
        for pore in hit:                            # completely inside: all its pixels are in the window
            if pore in state.drawn:
                _erase(state, pore)
            else:
                _move(labels, rejected_labels, pore)
                state.rejected[pore] = (MANUAL, "removed by hand (area)")
                state.added.discard(pore)
                state.removed.add(pore)
        return bool(hit)
    raise ValueError(f"unknown correction {c.action!r}")


def _label_at(labels: np.ndarray, x: float, y: float, radius: int = SEARCH_RADIUS_PX) -> int:
    """The pore under the point, else the nearest one within `radius`; 0 = none."""
    height, width = labels.shape
    xi, yi = int(round(x)), int(round(y))
    if 0 <= yi < height and 0 <= xi < width and labels[yi, xi] > 0:
        return int(labels[yi, xi])
    y0, y1 = max(0, yi - radius), min(height, yi + radius + 1)
    x0, x1 = max(0, xi - radius), min(width, xi + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return 0
    window = labels[y0:y1, x0:x1]
    ys, xs = np.nonzero(window)
    if len(ys) == 0:
        return 0
    distance = (ys + y0 - y) ** 2 + (xs + x0 - x) ** 2
    nearest = int(np.argmin(distance))
    return int(window[ys[nearest], xs[nearest]]) if distance[nearest] <= radius ** 2 else 0


def _move(source: np.ndarray, target: np.ndarray, pore: int) -> None:
    pixels = source == pore
    target[pixels] = pore
    source[pixels] = 0


def _erase(state: _Labels, pore: int) -> None:
    """A drawn pore does not exist without its drawing: it disappears completely."""
    state.labels[state.labels == pore] = 0
    for group in (state.added, state.removed, state.drawn):
        group.discard(pore)


def _rasterize(points, shape: tuple[int, int]) -> tuple[tuple[slice, slice], np.ndarray]:
    """Outline as filled mask (pixels whose centre lies inside), only in its bounding window."""
    polygon = np.round(np.asarray(points, dtype=np.float64)).astype(np.int32)
    x0, y0 = np.clip(polygon.min(0), 0, None)
    x1, y1 = np.minimum(polygon.max(0) + 1, (shape[1], shape[0]))
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon - (x0, y0)], 1)
    return (slice(y0, y1), slice(x0, x1)), mask.view(bool)


# --- helpers ---

def _read(path: Path, flags: int) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return image


def _decode(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def _rejected_dict(table: pd.DataFrame) -> dict[int, tuple[str, str]]:
    return {int(i): (str(f), str(r)) for i, f, r in zip(table["pore_id"], table["filter"], table["reason"])}


def _image_id(image: str, path: str) -> str:
    """<ID>_<file stem> -> ID."""
    return image[:-len(Path(path).stem) - 1]


def _number(value) -> float | None:
    return None if value is None or (isinstance(value, float) and value != value) else float(value)
