"""Generate synthetic microstructure test images with ground truth.

Images are built as layers. Every pixel gets a class in the label image:
    0 = embedding resin, 1 = matrix, 2 = pore
Layers so far:
    1. matrix: bright grains + dark network / needle colonies
    2. pores:  any number of pore populations (gas / shrinkage) painted on top
    3. resin:  the sample is a rough, rotated rectangle in the embedding resin,
               with ragged edge, breakouts, shrinkage gap and Schliffrand
    4. stitching: panorama made of camera frames, each with its own look,
               hard right-angled seams and small notches
    5. scale bar: burnt-in ImageJ-style overlay (black bar + label in a white
               box) somewhere at the image border; value, unit and bar length
               are random and independent of um_per_px (test of the scale
               detection only)
Ground truth per image: class labels, resin mask, pore instances (+ type and
size in meta.json), the matrix phase mask, the stitching frame map and the
scale bar mask (+ bar length, label and position in meta.json). The labels
show what lies under the scale bar.

Rendering works in tiles: only small global data (grain seeds, needle lines,
pore masks, coarse noise lattices) is created per image, every tile is drawn
from it. Memory per tile does not depend on the image size and tiles join
without seams.

Usage:
    python generate.py [config.yaml]
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

RESIN, MATRIX, PORE = 0, 1, 2
CLASS_COLORS_BGR = np.array([[200, 120, 40],     # resin: blue
                             [200, 200, 200],    # matrix: light gray
                             [40, 40, 220]],     # pore: red
                            np.uint8)
TILE_PX = 512                                    # output tile edge


# ---------------------------------------------------------------- noise

def smooth_noise(h, w, scale_px, rng, octaves=3):
    """Fractal value noise for a small local window, zero mean, std 1."""
    out = np.zeros((h, w), np.float32)
    amp = 1.0
    for _ in range(octaves):
        s = max(scale_px, 2.0)
        gh, gw = int(math.ceil(h / s)) + 3, int(math.ceil(w / s)) + 3
        grid = rng.standard_normal((gh, gw)).astype(np.float32)
        up = cv2.resize(grid, (int(gw * s), int(gh * s)), interpolation=cv2.INTER_CUBIC)
        out += amp * up[:h, :w]
        scale_px /= 2.0
        amp *= 0.5
    return (out - out.mean()) / (out.std() + 1e-6)


class NoiseField:
    """Fractal value noise over a whole image, evaluated on demand for a window
    or at points. Same coordinates give the same values, so tiles join without
    seams. Normalised to zero mean, std 1. scale_px = largest feature size."""

    def __init__(self, h, w, scale_px, rng, octaves=3):
        self.layers = []
        amp = 1.0
        for _ in range(octaves):
            s = max(scale_px, 2.0)
            grid = rng.standard_normal((int(math.ceil(h / s)) + 4,
                                        int(math.ceil(w / s)) + 4)).astype(np.float32)
            self.layers.append((grid, s, amp))
            scale_px /= 2.0
            amp *= 0.5
        self.mean, self.std = 0.0, 1.0
        v = self.points(rng.uniform(0, h, 20000), rng.uniform(0, w, 20000))
        self.mean, self.std = float(v.mean()), float(v.std()) + 1e-6
        self.sorted = np.sort((v - self.mean) / self.std)

    def _eval(self, ys, xs):
        out = np.zeros(ys.shape, np.float32)
        for grid, s, amp in self.layers:
            out += amp * cv2.remap(grid, xs / s + 1, ys / s + 1, cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REFLECT)
        return (out - self.mean) / self.std

    def window(self, y0, x0, h, w):
        ys = np.repeat(np.arange(y0, y0 + h, dtype=np.float32)[:, None], w, 1)
        xs = np.repeat(np.arange(x0, x0 + w, dtype=np.float32)[None, :], h, 0)
        return self._eval(ys, xs)

    def points(self, ys, xs):
        # remap maps are limited to 32767 per axis -> fold long point lists
        n, cols = len(ys), 1024
        pad = -n % cols
        fold = lambda a: np.pad(np.asarray(a, np.float32), (0, pad)).reshape(-1, cols)
        return self._eval(fold(ys), fold(xs)).ravel()[:n]

    def quantile(self, q):
        return float(np.quantile(self.sorted, q))


# ---------------------------------------------------------------- matrix layer

def needle_segments(H, W, um, patch_at, m, rng):
    """Star-shaped, branched dark needles grouped in colonies.
    Returns an int32 array of lines (x1, y1, x2, y2, thickness)."""
    segs = []
    area_mm2 = H * W * (um / 1000.0) ** 2
    n_cand = rng.poisson(m["colonies_per_mm2"] * area_mm2 * 2.0)
    radius = m["colony_radius_um"] / um
    length = m["needle_length_um"] / um
    thick = max(1, round(m["needle_width_um"] / um))
    lo, hi = m["needles_per_colony"]

    # Place colonies preferably where the patch field is high
    cys, cxs = rng.uniform(-radius, H + radius, n_cand), rng.uniform(-radius, W + radius, n_cand)
    prob = np.clip(0.5 + 0.5 * patch_at(np.clip(cys, 0, H - 1), np.clip(cxs, 0, W - 1)), 0, 1)
    for cy, cx, p in zip(cys, cxs, prob):
        if rng.uniform() > p:
            continue
        for _ in range(rng.integers(lo, hi + 1)):
            sy, sx = cy + rng.normal(0, radius * 0.5), cx + rng.normal(0, radius * 0.5)
            ang = rng.uniform(0, 2 * math.pi)
            ln = length * max(0.2, 1 + rng.normal(0, m["needle_length_variation"]))
            ey, ex = sy + ln * math.sin(ang), sx + ln * math.cos(ang)
            segs.append((int(sx), int(sy), int(ex), int(ey), thick))
            if rng.uniform() < m["branch_probability"]:
                for _ in range(rng.integers(1, 3)):
                    t = rng.uniform(0.2, 0.9)
                    by, bx = sy + t * (ey - sy), sx + t * (ex - sx)
                    bang = ang + rng.choice([-1, 1]) * rng.uniform(0.6, 1.4)
                    bl = ln * m["branch_length_frac"] * rng.uniform(0.5, 1.5)
                    segs.append((int(bx), int(by), int(bx + bl * math.cos(bang)),
                                 int(by + bl * math.sin(bang)), max(1, thick - 1)))
    return np.array(segs, np.int32).reshape(-1, 5)


def matrix_colors(m, rng):
    """RGB of the bright grains and the dark network for one image. Each phase:
    its gray level times a random brightness factor, plus its color offset from
    gray (the beige) times a random tint factor shared by both phases."""
    tint = rng.uniform(*m["tint_scale"])
    out = []
    for key, scale in (("bright_rgb", "bright_scale"), ("dark_rgb", "dark_scale")):
        rgb = np.array(m[key], np.float32)
        gray = rgb.mean()
        out.append(gray * rng.uniform(*m[scale]) + (rgb - gray) * tint)
    bright, dark = out
    # Keep the phases apart: darken the network if it came out too close to the grains
    short = m["min_phase_contrast"] - (bright.mean() - dark.mean())
    if short > 0:
        dark -= short
    return np.clip(bright, 0, 255), np.clip(dark, 0, 255)


class Matrix:
    """Matrix layer of one image. Global features are created once; render_ss()
    draws any window of it at supersampled resolution, color() turns that into
    the final-resolution image."""

    def __init__(self, h, w, cfg, rng):
        m = self.m = cfg["matrix"]
        ss = cfg["image"]["supersample"]
        um = cfg["image"]["um_per_px"]
        um_ss = um / ss
        H, W = h * ss, w * ss

        # Large-scale patches: needle-rich / fine-grained vs. coarse bright regions
        self.patch = NoiseField(H, W, m["patch_size_um"] / um_ss, rng, octaves=2)
        k = 2 * m["patch_contrast"]
        patch_at = lambda ys, xs: np.tanh(self.patch.points(ys, xs) * k)   # -1 .. 1

        # Grain seeds: rejection sampling, grains smaller where the patch field is high
        d_mean = m["grain_size_um"] / um_ss
        var = m["grain_size_variation"]
        d_min = max(d_mean * (1 - var), d_mean * 0.3)
        dens_max = 1.0 / (0.866 * d_min ** 2)               # seeds per px² (hexagonal packing)
        n_max = int(dens_max * H * W * 1.2) + 1
        ys, xs = rng.uniform(0, H, n_max), rng.uniform(0, W, n_max)
        d_local = np.clip(d_mean * (1.0 - var * patch_at(ys, xs)), d_mean * 0.3, None)
        keep = rng.uniform(0, 1, n_max) < (1.0 / (0.866 * d_local ** 2)) / dens_max
        self.seed_y, self.seed_x = ys[keep].astype(np.int32), xs[keep].astype(np.int32)
        self.tint = rng.normal(0, m["grain_tint_sd"], int(keep.sum())).astype(np.float32)

        # Wavy, partly broken grain boundaries
        self.amp = m["boundary_waviness_um"] / um_ss
        wscale = m["boundary_waviness_scale_um"] / um_ss
        self.warp_x = NoiseField(H, W, wscale, rng, octaves=2)
        self.warp_y = NoiseField(H, W, wscale, rng, octaves=2)
        self.gaps = None
        if m["network_continuity"] < 1.0:
            self.gaps = NoiseField(H, W, m["network_gap_scale_um"] / um_ss, rng, octaves=2)
            self.gap_thr = self.gaps.quantile(1.0 - m["network_continuity"])
        self.width = max(1, round(m["network_width_um"] / um_ss))

        self.needles = needle_segments(H, W, um_ss, patch_at, m, rng)
        self.shading = NoiseField(h, w, m["shading_scale_um"] / um, rng)   # final resolution

        # Dark specks: single darker pixels (final resolution), softened by the optics
        n = rng.poisson(m["dark_specks_per_mm2"] * h * w * (um / 1000.0) ** 2)
        self.spy, self.spx = rng.integers(0, h, n), rng.integers(0, w, n)
        self.spv = rng.uniform(*m["dark_speck_strength"], n).astype(np.float32)

        # Colors of this image: brightness of each phase and the beige tint vary
        # per image; the dark network always stays darker than the grains
        self.bright, self.dark = matrix_colors(m, rng)

        # Window margins (supersampled px)
        self.m_inner = self.width + 2                       # boundary detection + thickening
        self.m_warp = int(4 * self.amp) + 2                 # displacement of the boundaries
        self.m_seed = int(3 * d_mean * (1 + var)) + 2       # Voronoi needs neighbouring seeds

    def render_ss(self, y0, x0, h, w):
        """Dark phase (bool) and global grain index (int32) of the supersampled
        window [y0:y0+h, x0:x0+w]. The window may reach beyond the image."""
        mi = self.m_inner
        ry0, rx0, rh, rw = y0 - mi, x0 - mi, h + 2 * mi, w + 2 * mi
        ma = self.m_warp + self.m_seed
        ay0, ax0, ah, aw = ry0 - ma, rx0 - ma, rh + 2 * ma, rw + 2 * ma

        # Voronoi grains from all seeds in the extended region
        idx = np.nonzero((self.seed_y >= ay0) & (self.seed_y < ay0 + ah)
                         & (self.seed_x >= ax0) & (self.seed_x < ax0 + aw))[0]
        if idx.size == 0:
            return np.zeros((h, w), bool), np.zeros((h, w), np.int32)
        sy, sx = self.seed_y[idx] - ay0, self.seed_x[idx] - ax0
        src = np.full((ah, aw), 255, np.uint8)
        src[sy, sx] = 0
        _, lab = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
        lut = np.zeros(int(lab.max()) + 1, np.int32)
        lut[lab[sy, sx]] = idx                              # local label -> global grain

        # Wavy boundaries: sample the labels at smoothly displaced positions
        gy = np.arange(ry0, ry0 + rh, dtype=np.float32)[:, None]
        gx = np.arange(rx0, rx0 + rw, dtype=np.float32)[None, :]
        map_x = gx + self.amp * self.warp_x.window(ry0, rx0, rh, rw) - ax0
        map_y = gy + self.amp * self.warp_y.window(ry0, rx0, rh, rw) - ay0
        labels = cv2.remap(lab.astype(np.float32), map_x, map_y, cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_REPLICATE)

        k3 = np.ones((3, 3), np.uint8)
        boundary = cv2.dilate(labels, k3) != cv2.erode(labels, k3)
        # Broken network: keep boundary only where a fine noise field is high
        if self.gaps is not None:
            boundary &= self.gaps.window(ry0, rx0, rh, rw) > self.gap_thr
        if self.width > 2:
            kw = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.width - 1, self.width - 1))
            boundary = cv2.dilate(boundary.astype(np.uint8), kw) > 0
        dark = boundary[mi:mi + h, mi:mi + w].copy()
        gid = lut[labels[mi:mi + h, mi:mi + w].astype(np.int32)]

        # Needles crossing the window
        s = self.needles
        if len(s):
            pad = int(s[:, 4].max())
            x1, y1, x2, y2 = s[:, 0] - x0, s[:, 1] - y0, s[:, 2] - x0, s[:, 3] - y0
            hit = ((np.maximum(x1, x2) >= -pad) & (np.minimum(x1, x2) < w + pad)
                   & (np.maximum(y1, y2) >= -pad) & (np.minimum(y1, y2) < h + pad))
            canvas = np.zeros((h, w), np.uint8)
            for a, b, c, d, t in zip(x1[hit], y1[hit], x2[hit], y2[hit], s[hit, 4]):
                cv2.line(canvas, (int(a), int(b)), (int(c), int(d)), 255, int(t))
            dark |= canvas > 0
        return dark, gid

    def color(self, dark_ss, gid_ss, y0, x0, h, w):
        """RGB float image and dark-phase coverage of the final-resolution window."""
        m = self.m
        # Coverage fraction gives anti-aliased phase edges
        frac = cv2.resize(dark_ss.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        tint = cv2.resize(self.tint[gid_ss], (w, h), interpolation=cv2.INTER_AREA)
        img = self.bright * (1 - frac[..., None]) + self.dark * frac[..., None]
        img += (tint * (1 - frac))[..., None]
        img += (m["shading_sd"] * self.shading.window(y0, x0, h, w))[..., None]
        sel = (self.spy >= y0) & (self.spy < y0 + h) & (self.spx >= x0) & (self.spx < x0 + w)
        if sel.any():
            py, px = self.spy[sel] - y0, self.spx[sel] - x0
            img[py, px] *= 1 - self.spv[sel][:, None]
        return img, frac


# ---------------------------------------------------------------- resin layer

class Specimen:
    """Sample outline inside the embedding resin: a rough rectangle, randomly
    rotated, with rounded corners, a wavy and ragged edge, small breakouts and a
    shrinkage gap between resin and sample. Geometry in supersampled px; signed
    distance < 0 = inside the sample."""

    def __init__(self, h, w, cfg, rng):
        e = self.e = cfg["resin"]
        self.enabled = e["enabled"]
        ss = self.ss = cfg["image"]["supersample"]
        um = cfg["image"]["um_per_px"]
        um_ss = self.um_ss = um / ss
        H, W = h * ss, w * ss
        if not self.enabled:
            self.area_px = float(h * w)
            return

        # Rectangle: size relative to the image, slightly off-centre, rotated
        self.hw = rng.uniform(*e["size_frac"]) * W / 2
        self.hh = rng.uniform(*e["size_frac"]) * H / 2
        j = e["center_jitter_frac"]
        self.cx, self.cy = W / 2 + rng.uniform(-j, j) * W, H / 2 + rng.uniform(-j, j) * H
        ang = math.radians(rng.uniform(-e["max_angle_deg"], e["max_angle_deg"]))
        self.cos, self.sin = math.cos(ang), math.sin(ang)
        self.r = min(e["corner_radius_um"] / um_ss, self.hw, self.hh)

        # Edge: long waves + small-scale jaggedness
        self.wave = NoiseField(H, W, e["edge_waviness_scale_um"] / um_ss, rng, octaves=2)
        self.wave_amp = e["edge_waviness_um"] / um_ss
        self.rough = NoiseField(H, W, e["edge_roughness_scale_um"] / um_ss, rng, octaves=3)
        self.rough_amp = e["edge_roughness_um"] / um_ss

        # Breakouts: small bites out of the edge, centres on the rectangle outline
        perim_mm = 4 * (self.hw + self.hh) * um_ss / 1000
        n = rng.poisson(e["breakouts_per_mm"] * perim_mm)
        t, side = rng.uniform(-1, 1, n), rng.integers(0, 4, n)
        u = np.where(side < 2, t * self.hw, np.where(side == 2, self.hw, -self.hw))
        v = np.where(side < 2, np.where(side == 0, self.hh, -self.hh), t * self.hh)
        self.bx = self.cx + u * self.cos - v * self.sin
        self.by = self.cy + u * self.sin + v * self.cos
        self.br = np.array([sample_size(rng, *e["breakout_radius_um"], "loguniform")
                            for _ in range(n)]).reshape(-1) / um_ss

        # Shrinkage gap between resin and sample, width varies along the edge
        self.gap_nf = NoiseField(H, W, e["gap_scale_um"] / um_ss, rng, octaves=2)
        self.gap_mean, self.gap_sd = e["gap_mean_um"] / um_ss, e["gap_sd_um"] / um_ss

        # Resin appearance: shading, filler particles (circle outlines), dust specks
        self.shading = NoiseField(h, w, e["resin_shading_scale_um"] / um, rng)
        area_mm2 = h * w * (um / 1000) ** 2
        nf = rng.poisson(e["filler_per_mm2"] * area_mm2)
        self.fx, self.fy = rng.uniform(0, W, nf), rng.uniform(0, H, nf)
        self.fr = rng.uniform(*e["filler_diameter_um"], nf) / 2 / um_ss
        self.ring = max(1, round(e["filler_ring_um"] / um_ss))
        nsp = rng.poisson(e["specks_per_mm2"] * area_mm2)
        self.spy, self.spx = rng.integers(0, h, nsp), rng.integers(0, w, nsp)
        self.spv = rng.uniform(0.3, 1.0, nsp).astype(np.float32)

        # Sample area (final px) for the porosity budget, from a point grid
        step = max(1, int(math.sqrt(h * w / 200000)))
        gy, gx = np.mgrid[0:h:step, 0:w:step]
        sd = self.sd_points((gy.ravel() + 0.5) * ss, (gx.ravel() + 0.5) * ss)
        self.area_px = float((sd < 0).mean()) * h * w

    def _sd(self, gy, gx, wave, rough, near):
        dx, dy = gx - self.cx, gy - self.cy
        u = np.abs(dx * self.cos + dy * self.sin) - (self.hw - self.r)
        v = np.abs(-dx * self.sin + dy * self.cos) - (self.hh - self.r)
        sd = (np.sqrt(np.maximum(u, 0) ** 2 + np.maximum(v, 0) ** 2)
              + np.minimum(np.maximum(u, v), 0) - self.r)       # rounded rectangle
        sd = sd + self.wave_amp * wave
        for bx, by, br in zip(self.bx[near], self.by[near], self.br[near]):
            sd = np.maximum(sd, br - np.sqrt((gx - bx) ** 2 + (gy - by) ** 2))
        return (sd + self.rough_amp * rough).astype(np.float32)

    def sd_window(self, y0, x0, h, w):
        """Signed distance to the sample edge (ss px, < 0 inside) of a ss window."""
        if not self.enabled:
            return np.full((h, w), -1e6, np.float32)
        gy = np.arange(y0, y0 + h, dtype=np.float32)[:, None]
        gx = np.arange(x0, x0 + w, dtype=np.float32)[None, :]
        near = ((self.bx + self.br > x0) & (self.bx - self.br < x0 + w)
                & (self.by + self.br > y0) & (self.by - self.br < y0 + h))
        return self._sd(gy, gx, self.wave.window(y0, x0, h, w),
                        self.rough.window(y0, x0, h, w), near)

    def sd_points(self, ys, xs):
        ys, xs = np.asarray(ys, np.float32), np.asarray(xs, np.float32)
        return self._sd(ys, xs, self.wave.points(ys, xs), self.rough.points(ys, xs),
                        np.ones(len(self.bx), bool))

    def gap_window(self, sd, y0, x0, h, w):
        """Shrinkage gap (bool) of a ss window: resin side, next to the edge."""
        width = self.gap_mean + self.gap_sd * self.gap_nf.window(y0, x0, h, w)
        return (sd >= 0) & (sd < width)

    def resin_color(self, f_gap, y0, x0, h, w):
        """RGB float image of the resin for a final-resolution window."""
        e, ss = self.e, self.ss
        img = np.empty((h, w, 3), np.float32)
        img[:] = e["resin_rgb"]
        img += (e["resin_shading_sd"] * self.shading.window(y0, x0, h, w))[..., None]

        Y0, X0, Hs, Ws = y0 * ss, x0 * ss, h * ss, w * ss
        hit = ((self.fx + self.fr > X0) & (self.fx - self.fr < X0 + Ws)
               & (self.fy + self.fr > Y0) & (self.fy - self.fr < Y0 + Hs))
        if hit.any():
            ring = np.zeros((Hs, Ws), np.uint8)
            for cx, cy, r in zip(self.fx[hit], self.fy[hit], self.fr[hit]):
                cv2.circle(ring, (int(cx - X0), int(cy - Y0)), int(r), 255, self.ring)
            fr = cv2.resize(ring.astype(np.float32) / 255, (w, h),
                            interpolation=cv2.INTER_AREA)[..., None]
            img = img * (1 - fr) + np.array(e["filler_ring_rgb"], np.float32) * fr

        fg = f_gap[..., None]
        img = img * (1 - fg) + np.array(e["gap_rgb"], np.float32) * fg

        sel = (self.spy >= y0) & (self.spy < y0 + h) & (self.spx >= x0) & (self.spx < x0 + w)
        if sel.any():
            py, px, v = self.spy[sel] - y0, self.spx[sel] - x0, self.spv[sel][:, None]
            img[py, px] = img[py, px] * (1 - v) + np.array(e["speck_rgb"], np.float32) * v
        return img

    def edge_zone(self, col, sd):
        """Schliffrand: edge rounding from grinding makes the outermost rim of the
        sample slightly darker and out of focus. sd: signed distance (ss px) at
        final resolution."""
        e = self.e
        if e["edge_zone_um"] <= 0:
            return col
        d = np.clip(-sd, 0, None) * self.um_ss                   # µm inside the sample
        k = np.exp(-d / e["edge_zone_um"])[..., None]
        col = col * (1 - e["edge_darken"] * k)
        if e["edge_blur_px"] > 0:
            col = col + (cv2.GaussianBlur(col, (0, 0), e["edge_blur_px"]) - col) * k
        return col


# ---------------------------------------------------------------- pore layer

def sample_size(rng, lo, hi, distribution):
    if distribution == "uniform":
        return rng.uniform(lo, hi)
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))     # loguniform


def gas_pore(d_px, p, rng):
    """Round / elliptical pore with slightly rough outline, optional satellites.
    Returns a local boolean mask whose centre is the pore centre."""
    r = d_px / 2
    sat_r = []
    if rng.uniform() < p["satellite_probability"]:
        lo, hi = p["satellites"]
        sat_r = [r * rng.uniform(*p["satellite_size_frac"]) for _ in range(rng.integers(lo, hi + 1))]
    ar = rng.uniform(*p["aspect_ratio"])
    half = int(math.ceil(r * math.sqrt(ar) + 2 * max(sat_r, default=0) * (1 + p["satellite_gap"]))) + 2
    mask = np.zeros((2 * half + 1, 2 * half + 1), np.uint8)

    def blob(cy, cx, rad, aspect, rot):
        t = np.linspace(0, 2 * math.pi, 96, endpoint=False)
        rough = np.zeros_like(t)
        for k in range(2, 9):              # low harmonics = gentle bumps
            rough += rng.normal(0, 1) / k * np.cos(k * t + rng.uniform(0, 2 * math.pi))
        rr = rad * (1 + p["roughness"] * rough)
        a, b = math.sqrt(aspect), 1 / math.sqrt(aspect)
        x, y = rr * a * np.cos(t), rr * b * np.sin(t)
        xs = cx + x * math.cos(rot) - y * math.sin(rot)
        ys = cy + x * math.sin(rot) + y * math.cos(rot)
        pts = np.round(np.stack([xs, ys], 1) * 16).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1, cv2.LINE_8, shift=4)

    blob(half, half, r, ar, rng.uniform(0, math.pi))
    for sr in sat_r:
        ang = rng.uniform(0, 2 * math.pi)
        dist = (r + sr) * (1 + p["satellite_gap"])
        blob(half + dist * math.sin(ang), half + dist * math.cos(ang), sr,
             rng.uniform(*p["aspect_ratio"]), rng.uniform(0, math.pi))
    return mask > 0


def shrinkage_pore(d_px, p, rng, dark_local):
    """Irregular, ragged shrinkage cavity (Lunker): noisy blob in an elliptical
    envelope that grows along the dark interdendritic phase and breaks up into
    fragments at its rim. dark_local: dark phase around the centre (same shape)."""
    n = dark_local.shape[0]
    half = n // 2
    r = d_px / 2
    ar = rng.uniform(*p["aspect_ratio"])
    a, b = r * math.sqrt(ar), r / math.sqrt(ar)
    rot = rng.uniform(0, math.pi)
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    xr = xx * math.cos(rot) + yy * math.sin(rot)
    yr = -xx * math.sin(rot) + yy * math.cos(rot)
    envelope = 1.0 - ((xr / a) ** 2 + (yr / b) ** 2)          # 1 at centre, 0 at rim
    noise = smooth_noise(n, n, max(2.0, d_px * p["detail_scale"]), rng, octaves=3)
    value = (envelope + p["raggedness"] * 0.5 * noise
             + p["follow_matrix"] * (dark_local.astype(np.float32) - 0.5))
    return value > 0


def pore_density(h, w, cfg, rng):
    """Where pores occur: a smooth random field on a coarse grid, so pores gather
    in some regions and leave others almost free. Returns (cell size in final px,
    log-density per cell)."""
    d = cfg["pores"]["distribution"]
    scale = d["scale_um"] / cfg["image"]["um_per_px"]
    cell = max(1, int(scale / 8))
    gy, gx = np.mgrid[0:h:cell, 0:w:cell]
    field = NoiseField(h, w, scale, rng, octaves=2)
    logd = d["contrast"] * field.points(gy.ravel() + cell / 2, gx.ravel() + cell / 2)
    return cell, logd.reshape(gy.shape)


def place_pores(h, w, cfg, matrix, specimen, rng):
    """Pore layer: place the pores of all populations, only inside the sample.
    Returns a list of (y0, x0, local mask, type id) in supersampled coordinates
    and the target porosity (None = absolute counts); type id = index of the
    population + 1. Porosity and counts refer to the sample area."""
    pc = cfg["pores"]
    ss = cfg["image"]["supersample"]
    um_ss = cfg["image"]["um_per_px"] / ss
    H, W = h * ss, w * ss
    occupied = np.zeros((h, w), bool)        # final resolution: pores must not merge there
    gap = round(pc["min_gap_um"] / um_ss)
    gap_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1)) if gap else None
    pores, centres = [], []

    pops = [(t_id, p) for t_id, p in enumerate(pc["populations"], start=1)
            if p.get("enabled", True)]
    cell, logd = pore_density(h, w, cfg, rng)
    cols = logd.shape[1]
    site_cdf = {}                            # per population: density ** density_weight
    for t_id, p in pops:
        site_cdf[t_id] = np.cumsum(np.exp(p["density_weight"] * (logd - logd.max())).ravel())

    def try_place(t_id, p, max_area):
        """Place one pore; returns its area in final px or None (no free spot)."""
        for _attempt in range(30):
            d_px = sample_size(rng, *p["diameter_um"], p["distribution"]) / um_ss
            if math.pi / 4 * (d_px / ss) ** 2 * 0.3 > max_area:
                continue                     # clearly too large for the remaining budget
            # Position: near an existing pore (cluster) or from the density field
            if centres and rng.uniform() < p["cluster_fraction"]:
                cy, cx = centres[rng.integers(len(centres))]
                rad = p["cluster_radius_um"] / um_ss
                cy, cx = cy + rng.normal(0, rad), cx + rng.normal(0, rad)
            else:
                cdf = site_cdf[t_id]
                k = int(np.searchsorted(cdf, rng.uniform() * cdf[-1]))
                cy = (k // cols + rng.uniform()) * cell * ss
                cx = (k % cols + rng.uniform()) * cell * ss
            cy, cx = int(cy), int(cx)
            if not (0 <= cy < H and 0 <= cx < W):
                continue

            if p["shape"] == "gas":
                local = gas_pore(d_px, p, rng)
            else:
                half = int(math.ceil(d_px / 2 * math.sqrt(p["aspect_ratio"][1]) * 1.5)) + 2
                dark_local, _ = matrix.render_ss(cy - half, cx - half, 2 * half + 1, 2 * half + 1)
                local = shrinkage_pore(d_px, p, rng, dark_local)

            # Clip the local window to the image (pores may be cut by the border)
            half = local.shape[0] // 2
            y0, x0 = cy - half, cx - half
            sy0, sx0 = max(y0, 0), max(x0, 0)
            sy1, sx1 = min(y0 + local.shape[0], H), min(x0 + local.shape[1], W)
            local = local[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
            area = local.sum() / ss ** 2
            if area == 0 or area > max_area:
                continue

            # Inside the sample and no contact with other pores (min_gap);
            # otherwise try a new spot
            test = local
            if gap:
                test = cv2.dilate(np.pad(local, gap).astype(np.uint8), gap_k) > 0
            if specimen.enabled:
                sd = specimen.sd_window(sy0 - gap, sx0 - gap, *test.shape)
                if (sd[test] >= 0).any():
                    continue
            ty, tx = np.nonzero(test)
            ty, tx = (ty + sy0 - gap) // ss, (tx + sx0 - gap) // ss
            inside = (ty >= 0) & (ty < h) & (tx >= 0) & (tx < w)
            if occupied[ty[inside], tx[inside]].any():
                continue
            py, px = np.nonzero(local)
            occupied[(py + sy0) // ss, (px + sx0) // ss] = True
            pores.append((sy0, sx0, local, t_id))
            centres.append((cy, cx))
            return area
        return None

    target = pc["target_porosity"]
    if target is None:
        # Absolute counts per population
        area_mm2 = specimen.area_px * (cfg["image"]["um_per_px"] / 1000.0) ** 2
        for t_id, p in pops:
            for _ in range(rng.poisson(p["count_per_mm2"] * area_mm2)):
                try_place(t_id, p, math.inf)
        return pores, None

    # Target porosity: pick populations in the ratio of count_per_mm2 until it is reached
    target = rng.uniform(*target)
    budget = target * specimen.area_px
    weights = np.array([p["count_per_mm2"] for _, p in pops], float)
    weights /= weights.sum()
    total, misses = 0.0, 0
    while total < budget and misses < 200:
        t_id, p = pops[rng.choice(len(pops), p=weights)]
        a = try_place(t_id, p, budget * 1.02 - total)
        if a is None:
            misses += 1
        else:
            total, misses = total + a, 0
    return pores, target


def paint_pores(pores, boxes, y0, x0, h, w):
    """Pore mask and type map of a supersampled window."""
    mask = np.zeros((h, w), bool)
    types = np.zeros((h, w), np.uint8)
    hit = ((boxes[:, 2] > y0) & (boxes[:, 0] < y0 + h)
           & (boxes[:, 3] > x0) & (boxes[:, 1] < x0 + w))
    for i in np.nonzero(hit)[0]:
        py, px, local, t_id = pores[i]
        ly0, lx0 = max(y0 - py, 0), max(x0 - px, 0)
        ly1, lx1 = min(y0 + h - py, local.shape[0]), min(x0 + w - px, local.shape[1])
        part = local[ly0:ly1, lx0:lx1]
        wy, wx = py + ly0 - y0, px + lx0 - x0
        mask[wy:wy + part.shape[0], wx:wx + part.shape[1]] |= part
        types[wy:wy + part.shape[0], wx:wx + part.shape[1]][part] = t_id
    return mask, types


def pore_instances(pore_mask, type_map, cfg):
    """Ground-truth pores as a perfect detector sees them: connected components
    of the final pore mask (8-connectivity), with generator type and size."""
    um = cfg["image"]["um_per_px"]
    names = [p["name"] for p in cfg["pores"]["populations"]]
    n, inst, stats, cents = cv2.connectedComponentsWithStats(pore_mask.astype(np.uint8),
                                                             connectivity=8)
    info = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        box = inst[y:y + bh, x:x + bw] == i
        t = np.bincount(type_map[y:y + bh, x:x + bw][box], minlength=len(names) + 1)[1:]
        info.append({"id": i, "type": names[int(t.argmax())] if t.any() else None,
                     "area_px": area, "area_um2": round(area * um * um, 2),
                     "eq_diameter_um": round(2 * math.sqrt(area / math.pi) * um, 2),
                     "centroid_xy": [round(float(cents[i][0]), 1), round(float(cents[i][1]), 1)],
                     "bbox_xywh": [x, y, bw, bh]})
    if n - 1 > 65535:
        print(f"  warning: {n - 1} pores, instance PNG (16 bit) overflows")
    return inst.astype(np.uint16), info


# ---------------------------------------------------------------- stitching layer

def auto_grid(h, w, n, frame_aspect):
    """cols x rows = n whose cells are closest to the camera frame aspect."""
    best = None
    for rows in range(1, n + 1):
        if n % rows:
            continue
        cols = n // rows
        err = abs(math.log((w / cols) / (h / rows) / frame_aspect))
        if best is None or err < best[0]:
            best = (err, cols, rows)
    return best[1], best[2]


class Stitching:
    """Panorama look: the microscope software stitches the image from camera
    frames. Each frame has its own exposure, color, contrast, vignetting,
    illumination tilt and focus. Seams are hard and right-angled; the vertical
    seams are staggered per row, and sometimes a small rectangle of one frame
    juts into its neighbour (keeping the look of its own frame)."""

    def __init__(self, h, w, cfg, rng):
        s = self.s = cfg["stitching"]
        self.enabled = s["enabled"]
        if not self.enabled:
            return
        cols, rows = (auto_grid(h, w, s["frames"], s["frame_aspect"]) if s["grid"] == "auto"
                      else s["grid"])
        self.cols, self.rows = cols, rows
        ch, cw = h / rows, w / cols
        j = s["seam_jitter_frac"]

        # Seams: horizontal cuts across the full width, vertical cuts per row
        self.row_cuts = np.sort([int((r + rng.uniform(-j, j)) * ch) for r in range(1, rows)])
        self.col_cuts = [np.sort([int((c + rng.uniform(-j, j)) * cw) for c in range(1, cols)])
                         for _ in range(rows)]
        ry = [0, *self.row_cuts, h]
        self.cells = []                      # nominal rectangle of every frame
        for r in range(rows):
            cx = [0, *self.col_cuts[r], w]
            for c in range(cols):
                self.cells.append((ry[r], cx[c], ry[r + 1], cx[c + 1]))

        # Notches: small rectangles of one frame jutting into the neighbour
        self.notches = []                    # (y0, x0, y1, x1, frame)
        lo_l, hi_l = s["notch_length_frac"]
        lo_d, hi_d = s["notch_depth_frac"]
        for r in range(rows):                # vertical seams
            for k, x in enumerate(self.col_cuts[r]):
                a, b = r * cols + k, r * cols + k + 1
                for _ in range(rng.poisson(s["notches_per_seam"])):
                    ln = int(rng.uniform(lo_l, hi_l) * ch)
                    d = max(1, int(rng.uniform(lo_d, hi_d) * cw))
                    y = int(rng.uniform(ry[r], max(ry[r], ry[r + 1] - ln)))
                    if rng.uniform() < 0.5:
                        self.notches.append((y, x, y + ln, x + d, a))
                    else:
                        self.notches.append((y, x - d, y + ln, x, b))
        for r, y in enumerate(self.row_cuts):  # horizontal seams
            for _ in range(rng.poisson(s["notches_per_seam"] * cols)):
                ln = int(rng.uniform(lo_l, hi_l) * cw)
                d = max(1, int(rng.uniform(lo_d, hi_d) * ch))
                x = int(rng.uniform(0, max(1, w - ln)))
                xm = x + ln // 2
                a = r * cols + int(np.searchsorted(self.col_cuts[r], xm, side="right"))
                b = (r + 1) * cols + int(np.searchsorted(self.col_cuts[r + 1], xm, side="right"))
                if rng.uniform() < 0.5:
                    self.notches.append((y, x, y + d, x + ln, a))
                else:
                    self.notches.append((y - d, x, y, x + ln, b))

        # Look of every frame
        n = rows * cols
        self.gain = 1 + rng.normal(0, s["brightness_sd"], n)
        self.tint = 1 + rng.normal(0, s["color_sd"], (n, 3))
        self.contrast = 1 + rng.normal(0, s["contrast_sd"], n)
        self.vignetting = rng.uniform(*s["vignetting"], n)
        self.gradient = rng.normal(0, s["gradient_sd"], (n, 2))
        self.blur = rng.uniform(*s["blur_extra_px"], n)

    def frame_map(self, y0, x0, h, w):
        """Frame index of every pixel of a final-resolution window."""
        ys, xs = np.arange(y0, y0 + h), np.arange(x0, x0 + w)
        row = np.searchsorted(self.row_cuts, ys, side="right")
        fmap = np.empty((h, w), np.int32)
        for r in np.unique(row):
            fmap[row == r] = r * self.cols + np.searchsorted(self.col_cuts[r], xs, side="right")
        for ny0, nx0, ny1, nx1, f in self.notches:
            a0, b0 = max(ny0, y0), max(nx0, x0)
            a1, b1 = min(ny1, y0 + h), min(nx1, x0 + w)
            if a0 < a1 and b0 < b1:
                fmap[a0 - y0:a1 - y0, b0 - x0:b1 - x0] = f
        return fmap

    def apply(self, col, y0, x0, h, w):
        """Give every pixel the look of its frame. Returns image and frame map."""
        fmap = self.frame_map(y0, x0, h, w)
        out = np.empty_like(col)
        gy = np.arange(y0, y0 + h, dtype=np.float32)[:, None]
        gx = np.arange(x0, x0 + w, dtype=np.float32)[None, :]
        for f in np.unique(fmap):
            m = fmap == f
            c = col if self.blur[f] < 0.05 else cv2.GaussianBlur(col, (0, 0), float(self.blur[f]))
            cy0, cx0, cy1, cx1 = self.cells[f]
            ry = (gy - (cy0 + cy1) / 2) / max((cy1 - cy0) / 2, 1)   # -1 .. 1 inside the frame
            rx = (gx - (cx0 + cx1) / 2) / max((cx1 - cx0) / 2, 1)
            light = (self.gain[f] * (1 - self.vignetting[f] * (rx ** 2 + ry ** 2) / 2)
                     * (1 + self.gradient[f, 0] * rx + self.gradient[f, 1] * ry))
            v = ((c - 128) * self.contrast[f] + 128) * light[..., None] * self.tint[f]
            out[m] = v[m]
        return out, fmap

    def info(self):
        if not self.enabled:
            return {}
        return {"grid_cols_rows": [self.cols, self.rows],
                "frames": [{"id": f, "cell_y0x0y1x1": [int(v) for v in self.cells[f]],
                            "gain": round(float(self.gain[f]), 4),
                            "tint_rgb": [round(float(v), 4) for v in self.tint[f]],
                            "contrast": round(float(self.contrast[f]), 4),
                            "vignetting": round(float(self.vignetting[f]), 4),
                            "gradient_xy": [round(float(v), 4) for v in self.gradient[f]],
                            "blur_extra_px": round(float(self.blur[f]), 3)}
                           for f in range(len(self.cells))],
                "notches_y0x0y1x1_frame": [[int(v) for v in n] for n in self.notches]}


# ---------------------------------------------------------------- scale bar layer

UNIT_UM = {"nm": 1e-3, "µm": 1.0, "mm": 1e3, "cm": 1e4, "m": 1e6}


def scale_label(sb, rng):
    """Random value + unit, either a round value or a decimal one as ImageJ
    writes it for a given bar width. Returns (value, unit, text)."""
    units = list(sb["units"])
    weights = np.array([sb["units"][u] for u in units], float)
    unit = units[rng.choice(len(units), p=weights / weights.sum())]
    if rng.uniform() < sb["round_fraction"]:
        value = float(rng.choice(sb["round_values"]))
        text = f"{value:g} {unit}"
    else:
        value = sample_size(rng, *sb["decimal_range"], "loguniform")
        digits = int(rng.integers(sb["decimals"][0], sb["decimals"][1] + 1))
        text = f"{value:.{digits}f} {unit}"
        value = float(text.split()[0])                  # the value as written
    return value, unit, text


def scale_bar(img, cfg, rng):
    """Burn an ImageJ-style scale bar into the BGR image (in place): white box,
    solid black bar on top, label centred below. The box sits at a random place
    along a random image border. Returns (box mask, info) or (None, None)."""
    sb = cfg["scale_bar"]
    if not sb["enabled"] or rng.uniform() >= sb["probability"]:
        return None, None
    h, w = img.shape[:2]
    value, unit, text = scale_label(sb, rng)
    font_file = str(rng.choice(sb["fonts"]))
    font = ImageFont.truetype(font_file, int(rng.integers(*sb["font_size_px"], endpoint=True)))
    bar_h = int(rng.integers(*sb["bar_height_px"], endpoint=True))
    pad = int(rng.integers(*sb["box_padding_px"], endpoint=True))
    gap = int(rng.integers(*sb["text_gap_px"], endpoint=True))
    tx0, ty0, tx1, ty1 = font.getbbox(text)
    tw, th = tx1 - tx0, ty1 - ty0

    # Bar length: random, the box (bar or label, whichever is wider) must fit
    length = int(rng.integers(*sb["length_px"], endpoint=True))
    length = min(length, w - 2 * pad - 2)
    bw, bh = max(length, tw) + 2 * pad, pad + bar_h + gap + th + pad
    if bw > w or bh > h:
        print(f"  warning: scale bar does not fit into {w}x{h} px, skipped")
        return None, None

    # Position: one border, anywhere along it, a random distance away from it
    off = int(rng.integers(*sb["border_offset_px"], endpoint=True))
    side = rng.integers(4)                              # 0 top, 1 bottom, 2 left, 3 right
    if side < 2:
        x = int(rng.integers(0, w - bw + 1))
        y = min(off, h - bh) if side == 0 else max(h - bh - off, 0)
    else:
        y = int(rng.integers(0, h - bh + 1))
        x = min(off, w - bw) if side == 2 else max(w - bw - off, 0)

    # Draw: white box, bar centred in it, label centred below the bar
    patch = Image.new("RGB", (bw, bh), (255, 255, 255))
    draw = ImageDraw.Draw(patch)
    bx = (bw - length) // 2
    draw.rectangle([bx, pad, bx + length - 1, pad + bar_h - 1], fill=(0, 0, 0))
    draw.text(((bw - tw) // 2 - tx0, pad + bar_h + gap - ty0), text, font=font, fill=(0, 0, 0))
    img[y:y + bh, x:x + bw] = np.asarray(patch)[..., ::-1]

    # JPEG artefacts around the overlay (box no longer exactly 255, colour fringes)
    quality = None
    if rng.uniform() < sb["jpeg_probability"]:
        quality = int(rng.integers(*sb["jpeg_quality"], endpoint=True))
        m = 8
        ry0, rx0 = max(y - m, 0), max(x - m, 0)
        ry1, rx1 = min(y + bh + m, h), min(x + bw + m, w)
        _, buf = cv2.imencode(".jpg", img[ry0:ry1, rx0:rx1], [cv2.IMWRITE_JPEG_QUALITY, quality])
        img[ry0:ry1, rx0:rx1] = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    mask = np.zeros((h, w), bool)
    mask[y:y + bh, x:x + bw] = True
    info = {"text": text, "value": value, "unit": unit,
            "value_um": round(value * UNIT_UM[unit], 10),
            "bar_length_px": length,
            "um_per_px": round(value * UNIT_UM[unit] / length, 10),
            "bar_xywh": [x + bx, y + pad, length, bar_h],
            "box_xywh": [x, y, bw, bh],
            "border": ["top", "bottom", "left", "right"][side],
            "font": Path(font_file).name, "font_size_px": font.size,
            "jpeg_quality": quality}
    return mask, info


# ---------------------------------------------------------------- image

def optics(img, cfg):
    """Optical blur of the microscope (before stitching: seams stay sharp)."""
    s = cfg["matrix"]["blur_sigma_px"]
    return cv2.GaussianBlur(img, (0, 0), s) if s > 0 else img


def sensor(img, cfg, rng):
    """Sensor noise, to uint8 BGR."""
    img = img + rng.normal(0, cfg["matrix"]["noise_sd"], img.shape).astype(np.float32)
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def render_image(h, w, cfg, rng):
    """All layers, tile by tile. Returns a dict: BGR image, dark matrix phase,
    pore mask, pore type map, resin mask, stitching frame map (all final
    resolution), target porosity, specimen and stitching."""
    ss = cfg["image"]["supersample"]
    pc = cfg["pores"]
    matrix = Matrix(h, w, cfg, rng)
    specimen = Specimen(h, w, cfg, rng)
    stitching = Stitching(h, w, cfg, rng)
    pores, target = (place_pores(h, w, cfg, matrix, specimen, rng) if pc["enabled"]
                     else ([], None))
    boxes = np.array([(py, px, py + l.shape[0], px + l.shape[1]) for py, px, l, _ in pores],
                     np.int64).reshape(-1, 4)
    pore_color = np.array(pc["pore_rgb"], np.float32)
    pore_shade = NoiseField(h, w, 10.0, rng)

    img = np.empty((h, w, 3), np.uint8)
    phase = np.empty((h, w), bool)
    pore_mask = np.zeros((h, w), bool)
    types = np.zeros((h, w), np.uint8)
    resin_mask = np.zeros((h, w), bool)
    frames = np.zeros((h, w), np.uint8)
    # Tile margin for the blurs (they add up: edge zone -> optics -> frame focus)
    blur = cfg["matrix"]["blur_sigma_px"]
    if specimen.enabled:
        blur += cfg["resin"]["edge_blur_px"]
    if stitching.enabled:
        blur += cfg["stitching"]["blur_extra_px"][1]
    mb = int(math.ceil(3 * blur)) + 1 if blur > 0 else 0

    for ty in range(0, h, TILE_PX):
        for tx in range(0, w, TILE_PX):
            th, tw = min(TILE_PX, h - ty), min(TILE_PX, w - tx)
            wy, wx, wh, ww = ty - mb, tx - mb, th + 2 * mb, tw + 2 * mb
            dark_ss, gid_ss = matrix.render_ss(wy * ss, wx * ss, wh * ss, ww * ss)
            col, frac = matrix.color(dark_ss, gid_ss, wy, wx, wh, ww)
            c = (slice(mb, mb + th), slice(mb, mb + tw))
            if pores:
                pm_ss, pt_ss = paint_pores(pores, boxes, wy * ss, wx * ss, wh * ss, ww * ss)
                pf = cv2.resize(pm_ss.astype(np.float32), (ww, wh), interpolation=cv2.INTER_AREA)
                shade = pc["pore_shading_sd"] * pore_shade.window(wy, wx, wh, ww)
                col = col * (1 - pf[..., None]) + (pore_color + shade[..., None]) * pf[..., None]
                pore_mask[ty:ty + th, tx:tx + tw] = (pf > 0.5)[c]
                # Max per ss block: tiny pores keep their type (pores never overlap)
                types[ty:ty + th, tx:tx + tw] = pt_ss.reshape(wh, ss, ww, ss).max((1, 3))[c]
            if specimen.enabled:
                Y0, X0, Hs, Ws = wy * ss, wx * ss, wh * ss, ww * ss
                sd = specimen.sd_window(Y0, X0, Hs, Ws)
                area = lambda a: cv2.resize(a, (ww, wh), interpolation=cv2.INTER_AREA)
                f_spec = area((sd < 0).astype(np.float32))
                f_gap = area(specimen.gap_window(sd, Y0, X0, Hs, Ws).astype(np.float32))
                col = specimen.edge_zone(col, area(sd))
                resin = specimen.resin_color(f_gap, wy, wx, wh, ww)
                fs = f_spec[..., None]
                col = resin * (1 - fs) + col * fs
                resin_mask[ty:ty + th, tx:tx + tw] = (f_spec <= 0.5)[c]
            col = optics(col, cfg)
            if stitching.enabled:
                col, fmap = stitching.apply(col, wy, wx, wh, ww)
                frames[ty:ty + th, tx:tx + tw] = fmap[c]
            img[ty:ty + th, tx:tx + tw] = sensor(col, cfg, rng)[c]
            phase[ty:ty + th, tx:tx + tw] = (frac > 0.5)[c]
    phase &= ~resin_mask
    return {"img": img, "phase": phase, "pores": pore_mask, "types": types,
            "resin": resin_mask, "frames": frames, "target": target,
            "matrix": matrix, "specimen": specimen, "stitching": stitching}


def vary(cfg, rng):
    """Config of one image: every entry of cfg["variation"] ("section.key":
    [min, max]) is drawn uniformly and replaces that value. A list value (e.g.
    an RGB color) gets the drawn value in every element. Returns (config,
    drawn values)."""
    img_cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
    drawn = {}
    for path, (lo, hi) in (cfg.get("variation") or {}).items():
        section, key = path.split(".")
        v = round(float(rng.uniform(lo, hi)), 4)
        old = img_cfg[section][key]
        img_cfg[section][key] = [v] * len(old) if isinstance(old, list) else v
        drawn[path] = v
    return img_cfg, drawn


def write_png(path, img):
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(str(path))     # works with non-ASCII paths on Windows


def main():
    here = Path(__file__).resolve().parent
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "synth_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    run = cfg["run"]
    rng = np.random.default_rng(run["seed"])
    # own streams: scale bar and parameter variation do not change the other draws
    bar_rng, var_rng = rng.spawn(2)

    out = here / run["out_dir"]
    img_dir, gt_dir = out / run["id"], out / f"{run['id']}_gt"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    meta = {"config": cfg, "classes": {"resin": RESIN, "matrix": MATRIX, "pore": PORE},
            "images": []}
    for i in range(1, run["count"] + 1):
        name = f"{run['id']}_{i:03d}"
        h, w = cfg["image"]["height_px"], cfg["image"]["width_px"]
        img_cfg, drawn = vary(cfg, var_rng)
        res = render_image(h, w, img_cfg, rng)
        img, dark_phase, pore_mask, types = res["img"], res["phase"], res["pores"], res["types"]
        resin_mask, target, specimen = res["resin"], res["target"], res["specimen"]
        labels = np.full((h, w), MATRIX, np.uint8)
        labels[pore_mask] = PORE
        labels[resin_mask] = RESIN
        pore_info = []
        if cfg["pores"]["enabled"]:
            inst, pore_info = pore_instances(pore_mask, types, cfg)
            write_png(gt_dir / f"{name}_pores.png", inst)

        bar_mask, bar_info = scale_bar(img, cfg, bar_rng)
        if bar_mask is not None:
            write_png(gt_dir / f"{name}_scalebar.png", bar_mask.astype(np.uint8) * 255)
        write_png(img_dir / f"{name}.png", img)
        write_png(gt_dir / f"{name}_labels.png", labels)
        write_png(gt_dir / f"{name}_labels_color.png", CLASS_COLORS_BGR[labels])
        write_png(gt_dir / f"{name}_phase.png", dark_phase.astype(np.uint8) * 255)
        write_png(gt_dir / f"{name}_resin.png", resin_mask.astype(np.uint8) * 255)
        if res["stitching"].enabled:
            write_png(gt_dir / f"{name}_frames.png", res["frames"])
        sample_px = max(int((~resin_mask).sum()), 1)
        porosity = float((labels == PORE).sum()) / sample_px     # relative to the sample
        sample = {}
        if specimen.enabled:
            sample = {"center_xy_px": [round(specimen.cx / specimen.ss, 1),
                                       round(specimen.cy / specimen.ss, 1)],
                      "size_wh_px": [round(2 * specimen.hw / specimen.ss, 1),
                                     round(2 * specimen.hh / specimen.ss, 1)],
                      "angle_deg": round(math.degrees(math.atan2(specimen.sin, specimen.cos)), 2)}
        meta["images"].append({"name": name, "height_px": h, "width_px": w,
                               "um_per_px": cfg["image"]["um_per_px"],
                               "sample_fraction": round(sample_px / (h * w), 4),
                               "sample": sample,
                               "variation": drawn,
                               "matrix_rgb": {"bright": [round(float(v), 1) for v in res["matrix"].bright],
                                              "dark": [round(float(v), 1) for v in res["matrix"].dark]},
                               "stitching": res["stitching"].info(),
                               "scale_bar": bar_info,
                               "dark_phase_fraction": round(float(dark_phase.sum()) / sample_px, 4),
                               "porosity": round(porosity, 5),
                               "target_porosity": None if target is None else round(target, 5),
                               "pore_count": len(pore_info), "pores": pore_info})
        bar_txt = f", scale bar '{bar_info['text']}' = {bar_info['bar_length_px']} px" if bar_info else ""
        print(f"{name}: {w}x{h} px, sample {sample_px / (h * w):.0%} of the image, "
              f"{len(pore_info)} pores, porosity {porosity:.2%}{bar_txt}")

    (gt_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
