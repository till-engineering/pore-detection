# Good to know

> Known behaviour that is accepted for now but may cause problems later.

## Pore numbering is not stable across machines

`connected_components` uses `cv2.connectedComponents`. OpenCV labels the image in parallel
stripes, so the order of the pore ids (`pore_id`) depends on the image size and on the number of
CPU threads. The pores themselves (pixels, areas, count) are always the same.

- Same image on another machine (other core count) or with another ROI crop → other `pore_id`s.
- Within one run the ids are consistent (mask, labels, `pores.json`).
- Fix if needed: renumber the labels in a fixed order (e.g. top-left first pixel, row by row)
  inside `connected_components`. Costs some time per image.

## GPU and CPU path differ by float rounding

A GPU path computes the same thing as the CPU path, but not bit for bit (e.g. `gauss`: max.
difference ~4e-7). A pixel that lies exactly at a threshold can therefore become pore on one
machine and not on the other. Not seen on the test images (0 differing mask pixels).

## GPU: whole image also computes tiles without sample

If the whole image fits into the GPU memory, a GPU step runs on the whole cropped image in one
piece – also on tiles that contain no sample. On the CPU those tiles are skipped (filters leave
them unchanged). Inside the sample the result is the same; only filter values outside the sample
differ. Matters only for a later step that looks at the gray values outside the sample (e.g. a
global threshold over the whole cropped image).

## GPU is only worth it for expensive methods

Every GPU step copies its data to the GPU and back, and loading PyTorch + starting CUDA costs
~3 s per run. For cheap filters (e.g. `gauss`) OpenCV on the CPU is faster: 5 × 99 MP with
`gauss` took 9.3 s on the CPU and 15.9 s on the RTX 2060, so `gauss` has no GPU path any more.
torch is only loaded if a configured method has a GPU path.

The GPU version of `gauss` (two separable `conv2d` with the kernel of `cv2.getGaussianKernel`,
`F.pad(mode="reflect")` = `BORDER_REFLECT_101`) matched the CPU path up to 4e-7 – a good
template for the first real GPU method.

## Memory: up to three originals at once

While image k is written in the background and image k+2 is prefetched (`runtime.prefetch`),
three original images are in memory for a short time (plus the working data of image k+1).
Example: 400 MP panorama ≈ 1.2 GB per original → about 3.6 GB just for the originals.
If memory gets tight: set `runtime.prefetch: false`.
