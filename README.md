# RingUNet Inference

Finds the center and radius of a ring/beam pattern in noisy grayscale images using a trained U-Net.

## Requirements

```bash
pip install torch numpy opencv-python tifffile
```

## Setup

1. Open `infer_new_data.py`.
2. Edit the paths in the `InferConfig` class near the top of the file:

```python
self.checkpoint_path = Path(r"path/to/best_model.pth")
self.input_dir = Path(r"path/to/your/raw_images")
self.output_dir = Path(r"path/to/save/results")
```

3. Leave `work_h`, `work_w`, `threshold`, and `crop_size` as-is unless you know they should change (they must match how the model was trained).

## Run

Just press Run in your editor, or:

```bash
python infer_new_data.py
```

## Input

`input_dir` should contain raw `.tif`/`.tiff` images directly (not in subfolders). Images must be **single-channel grayscale** — color/multi-channel TIFFs are skipped with a warning.

Images can be **any size or aspect ratio**. Each one is resized to fit inside the model's working resolution while keeping its original proportions, then padded with black to fill the rest — so the ring's shape is never stretched or distorted. Results are mapped back to that image's own original pixel coordinates automatically.

## Output

For each image, saved to `output_dir`:

- `<name>_pred.png` — the image with the detected ring drawn in red.
- `registered_800/<name>_registered.tif` — an 800x800 crop centered on the detected ring.
- `results.csv` — one row per image: `Frame, CenterX, CenterY, RingRadius`.

Images where no ring is found print a warning and get blank values in the CSV, but the script keeps going.

## License

MIT
