"""
Inference script for RingUNet - runnable directly (green Run button), no
command-line arguments needed. Just edit the paths in InferConfig below.

Only single-channel (2D) grayscale TIFFs are supported - RGB/multi-channel
images are skipped with a warning.

Images of any size or aspect ratio are supported: each image is resized
aspect-ratio-preserving to fit inside the model's working resolution
(work_h x work_w) and zero-padded to fill the rest, so the ring's shape
isn't distorted. Predictions are then mapped back into that image's own
original pixel coordinates.
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn


# ============================================================================
# CONFIG - edit these paths, then just press Run
# ============================================================================
class InferConfig:
    def __init__(self):
        # Path to the trained model checkpoint
        self.checkpoint_path = Path(r"path/to/best_model.pth")

        # Folder containing the new raw .tif/.tiff images to run inference on
        self.input_dir = Path(r"path/to/your/raw_images")

        # Where to save results (overlays, registered crops, results.csv)
        self.output_dir = Path(r(r"path/to/save/results")


        # Model input size - MUST match what the model was trained with
        self.work_h = 480
        self.work_w = 640

        # Mask binarization threshold used to decode the predicted ring
        self.threshold = 0.5

        # Size of the registered, centered crop saved for each image (square)
        self.crop_size = 800


# ============================================================================
# MODEL (must match the architecture used during training exactly)
# ============================================================================
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class RingUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = conv_block(1, 16)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = conv_block(32, 64)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = conv_block(64, 128)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(128, 256)

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = conv_block(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = conv_block(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = conv_block(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = conv_block(32, 16)

        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        mask = torch.sigmoid(self.out_conv(d1))
        return {'mask': mask}


def normalize_image(image):
    img_min = image.min()
    img_max = image.max()
    if img_max - img_min > 0:
        return (image - img_min) / (img_max - img_min)
    else:
        return np.zeros_like(image)


def letterbox_resize(image, target_h, target_w):
    """
    Aspect-ratio-preserving resize: scales `image` down/up so it fits
    entirely inside (target_h, target_w), then zero-pads the remainder.
    Returns (padded_image, scale, pad_x, pad_y) where:
      - scale is the single uniform scale factor applied to both axes
      - pad_x, pad_y are the zero-padding added on the left/top
        (padding is split evenly, so this is also half the total pad)
    To map a point from the padded/resized image back to original
    coordinates: orig = (coord - pad) / scale
    """
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w), dtype=resized.dtype)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas, scale, pad_x, pad_y


def decode_mask_to_circle(mask_np, threshold=0.5):
    binary = (mask_np > threshold)
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return None
    cx = float(xs.mean())
    cy = float(ys.mean())
    radius = float(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).mean())
    return cx, cy, radius


def crop_centered(image, cx, cy, size=800):
    """Crop a `size`x`size` window out of `image`, centered on (cx, cy).
    This 'registers' the image so the detected ring center lands exactly in
    the middle of the output. If the requested window goes past the edge of
    the source image, the missing area is filled with zeros (black)."""
    h, w = image.shape[:2]
    half = size // 2

    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = x0 + size
    y1 = y0 + size

    canvas = np.zeros((size, size), dtype=image.dtype)

    src_x0, src_y0 = max(x0, 0), max(y0, 0)
    src_x1, src_y1 = min(x1, w), min(y1, h)

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if src_x1 > src_x0 and src_y1 > src_y0:
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]

    return canvas


def find_image_files(folder):
    exts = ['.tif', '.tiff']
    files = []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in exts and not f.name.startswith('.') and not f.name.startswith('~'):
            files.append(f)
    return files


def main():
    config = InferConfig()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    registered_dir = config.output_dir / "registered_800"
    registered_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = RingUNet()
    # weights_only=False: safe here since this checkpoint was produced by
    # our own training script, not downloaded from an untrusted source.
    checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}, "
              f"val_loss={checkpoint.get('val_loss', '?')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded raw state_dict checkpoint.")

    model = model.to(device)
    model.eval()

    image_files = find_image_files(config.input_dir)
    print(f"Found {len(image_files)} images in {config.input_dir}")

    if len(image_files) == 0:
        print("No .tif/.tiff files found. Nothing to do.")
        return

    results = []

    with torch.no_grad():
        for img_path in image_files:
            raw_image_original = tiff.imread(img_path)  # keep native dtype/scale for the saved crop

            if raw_image_original.ndim != 2:
                print(f"  WARNING: {img_path.name} is not a single-channel grayscale image "
                      f"(shape={raw_image_original.shape}) - skipping. This script only "
                      f"supports 2D grayscale TIFFs.")
                results.append({'name': img_path.stem, 'center_x': None, 'center_y': None,
                                 'radius_px': None, 'detected': False})
                continue

            raw_image = raw_image_original.astype(np.float32)
            raw_image_norm = normalize_image(raw_image)

            model_input, scale, pad_x, pad_y = letterbox_resize(
                raw_image_norm, config.work_h, config.work_w
            )
            model_input_t = torch.from_numpy(model_input).float().unsqueeze(0).unsqueeze(0).to(device)

            pred = model(model_input_t)
            pred_mask = pred['mask'][0, 0].cpu().numpy()

            decoded = decode_mask_to_circle(pred_mask, threshold=config.threshold)

            if decoded is None:
                print(f"  WARNING: {img_path.name} - no ring detected above threshold {config.threshold}")
                results.append({'name': img_path.stem, 'center_x': None, 'center_y': None,
                                 'radius_px': None, 'detected': False})
                continue

            cx_mask, cy_mask, r_mask = decoded
            # Undo the letterbox: subtract padding, then divide by the single
            # uniform scale (same for x and y, since aspect ratio was preserved).
            cx_raw = (cx_mask - pad_x) / scale
            cy_raw = (cy_mask - pad_y) / scale
            r_raw = r_mask / scale

            results.append({'name': img_path.stem, 'center_x': cx_raw, 'center_y': cy_raw,
                             'radius_px': r_raw, 'detected': True})

            # Registered crop: same physical center for every image, fixed
            # size, cropped from the ORIGINAL (non-normalized) raw data so
            # pixel values are preserved for any downstream analysis.
            registered_crop = crop_centered(raw_image_original, cx_raw, cy_raw, size=config.crop_size)
            tiff.imwrite(str(registered_dir / f"{img_path.stem}_registered.tif"), registered_crop)

            # Save overlay
            img_disp = (np.clip(raw_image_norm, 0, 1) * 255).astype(np.uint8)
            img_color = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)
            cv2.circle(img_color, (int(cx_raw), int(cy_raw)), max(int(r_raw), 1), (0, 0, 255), 3)
            cv2.circle(img_color, (int(cx_raw), int(cy_raw)), 5, (0, 0, 255), -1)
            cv2.putText(img_color, f"pred r={r_raw:.1f}px", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.imwrite(str(config.output_dir / f"{img_path.stem}_pred.png"), img_color)
            print(f"  {img_path.name}: center=({cx_raw:.1f}, {cy_raw:.1f}), radius={r_raw:.1f}px")

    # ------------------------------------------------------------------
    # Write results.csv in the format: Frame, CenterX, CenterY, RingRadius
    # Frame is a 1-based sequential index in file order. Values are
    # formatted to 3 decimal places. Undetected frames get blank fields.
    # ------------------------------------------------------------------
    csv_path = config.output_dir / "results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'CenterX', 'CenterY', 'RingRadius'])
        for i, r in enumerate(results, start=1):
            if r['detected']:
                writer.writerow([
                    i,
                    f"{r['center_x']:.3f}",
                    f"{r['center_y']:.3f}",
                    f"{r['radius_px']:.3f}",
                ])
            else:
                writer.writerow([i, '', '', ''])

    n_detected = sum(1 for r in results if r['detected'])
    print(f"\nDone. {n_detected}/{len(results)} images had a ring detected.")
    print(f"Overlays saved to: {config.output_dir}")
    print(f"Registered {config.crop_size}x{config.crop_size} crops saved to: {registered_dir}")
    print(f"Results table saved to: {csv_path}")


if __name__ == "__main__":
    main()
