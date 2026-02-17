#!/usr/bin/env python3
"""Semi-automatic line-splitting tool for handwritten document scans.

Usage:
    python split_lines.py split <scan>   -- Detect lines and review in GUI
    python split_lines.py preview <scan> -- Save annotated preview image
    python split_lines.py label          -- Label unlabeled line images

CLI flags:
    --min-height  Minimum line height in pixels (default 20)
    --padding     Vertical padding around each line (default 8)
    --no-gui      Skip GUI, auto-save all detected lines
    --no-split-pages  Treat scan as single page (don't split at gutter)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Optional: scikit-image for Sauvola thresholding on aged paper
try:
    from skimage.filters import threshold_sauvola
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# ---------------------------------------------------------------------------
# Core detection functions
# ---------------------------------------------------------------------------

def load_and_binarize(image_path):
    """Load an image and produce a binary (inverted) version for analysis.

    Returns the original color image, grayscale, and a cleaned binary suitable
    for projection-profile line detection.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if HAS_SKIMAGE:
        thresh = threshold_sauvola(gray, window_size=51, k=0.2)
        binary_raw = (gray < thresh).astype(np.uint8) * 255
    else:
        binary_raw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 15
        )

    # Light cleanup: small opening to remove speckles (preserve thin strokes)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_clean = cv2.morphologyEx(binary_raw, cv2.MORPH_OPEN, kernel_open)

    # For line detection: heavy horizontal dilation merges characters into
    # solid bands per line, then vertical erosion widens inter-line gaps.
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (120, 1))
    binary_lines = cv2.dilate(binary_clean, kernel_h, iterations=1)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    binary_lines = cv2.erode(binary_lines, kernel_v, iterations=1)

    return img, gray, binary_raw, binary_lines


def deskew(image, binary):
    """Correct small rotation angles (< 5 degrees)."""
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:
        return image, binary, 0.0

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # minAreaRect returns angles in [-90, 0); normalise to small correction
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > 5:
        return image, binary, 0.0  # too large, likely not a skew issue

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, binary, angle


def split_pages(image, binary, binary_lines, force_single=False):
    """Split a two-page spread at the gutter.

    Returns list of (image, binary_lines, page_label).
    Uses `binary` (raw) for gutter detection, splits both image and binary_lines.
    """
    if force_single:
        return [(image, binary_lines, "p1")]

    h, w = image.shape[:2]
    # Vertical projection: sum dark pixels per column
    v_proj = np.sum(binary, axis=0).astype(float)

    # Look for a valley in the central 30% of the image
    center_start = int(w * 0.35)
    center_end = int(w * 0.65)
    center_region = v_proj[center_start:center_end]

    # Smooth to avoid noise
    kernel_size = max(w // 50, 5)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = cv2.GaussianBlur(center_region.reshape(1, -1), (kernel_size, 1), 0).flatten()

    # Find the deepest valley (minimum)
    gutter_local = int(np.argmin(smoothed))
    gutter_x = gutter_local + center_start

    # Check if the valley is significantly lower than surroundings
    valley_val = smoothed[gutter_local]
    median_val = np.median(smoothed)
    if valley_val < median_val * 0.65:
        # Clear gutter detected — split
        left_img = image[:, :gutter_x]
        left_bl = binary_lines[:, :gutter_x]
        right_img = image[:, gutter_x:]
        right_bl = binary_lines[:, gutter_x:]
        return [(left_img, left_bl, "p1"), (right_img, right_bl, "p2")]

    # No clear gutter — treat as single page
    return [(image, binary_lines, "p1")]


def detect_lines(binary_lines, min_height=20):
    """Detect horizontal text lines via projection profile with local minima.

    Expects the morphologically-processed binary (binary_lines) where
    characters have been merged horizontally and separated vertically.
    Returns list of (y_start, y_end).
    """
    h, w = binary_lines.shape[:2]

    # Horizontal projection: sum dark pixels per row
    h_proj = np.sum(binary_lines, axis=1).astype(float)

    # Smooth the projection
    kernel_size = max(min_height, 5)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = cv2.GaussianBlur(h_proj.reshape(-1, 1), (1, kernel_size), 0).flatten()

    peak = np.max(smoothed)
    if peak == 0:
        return []

    # Find text region (rows with significant content)
    text_threshold = peak * 0.15
    text_rows = np.where(smoothed > text_threshold)[0]
    if len(text_rows) == 0:
        return []
    text_start = text_rows[0]
    text_end = text_rows[-1]

    # Within text region, find local minima as line separators.
    # A local minimum is a valley between two peaks.
    region = smoothed[text_start:text_end + 1]
    region_len = len(region)

    # Use a relative approach: for each position, compare to local neighbors
    # A valley is where the value is significantly below the local max
    half_window = max(min_height * 2, 40)
    valleys = []
    for y in range(half_window, region_len - half_window):
        local_max = max(np.max(region[y - half_window:y]),
                        np.max(region[y + 1:y + half_window + 1]))
        if local_max > 0 and region[y] < local_max * 0.6:
            valleys.append(y + text_start)

    if not valleys:
        # No valleys found — treat entire text region as one line
        return [(text_start, text_end)]

    # Cluster nearby valley rows into separator regions, take midpoints
    separators = []
    group_start = valleys[0]
    prev = valleys[0]
    for v in valleys[1:]:
        if v - prev > 3:
            mid = (group_start + prev) // 2
            separators.append(mid)
            group_start = v
        prev = v
    separators.append((group_start + prev) // 2)

    # Build lines from separators
    lines = []
    prev_y = text_start
    for sep in separators:
        if sep - prev_y >= min_height:
            lines.append((prev_y, sep))
        prev_y = sep
    # Last line
    if text_end - prev_y >= min_height:
        lines.append((prev_y, text_end))

    return lines


def crop_lines(image, lines, padding=8):
    """Crop detected lines from the image with padding. Returns list of cropped images."""
    h, w = image.shape[:2]
    crops = []
    for y_start, y_end in lines:
        y0 = max(0, y_start - padding)
        y1 = min(h, y_end + padding)
        crops.append(image[y0:y1, :])
    return crops


def process_scan(scan_path, min_height=20, padding=8, no_split_pages=False):
    """Full pipeline: load, binarize, deskew, split pages, detect lines.

    Returns list of dicts with keys: page_label, line_index, y_start, y_end,
    crop (numpy array), page_image (full page for context).
    """
    img, gray, binary, binary_lines = load_and_binarize(scan_path)
    img, binary, angle = deskew(img, binary)
    # Apply same deskew to binary_lines
    if angle != 0.0:
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        binary_lines = cv2.warpAffine(binary_lines, M, (w, h),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0)

    pages = split_pages(img, binary, binary_lines, force_single=no_split_pages)

    results = []
    for page_img, page_bl, page_label in pages:
        lines = detect_lines(page_bl, min_height=min_height)
        crops = crop_lines(page_img, lines, padding=padding)
        for i, (crop, (y0, y1)) in enumerate(zip(crops, lines)):
            results.append({
                "page_label": page_label,
                "line_index": i + 1,
                "y_start": y0,
                "y_end": y1,
                "crop": crop,
                "page_image": page_img,
            })

    return results, pages


def make_line_filename(scan_path, page_label, line_index):
    """Generate filename: {scan_basename}-{page}-line{N}.jpg"""
    stem = Path(scan_path).stem
    return f"{stem}-{page_label}-line{line_index}.jpg"


# ---------------------------------------------------------------------------
# Preview command
# ---------------------------------------------------------------------------

def cmd_preview(args):
    """Save an annotated preview image with detected line boundaries."""
    results, pages = process_scan(
        args.scan, min_height=args.min_height, padding=args.padding,
        no_split_pages=args.no_split_pages
    )

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
    ]

    for page_img, _, page_label in pages:
        vis = page_img.copy()
        page_lines = [r for r in results if r["page_label"] == page_label]

        for i, r in enumerate(page_lines):
            color = colors[i % len(colors)]
            y0, y1 = r["y_start"], r["y_end"]
            h, w = vis.shape[:2]
            # Draw semi-transparent overlay
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, y0), (w, y1), color, -1)
            cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
            # Draw border lines
            cv2.line(vis, (0, y0), (w, y0), color, 2)
            cv2.line(vis, (0, y1), (w, y1), color, 2)
            # Label
            cv2.putText(vis, f"line {r['line_index']}", (10, y0 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        out_name = f"{Path(args.scan).stem}-{page_label}-preview.jpg"
        cv2.imwrite(out_name, vis)
        print(f"Saved preview: {out_name} ({len(page_lines)} lines detected)")


# ---------------------------------------------------------------------------
# Split command (with tkinter GUI)
# ---------------------------------------------------------------------------

def save_crops(results, scan_path, output_dir="dataset/images"):
    """Save accepted line crops to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for r in results:
        fname = make_line_filename(scan_path, r["page_label"], r["line_index"])
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, r["crop"])
        saved.append(fname)
        print(f"  Saved: {out_path}")
    return saved


def cmd_split_no_gui(args):
    """Non-interactive split: detect and save all lines."""
    results, _ = process_scan(
        args.scan, min_height=args.min_height, padding=args.padding,
        no_split_pages=args.no_split_pages
    )
    print(f"Detected {len(results)} lines.")
    saved = save_crops(results, args.scan)
    print(f"Saved {len(saved)} line images to dataset/images/")


def cmd_split_gui(args):
    """Interactive GUI for reviewing detected lines."""
    import tkinter as tk
    from tkinter import ttk

    results, pages = process_scan(
        args.scan, min_height=args.min_height, padding=args.padding,
        no_split_pages=args.no_split_pages
    )

    if not results:
        print("No lines detected.")
        return

    root = tk.Tk()
    root.title(f"Line Review — {Path(args.scan).name}")

    # State: which lines are accepted
    accepted = {i: tk.BooleanVar(value=True) for i in range(len(results))}

    # Build scrollable canvas
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(main_frame)
    scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollable = ttk.Frame(canvas)

    scrollable.bind("<Configure>",
                    lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Mouse wheel scrolling
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    # Add line previews
    tk_images = []  # prevent GC
    for i, r in enumerate(results):
        frame = ttk.Frame(scrollable, relief="groove", borderwidth=2)
        frame.pack(fill=tk.X, padx=5, pady=3)

        # Checkbox
        cb = ttk.Checkbutton(frame, text=f"{r['page_label']} line {r['line_index']}",
                             variable=accepted[i])
        cb.pack(side=tk.LEFT, padx=5)

        # Thumbnail of the crop
        crop_rgb = cv2.cvtColor(r["crop"], cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        # Scale to fit width ~800px
        scale = min(1.0, 800 / pil_img.width)
        display_size = (int(pil_img.width * scale), int(pil_img.height * scale))
        pil_img = pil_img.resize(display_size, Image.LANCZOS)

        tk_img = tk.PhotoImage(data=_pil_to_ppm(pil_img))
        tk_images.append(tk_img)

        label = ttk.Label(frame, image=tk_img)
        label.pack(side=tk.LEFT, padx=5, pady=2)

    # Buttons
    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill=tk.X, padx=10, pady=5)

    def on_save():
        selected = [results[i] for i in range(len(results)) if accepted[i].get()]
        save_crops(selected, args.scan)
        print(f"Saved {len(selected)} / {len(results)} lines.")
        root.destroy()

    def on_cancel():
        print("Cancelled — no lines saved.")
        root.destroy()

    def on_select_all():
        for v in accepted.values():
            v.set(True)

    def on_deselect_all():
        for v in accepted.values():
            v.set(False)

    ttk.Button(btn_frame, text="Select All", command=on_select_all).pack(side=tk.LEFT, padx=3)
    ttk.Button(btn_frame, text="Deselect All", command=on_deselect_all).pack(side=tk.LEFT, padx=3)
    ttk.Button(btn_frame, text="Save Selected", command=on_save).pack(side=tk.RIGHT, padx=3)
    ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=3)

    # Status bar
    status = ttk.Label(root, text=f"{len(results)} lines detected from {Path(args.scan).name}")
    status.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)

    root.mainloop()


def _pil_to_ppm(pil_img):
    """Convert a PIL Image to PPM bytes for tkinter PhotoImage."""
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="PPM")
    return buf.getvalue()


def cmd_split(args):
    """Entry point for 'split' command."""
    if args.no_gui:
        cmd_split_no_gui(args)
    else:
        cmd_split_gui(args)


# ---------------------------------------------------------------------------
# Pre-OCR support
# ---------------------------------------------------------------------------

def _materialize_model(model, device):
    """Move model to device, fixing any tensors stuck on meta device."""
    import torch
    # Fix plain tensor attributes on meta device (e.g. sinusoidal embeddings)
    for name, mod in model.named_modules():
        for attr_name in list(vars(mod).keys()):
            val = getattr(mod, attr_name, None)
            if isinstance(val, torch.Tensor) and val.device.type == "meta":
                setattr(mod, attr_name,
                        torch.zeros(val.shape, dtype=val.dtype, device=device))
    model.to(device)


def load_ocr_models(model_dir="model_output"):
    """Load base and fine-tuned TrOCR models for pre-OCR. Returns dict of models."""
    import torch
    from transformers import VisionEncoderDecoderModel, TrOCRProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}

    # Spanish base model (printed text but Spanish-aware decoder)
    print("Loading Spanish TrOCR model...")
    base_proc = TrOCRProcessor.from_pretrained("qantev/trocr-base-spanish")
    base_model = VisionEncoderDecoderModel.from_pretrained(
        "qantev/trocr-base-spanish")
    # Force all parameters off meta device (transformers 5.x compat)
    _materialize_model(base_model, device)
    models["spanish-base"] = (base_proc, base_model)

    # Fine-tuned model (if available)
    if Path(model_dir).exists() and (Path(model_dir) / "model.safetensors").exists():
        print("Loading fine-tuned model...")
        try:
            ft_proc = TrOCRProcessor.from_pretrained(model_dir)
            ft_model = VisionEncoderDecoderModel.from_pretrained(model_dir)
            _materialize_model(ft_model, device)
            models["finetuned"] = (ft_proc, ft_model)
        except Exception as e:
            print(f"  Could not load fine-tuned model: {e}")

    print(f"OCR ready ({len(models)} model(s) loaded on {device}).")
    return models, device


def ocr_image(img_path, models, device):
    """Run OCR on an image using all loaded models.

    Returns (best_text, model_name, details) where details is a list of
    (model_name, text, score) for each model.
    """
    import torch

    pil_img = Image.open(img_path).convert("RGB")
    details = []

    for name, (processor, model) in models.items():
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=128,
            )
        text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        # Compute average log-probability as confidence score
        if outputs.scores:
            log_probs = []
            for i, score in enumerate(outputs.scores):
                token_id = outputs.sequences[0, i + 1]
                log_prob = torch.nn.functional.log_softmax(score, dim=-1)
                log_probs.append(log_prob[0, token_id].item())
            avg_score = sum(log_probs) / len(log_probs) if log_probs else -999
        else:
            avg_score = -999
        details.append((name, text, avg_score))

    # Pick the result with highest confidence
    details.sort(key=lambda x: x[2], reverse=True)
    best_name, best_text, _ = details[0]
    return best_text, best_name, details


# ---------------------------------------------------------------------------
# Label command
# ---------------------------------------------------------------------------

def cmd_label(args):
    """GUI labeling: show each unlabeled image with a text entry field."""
    import tkinter as tk
    from tkinter import ttk

    images_dir = Path("dataset/images")
    labels_file = Path("dataset/labels.jsonl")

    if not images_dir.exists():
        print("No dataset/images/ directory found.")
        return

    # Load existing labels
    labeled = set()
    if labels_file.exists():
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                labeled.add(entry["image"])

    # Find unlabeled images
    all_images = sorted(images_dir.glob("*.jpg"))
    unlabeled = [img for img in all_images if f"images/{img.name}" not in labeled]

    if not unlabeled:
        print("All images are labeled!")
        return

    print(f"Found {len(unlabeled)} unlabeled images ({len(labeled)} already labeled).")

    # Load OCR models if requested
    ocr_models = None
    ocr_device = None
    if getattr(args, "pre_ocr", False):
        ocr_models, ocr_device = load_ocr_models()

    root = tk.Tk()
    root.title("Label Lines")

    state = {"index": 0, "new_labels": []}

    # Image display
    img_label = ttk.Label(root)
    img_label.pack(padx=10, pady=5)

    # Filename label
    name_label = ttk.Label(root, font=("TkDefaultFont", 10, "bold"))
    name_label.pack(padx=10)

    # Progress label
    progress_label = ttk.Label(root)
    progress_label.pack(padx=10)

    # OCR info label (shows which model was used)
    ocr_info_label = ttk.Label(root, foreground="gray")
    ocr_info_label.pack(padx=10)

    # Text entry
    entry_frame = ttk.Frame(root)
    entry_frame.pack(fill=tk.X, padx=10, pady=5)
    ttk.Label(entry_frame, text="Transcription:").pack(side=tk.LEFT)
    text_var = tk.StringVar()
    entry = ttk.Entry(entry_frame, textvariable=text_var, width=100)
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    entry.focus_set()

    tk_images = []  # prevent GC

    def show_image(idx):
        """Display the image at the given index."""
        if idx >= len(unlabeled):
            _finish()
            return

        img_path = unlabeled[idx]
        pil_img = Image.open(img_path)
        # Scale to fit width ~900px, keep aspect ratio
        scale = min(1.0, 900 / pil_img.width)
        display_size = (int(pil_img.width * scale), int(pil_img.height * scale))
        pil_img = pil_img.resize(display_size, Image.LANCZOS)

        tk_img = tk.PhotoImage(data=_pil_to_ppm(pil_img))
        tk_images.clear()
        tk_images.append(tk_img)
        img_label.configure(image=tk_img)

        name_label.configure(text=img_path.name)
        n_done = len(state["new_labels"])
        progress_label.configure(
            text=f"Image {idx + 1} / {len(unlabeled)}  |  {n_done} labeled this session")

        # Pre-fill with OCR if available
        if ocr_models:
            best_text, model_name, details = ocr_image(img_path, ocr_models, ocr_device)
            text_var.set(best_text)
            info_parts = [f"{name}: {text[:50]}" for name, text, score in details]
            ocr_info_label.configure(text=f"Pre-OCR ({model_name}): " + " | ".join(info_parts))
        else:
            text_var.set("")
            ocr_info_label.configure(text="")

        entry.focus_set()
        entry.select_range(0, tk.END)

    def _save_and_next():
        """Save the current transcription and move to next image."""
        text = text_var.get().strip()
        idx = state["index"]
        if text:
            img_path = unlabeled[idx]
            entry_data = {"image": f"images/{img_path.name}", "text": text}
            state["new_labels"].append(entry_data)
            # Append immediately so progress isn't lost on crash
            # Ensure previous content ends with newline
            needs_newline = False
            if labels_file.exists() and labels_file.stat().st_size > 0:
                with open(labels_file, "rb") as fb:
                    fb.seek(-1, 2)
                    needs_newline = fb.read(1) != b"\n"
            with open(labels_file, "a", encoding="utf-8") as f:
                if needs_newline:
                    f.write("\n")
                f.write(json.dumps(entry_data, ensure_ascii=False) + "\n")
            print(f"  Labeled: {img_path.name}")
        state["index"] += 1
        show_image(state["index"])

    def _skip():
        """Skip the current image."""
        state["index"] += 1
        show_image(state["index"])

    def _finish():
        """Close the labeling window."""
        n = len(state["new_labels"])
        print(f"Labeling done. {n} new labels saved to {labels_file}")
        root.destroy()

    # Bind Enter key to save
    entry.bind("<Return>", lambda e: _save_and_next())

    # Buttons
    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill=tk.X, padx=10, pady=5)
    ttk.Button(btn_frame, text="Save & Next (Enter)", command=_save_and_next).pack(side=tk.LEFT, padx=3)
    ttk.Button(btn_frame, text="Skip", command=_skip).pack(side=tk.LEFT, padx=3)
    ttk.Button(btn_frame, text="Quit", command=_finish).pack(side=tk.RIGHT, padx=3)

    show_image(0)
    root.mainloop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Semi-automatic line-splitting tool for handwritten document scans."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- split --
    sp_split = subparsers.add_parser("split", help="Detect and extract lines from a scan")
    sp_split.add_argument("scan", help="Path to scan image")
    sp_split.add_argument("--min-height", type=int, default=20,
                          help="Minimum line height in pixels (default: 20)")
    sp_split.add_argument("--padding", type=int, default=8,
                          help="Vertical padding around each line (default: 8)")
    sp_split.add_argument("--no-gui", action="store_true",
                          help="Skip GUI, save all detected lines automatically")
    sp_split.add_argument("--no-split-pages", action="store_true",
                          help="Treat scan as a single page (don't split at gutter)")
    sp_split.set_defaults(func=cmd_split)

    # -- preview --
    sp_preview = subparsers.add_parser("preview", help="Save annotated preview image")
    sp_preview.add_argument("scan", help="Path to scan image")
    sp_preview.add_argument("--min-height", type=int, default=20,
                            help="Minimum line height in pixels (default: 20)")
    sp_preview.add_argument("--padding", type=int, default=8,
                            help="Vertical padding around each line (default: 8)")
    sp_preview.add_argument("--no-split-pages", action="store_true",
                            help="Treat scan as a single page (don't split at gutter)")
    sp_preview.set_defaults(func=cmd_preview)

    # -- label --
    sp_label = subparsers.add_parser("label", help="Interactively label line images")
    sp_label.add_argument("--pre-ocr", action="store_true",
                          help="Pre-fill transcriptions using TrOCR (base + fine-tuned)")
    sp_label.set_defaults(func=cmd_label)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
