# Spanish Handwriting OCR

Fine-tuning Microsoft TrOCR to read 19th/20th century Spanish handwritten civil records (marriage certificates, birth records, etc.) from Zacatecas, Mexico.

## Project Structure

```
split_lines.py          # Line segmentation tool (main workflow tool)
train_trocr.py          # Fine-tuning script (TrOCR base -> Spanish handwriting)
inference_trocr.py      # Single-image inference
trocr_spanish.py        # Inference + spaCy entity extraction pipeline
load_dataset_test.py    # Dataset loading sanity check
dataset/
  labels.jsonl          # Ground truth: {"image": "images/...", "text": "..."}
  images/               # Cropped line images (JPG)
scans/                  # Original two-page spread scans
model_output/           # Fine-tuned model checkpoint
```

## Workflow

### 1. Segment scans into lines
```bash
# Preview detected lines (saves annotated image)
python split_lines.py preview scans/<scan>.jpg

# Interactive GUI to review/accept lines before saving
python split_lines.py split scans/<scan>.jpg

# Auto-save all detected lines (no GUI)
python split_lines.py split scans/<scan>.jpg --no-gui
```

### 2. Label lines
```bash
# GUI labeling with Spanish OCR pre-fill
python split_lines.py label --pre-ocr

# GUI labeling without pre-fill
python split_lines.py label
```

### 3. Train
```bash
python train_trocr.py
```

### 4. Inference
```bash
python inference_trocr.py
```

## Key Details

- **Language**: All manuscripts are in Spanish. Use `qantev/trocr-base-spanish` as the base model for OCR, not the English one.
- **Base model**: `microsoft/trocr-base-handwritten` (for training), `qantev/trocr-base-spanish` (for pre-OCR labeling)
- **Labels format**: JSONL, one entry per line. Image paths are relative to `dataset/` (e.g. `images/foo.jpg`).
- **Line naming convention**: `{scan_basename}-p{page}-line{N}.jpg`
- **Scans are two-page spreads** — `split_lines.py` auto-detects the gutter and splits them.

## Dependencies

```
pip install torch transformers datasets Pillow opencv-python numpy scikit-image accelerate
```

`scikit-image` is used for Sauvola thresholding on aged paper. `accelerate` is needed for transformers 5.x model loading.

## Git

- Committer: ernie.duran@gmail.com
- Remote: github.com/ernied/spanish-handwriting-ocr
