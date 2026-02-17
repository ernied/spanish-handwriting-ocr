from PIL import Image, ImageEnhance
import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import spacy

# Load Spanish NLP
nlp = spacy.load("es_core_news_sm")

# Load fine-tuned TrOCR processor and model
model_dir = "model_output"
processor = TrOCRProcessor.from_pretrained(model_dir)
model = VisionEncoderDecoderModel.from_pretrained(model_dir)


def preprocess_image(image_path, enhance_contrast=True, show_image=False):
    """
    Load, convert to grayscale, enhance contrast, and resize image to 384x384.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pil_img = Image.fromarray(gray).convert("RGB")

    if enhance_contrast:
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(2.0)  # You can try 1.5 - 2.5

    # Resize to 384x384 (TrOCR likes this size)
    resized = pil_img.resize((384, 384), Image.LANCZOS)

    if show_image:
        resized.show()

    return resized


def transcribe_image(pil_image):
    """
    Use TrOCR to generate a transcription from the image.
    """
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription


def analyze_text(text):
    """
    Use spaCy to extract named entities (names, dates, etc.).
    """
    print("\n🧠 Extracted Entities:")
    doc = nlp(text)
    for ent in doc.ents:
        print(f"{ent.text} ({ent.label_})")


def process_handwritten_line(image_path):
    print(f"📄 Processing image: {image_path}")
    image = preprocess_image(image_path, enhance_contrast=True, show_image=False)
    text = transcribe_image(image)

    print("\n📝 Transcribed Text:")
    print(text)

    analyze_text(text)


# === 🔽 Update this path to point to your image ===
image_path = "amado-duran-1899-oneline.jpg"

if __name__ == "__main__":
    process_handwritten_line(image_path)
