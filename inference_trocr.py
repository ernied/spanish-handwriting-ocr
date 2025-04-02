from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import torch
import os

# Load model + processor
model_dir = "model_output"  # update if saved elsewhere
model = VisionEncoderDecoderModel.from_pretrained(model_dir)
processor = TrOCRProcessor.from_pretrained(model_dir)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Set the test image path
image_path = "amado-duran-1899.jpg"  # 🔁 update this to your test line
assert os.path.exists(image_path), f"❌ Image not found: {image_path}"

print(f"📷 Loading image: {image_path}")
image = Image.open(image_path).convert("RGB")

# Preprocess and run inference
print("🧠 Running inference...")
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

# Generate tokens
generated_ids = model.generate(pixel_values)
print(f"🔤 Generated IDs: {generated_ids}")

# Decode to text
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n✅ Transcription Result:")
print(transcription)
