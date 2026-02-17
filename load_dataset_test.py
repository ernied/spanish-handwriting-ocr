from datasets import load_dataset, Features, Value
from PIL import Image
from transformers import TrOCRProcessor

# Load TrOCR processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

# Step 1: Load the dataset with string path only
features = Features({
    "image": Value("string"),
    "text": Value("string")
})
dataset = load_dataset("json", data_files="dataset/labels.jsonl", features=features)["train"]

# Step 2: Load images manually into a Python list (not using map)
loaded_examples = []
for example in dataset:
    image_path = f"dataset/{example['image']}"
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values[0]
        loaded_examples.append({
            "image": image,
            "pixel_values": pixel_values,
            "text": example["text"]
        })
    except Exception as e:
        print(f"Error loading {image_path}: {e}")

# ✅ Confirm loaded data
print("✅ Loaded examples:", len(loaded_examples))
print("📝 First transcription:", loaded_examples[0]["text"])
loaded_examples[0]["image"].show()
