from datasets import load_dataset, Features, Value
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from PIL import Image
import torch

# Load processor and base model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load dataset with explicit schema
features = Features({
    "image": Value("string"),
    "text": Value("string")
})
dataset = load_dataset("json", data_files={"train": "dataset/labels.jsonl"}, features=features)["train"]

# Load and preprocess images + labels
def load_and_process(example):
    image_path = f"dataset/{example['image']}"
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values[0]
    
    # Tokenize labels (text)
    labels = processor.tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).input_ids[0]

    example["pixel_values"] = pixel_values
    example["labels"] = labels
    return example

dataset = dataset.map(load_and_process, remove_columns=["image", "text"])

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="model_output",
    per_device_train_batch_size=1,
    predict_with_generate=True,
    num_train_epochs=10,
    logging_dir="logs",
    logging_steps=1,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    save_strategy="epoch"
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.feature_extractor,
)

# Train!
trainer.train()

# Save model and processor
model.save_pretrained("model_output")
processor.save_pretrained("model_output")

print("✅ Training complete! Model saved to 'model_output/'")
