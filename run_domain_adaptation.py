import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, TrainingArguments, Trainer
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Domain Adaptation with GPT-2 for Sequence Classification")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to CSV file with 'texto' column")
    parser.add_argument("--output_dir", type=str, default="./llm_output", help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--save_steps", type=int, default=10_000, help="Save steps")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Save total limit")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")

    args = parser.parse_args()

    # Definir los argumentos de entrenamiento y modelo
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        fp16_opt_level="O2",
    )

    model_name_or_path = args.model_name_or_path
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

    # Add a new pad token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = GPT2ForSequenceClassification.from_pretrained(model_name_or_path)

    # Cargar tus datos desde un archivo CSV
    train_data = pd.read_csv(args.dataset_path)
    texts = list(train_data["texto"])

    # Tokenizar tus datos
    tokenized_data = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)

    # Configurar el objeto Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
    )

    # Iniciar el entrenamiento
    trainer.train()

if __name__ == "__main__":
    main()
