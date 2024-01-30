import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import pandas as pd

def fine_tune_gpt2(train_file, output_dir, num_train_epochs=3, per_device_train_batch_size=2, save_steps=10_000):
    # Cargar tus datos desde un archivo CSV
    train_data = pd.read_csv(train_file)
    texts = list(train_data["input_text"])

    # Tokenizar tus datos
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenized_data = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)

    # Configurar el modelo GPT-2
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=GPT2Config.from_pretrained("gpt2"))

    # Configurar los argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=save_steps,
        learning_rate=2e-5,  # Ajustar según sea necesario
        fp16=True,  # Habilitar FP16
        max_steps=50,  # Ajustar según sea necesario
        max_seq_length=128,  # Ajustar según sea necesario
    )

    # Configurar el objeto Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=TextDataset(tokenized_data, tokenizer=tokenizer, block_size=128),
    )

    # Iniciar el fine-tuning
    trainer.train()

if __name__ == "__main__":
    fine_tune_gpt2(train_file="historia.csv", output_dir="./finetuned_gpt2_model")
