import pandas as pd
from transformers import GPT2Tokenizer

def create_custom_tokenizer(train_file):
    # Load data from CSV
    train_data = pd.read_csv(train_file)
    texts = list(train_data["texto"])

    # Create a new GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', model_max_length=1024)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Tokenize the data
    tokenized_data = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)

    return tokenizer, tokenized_data

def fine_tune_gpt2(train_file, output_dir, num_train_epochs=3, per_device_train_batch_size=2, save_steps=10_000):
    # Create custom tokenizer
    tokenizer, tokenized_data = create_custom_tokenizer(train_file)

    # Configure the GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=save_steps,
        learning_rate=2e-5,
        fp16=True,
        max_steps=50,
        max_seq_length=128,
    )

    # Configure the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=TextDataset(tokenized_data, tokenizer=tokenizer, block_size=128),
    )

    # Start fine-tuning
    trainer.train()

if __name__ == "__main__":
    fine_tune_gpt2(train_file="historia.csv", output_dir="./finetuned_gpt2_model")
