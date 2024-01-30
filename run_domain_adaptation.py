import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from tokenizers import Tokenizer, models, trainers, processors

def create_custom_tokenizer(train_file):
    # Cargar tus datos desde un archivo CSV
    with open(train_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    texts = [line.split(',')[0].strip() for line in lines[1:]]  # Ignorar la primera línea con encabezados

    # Crear un tokenizer personalizado
    tokenizer = Tokenizer(models.BPE())
    trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[EOS]"])
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP] [MASK] [EOS]",
        pair="[CLS] $A [SEP] $B:1 [MASK] [EOS]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ("[MASK]", tokenizer.token_to_id("[MASK]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )
    tokenizer.enable_truncation(max_length=128)

    return tokenizer

def fine_tune_gpt2(train_file, output_dir, num_train_epochs=3, per_device_train_batch_size=2, save_steps=10_000):
    # Cargar tus datos desde un archivo CSV
    train_data = torch.load(train_file)
    texts = list(train_data["texto"])

    # Crear y guardar el tokenizer personalizado
    tokenizer = create_custom_tokenizer(train_file)
    tokenizer.save("custom_tokenizer.json")

    # Tokenizar tus datos
    encoded_data = tokenizer.encode_batch(texts)
    tokenized_data = [torch.tensor(encoded.ids) for encoded in encoded_data]

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
        train_dataset=TextDataset(tokenized_data),
    )

    # Iniciar el fine-tuning
    trainer.train()

if __name__ == "__main__":
    fine_tune_gpt2(train_file="historia.csv", output_dir="./finetuned_gpt2_model")
