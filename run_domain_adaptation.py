# Importar las bibliotecas necesarias
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, GPT2ForSequenceClassification
from transformers import TrainingArguments, Trainer

# ...

# Definir los argumentos de entrenamiento y modelo
training_args = TrainingArguments(
    output_dir="./llm_output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Reducir el tamaño del lote si la V100 se queda sin memoria
    save_steps=10_000,
    save_total_limit=2,
    learning_rate=5e-5,
    fp16=True,
    fp16_opt_level="O2",
)

model_name_or_path = "gpt2"  # Puedes ajustar según tus necesidades
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(moadel_name_or_path)

# Crear un modelo de clasificación sobre GPT-2
model = GPT2ForSequenceClassification.from_pretrained(model_name_or_path)

# Cargar tus datos desde un archivo CSV
# Asumiendo que tienes una columna llamada 'texto' en tu archivo CSV
import pandas as pd

train_data = pd.read_csv("tu_archivo.csv")
train_data = list(train_data["texto"])

# Tokenizar tus datos
train_data = tokenizer(train_data, return_tensors="pt", truncation=True, padding=True)

# Configurar el objeto Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
)

# Iniciar el entrenamiento
trainer.train()
