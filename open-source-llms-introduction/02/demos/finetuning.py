from dotenv import load_dotenv
from mlx_tune import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer,SFTConfig
from transformers import TrainingArguments
from huggingface_hub import login
import os



# 1. Carga las variables del archivo .env
load_dotenv()

token = os.getenv("HF_TOKEN")

if token:
    login(token=token)
else:
    print("⚠️ No se encontró el HF_TOKEN en el archivo .env")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", # Target attention layers
                      "gate_proj", "up_proj", "down_proj",], # Target FFN layers
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
)

# Construimos la ruta absoluta basada en lo que me pasaste
ruta_archivo = "/Users/conrad/Documents/LLM/open-source-llms-introduction/02/demos/training_data/train.jsonl"

# Verificamos si existe antes de cargar
if not os.path.exists(ruta_archivo):
    raise FileNotFoundError(f"⚠️ ¡Error! No encontré el archivo en: {ruta_archivo}")

# Ahora cargamos el dataset con la ruta segura
dataset = load_dataset("json", data_files=ruta_archivo, split="train")
print("✅ Dataset cargado correctamente desde la ruta absoluta.")

# dataset = load_dataset("json", data_files="./training_data/train.jsonl", split="train")


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):

        text = f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
        texts.append(text)
    return { "text" : texts, }
dataset = dataset.map(formatting_prompts_func, batched = True)


training_args = SFTConfig(
    output_dir = "outputs",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    num_train_epochs = 3,
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    save_strategy = "epoch",
    # PASÁS LOS PARÁMETROS ACÁ:
    dataset_text_field = "text",
    max_length = 2048,
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    processing_class = tokenizer, # Recordá que cambiamos tokenizer por esto
    args = training_args,
)


trainer.train()


model.save_pretrained("globo_assist_lora_adapter")