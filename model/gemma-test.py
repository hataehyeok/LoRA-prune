import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model
)
from trl import SFTTrainer, SFTConfig

# Base model and tokenizer names.
base_model_name = "google/gemma-2b-it"

# Load base model to GPU memory.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True).to(device)

# Load tokenizer.
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Dataset for fine-tuning.
training_dataset_name = "daekeun-ml/naver-news-summarization-ko"
training_dataset = load_dataset(training_dataset_name, split="train")

# Check the data.
print(training_dataset)

# Dataset 11 is a QA sample in English.
print(training_dataset[11])
print(training_dataset[0])

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)

sft_config = SFTConfig(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    # optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=50,
    learning_rate=4e-5,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    dataset_text_field="summary",
)

# View the number of trainable parameters.

peft_model = get_peft_model(base_model, peft_config)
peft_model.print_trainable_parameters()

# Initialize an SFT trainer.

sft_trainer = SFTTrainer(
    model=base_model,
    train_dataset=training_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=sft_config,
)

# Run the trainer.
sft_trainer.train()