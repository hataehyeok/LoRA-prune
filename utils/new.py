import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import requests
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    get_scheduler,
)
from accelerate import init_empty_weights, infer_auto_device_map
from peft import LoraConfig
from trl import SFTTrainer
from config import API_URL, HEADERS

# Set environment variables to use GPUs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Initialize the distributed process group
dist.init_process_group(backend='nccl')

# Set local rank for distributed training
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device("cuda", local_rank)
torch.cuda.set_device(local_rank)

def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True
    )
    
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=None,
            low_cpu_mem_usage=True
        )
    
    device_map = infer_auto_device_map(model, max_memory={local_rank: "16GiB", "cpu": "32GiB"})
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        low_cpu_mem_usage=True
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    return model, tokenizer

def preprocess_function(examples):
    if "sentence2" in examples:
        inputs = examples["sentence1"] + " [SEP] " + examples["sentence2"]
    else:
        inputs = examples["sentence"]
    labels = examples["label"]
    return {"input_text": inputs, "labels": labels}

def llama_finetuning():
    def query(payload):
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        return response.json()
        
    if local_rank == 0:
        output = query({
            "inputs": "Can you please let us know more details about your ",
        })
        print(output)

    model_id = "meta-llama/Meta-Llama-3-8B"
    model, tokenizer = get_model_and_tokenizer(model_id)
    model.to(device)

    # Wrap the model with DistributedDataParallel
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    glue_dataset = load_dataset("glue", "sst2")
    if local_rank == 0:
        print(glue_dataset)

    tokenized_dataset = glue_dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.map(
        lambda examples: tokenizer(examples["input_text"], truncation=True, padding="max_length"),
        batched=True,
    )

    # Create DistributedSampler for training and validation datasets
    train_sampler = DistributedSampler(tokenized_dataset["train"], num_replicas=dist.get_world_size(), rank=local_rank)
    eval_sampler = DistributedSampler(tokenized_dataset["validation"], num_replicas=dist.get_world_size(), rank=local_rank)

    # Create DataLoader with DistributedSampler
    train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=1, sampler=train_sampler)  # Reduce batch size
    eval_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=1, sampler=eval_sampler)  # Reduce batch size

    # LoRA configuration
    peft_config = LoraConfig(
        r=8,  # Low-rank adaptation size
        lora_alpha=16, 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./llama3-glue",
        per_device_train_batch_size=1,  # Reduce batch size
        gradient_accumulation_steps=32,
        num_train_epochs=3,
        logging_steps=10,
        learning_rate=2e-4,
        evaluation_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        fp16=True
    )

    # Custom training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * training_args.num_train_epochs
    )

    model.train()
    for epoch in range(training_args.num_train_epochs):
        train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()

            if step % training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if step % training_args.logging_steps == 0 and local_rank == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    # Evaluation
    model.eval()
    total_loss = 0
    for step, batch in enumerate(eval_dataloader):
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(eval_dataloader)
    if local_rank == 0:
        print(f"Validation Loss: {avg_loss}")

    # Save the fine-tuned model and tokenizer
    if local_rank == 0:
        model.module.save_pretrained("./llama3-glue-finetuned")
        tokenizer.save_pretrained("./llama3-glue-finetuned")

if __name__ == '__main__':
    print('local_rank:', local_rank)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    llama_finetuning()
    dist.destroy_process_group()
