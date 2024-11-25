import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

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
from config import API_URL_2, HEADERS_2

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
    
    device_map = infer_auto_device_map(model, max_memory={0: "16GiB", "cpu": "32GiB"})
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        low_cpu_mem_usage=True
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.config.hidden_activation = "gelu_pytorch_tanh"  # Ensure the correct activation function is set
    model.gradient_checkpointing_enable()
    return model, tokenizer

def preprocess_function(examples):
    if "sentence2" in examples:
        inputs = examples["sentence1"] + " [SEP] " + examples["sentence2"]
    else:
        inputs = examples["sentence"]
    labels = examples["label"]
    return {"input_text": inputs, "labels": labels}

def model_finetuning():
    def query(payload):
        response = requests.post(API_URL_2, headers=HEADERS_2, json=payload)
        return response.json()
    
    output = query({
        "inputs": "Can you please let us know more details about your ",
    })
    print(output)

    model_id = "google/gemma-2b"
    model, tokenizer = get_model_and_tokenizer(model_id)

    glue_dataset = load_dataset("glue", "sst2")

    tokenized_dataset = glue_dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.map(
        lambda examples: tokenizer(examples["input_text"], truncation=True, padding="max_length"),
        batched=True,
    )

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
        output_dir="./gemma-glue",
        per_device_train_batch_size=2,  # Reduce batch size
        gradient_accumulation_steps=32,
        num_train_epochs=3,
        logging_steps=10,
        learning_rate=2e-4,
        evaluation_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        fp16=True
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        peft_config=peft_config,
        args=training_args
    )

    trainer.train()

    results = trainer.evaluate()
    print(results)

    trainer.save_model("./gemma2b-glue-finetuned")
    tokenizer.save_pretrained("./gemma2b-glue-finetuned")


if __name__ == '__main__':
    model_finetuning()
