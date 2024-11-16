import os

local_rank = int(os.environ["LOCAL_RANK"])

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import requests
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    )
from accelerate import init_empty_weights, infer_auto_device_map
from peft import LoraConfig
from trl import SFTTrainer
from config import API_URL, HEADERS
import random

# Set environment variables to use GPUs
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set local rank for distributed training
seed = 2103
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device(f'cuda:{local_rank}')
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
# device = torch.device("cuda", local_rank)
# torch.cuda.set_device(local_rank)

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

    print('-------------------------------------------------------------------------\n')
    print('                       Model loaded successfully!\n')
    print('-------------------------------------------------------------------------\n')
    return model, tokenizer

def preprocess_function(examples):
    if "sentence2" in examples:
        inputs = examples["sentence1"] + " [SEP] " + examples["sentence2"]
    else:
        inputs = examples["sentence"]
    labels = examples["label"]
    return {"input_text": inputs, "labels": labels}

def llama_finetuning():

    dist.init_process_group(backend='nccl')
    
    def query(payload):
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        return response.json()
        
    output = query({
        "inputs": "Can you please let us know more details about your ",
    })
    print(output)

    model_id = "meta-llama/Meta-Llama-3-8B"
    model, tokenizer = get_model_and_tokenizer(model_id)
    model = model.cuda(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    
    glue_dataset = load_dataset("glue", "sst2")
    print('-------------------------------------------------------------------------\n')
    print('                       GLUE dataset loaded successfully!\n')
    print(glue_dataset)
    print('-------------------------------------------------------------------------\n')

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
    # TODO
    # adjust gradient accumulation
    training_args = TrainingArguments(
        output_dir="./llama3-glue",
        per_device_train_batch_size=2,
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

    print('-------------------------------------------------------------------------\n')
    print('                       Trainer loaded successfully!\n')
    print('-------------------------------------------------------------------------\n')

    trainer.train()

    results = trainer.evaluate()
    print(results)

    trainer.save_model("./llama3-glue-finetuned")
    tokenizer.save_pretrained("./llama3-glue-finetuned")


if __name__ == '__main__':
    print('local_rank:', local_rank)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    
    # dist.init_process_group(backend='nccl')
    llama_finetuning()
    # dist.destroy_process_group()
