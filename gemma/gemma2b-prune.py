import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments
from transformers import TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def generate_prompt(example):
    prompt_list = []
    for i in range(len(example['document'])):
        prompt_list.append(r"""<bos><start_of_turn>user
다음 글을 요약해주세요:

{}<end_of_turn>
<start_of_turn>model
{}<end_of_turn><eos>""".format(example['document'][i], example['summary'][i]))
    return prompt_list

def dataset_loading():
    dataset = load_dataset("daekeun-ml/naver-news-summarization-ko")
    train_data = dataset['train']
    test_data = dataset['test']['document'][10]
    return train_data, test_data

def fine_tuning(train_data):
    lora_config = LoraConfig(
        r=6,
        lora_alpha = 8,
        lora_dropout = 0.05,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    BASE_MODEL = "google/gemma-2b-it"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map=None,
        low_cpu_mem_usage=True,
    )

    args = TrainingArguments(
        output_dir="outputs",
        num_train_epochs = 0.5,
        max_steps=30,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_8bit",
        warmup_steps=0,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=100,
        push_to_hub=False,
        report_to='none',
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        max_seq_length=256,
        tokenizer=tokenizer,
        args=args,
        peft_config=lora_config,
        formatting_func=generate_prompt,
    )

    print(train_data)
    trainer.train()

    ADAPTER_MODEL = "lora_adapter"

    trainer.model.save_pretrained(ADAPTER_MODEL)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map='auto', torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, ADAPTER_MODEL, device_map='auto', torch_dtype=torch.float16)

    model = model.merge_and_unload()
    model.save_pretrained('gemma-2b-it-sum-ko')

def model_test(test_data):
    BASE_MODEL = "google/gemma-2b-it"
    FINETUNE_MODEL = "./gemma-2b-it-sum-ko"

    finetune_model = AutoModelForCausalLM.from_pretrained(FINETUNE_MODEL, device_map={"":0})
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    pipe_finetuned = pipeline("text-generation", model=finetune_model, tokenizer=tokenizer, max_new_tokens=512)

    messages = [
        {
            "role": "user",
            "content": "다음 글을 요약해주세요:\n\n{}".format(test_data)
        }
    ]

    prompt = pipe_finetuned.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe_finetuned(
        prompt,
        do_sample=True,
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        add_special_tokens=True
    )
    print(outputs[0]["generated_text"][len(prompt):])


def sample_test():
    dataset = load_dataset("daekeun-ml/naver-news-summarization-ko")
    BASE_MODEL = "google/gemma-2b-it"

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map={"":0})
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    doc = dataset['train']['document'][0]
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    messages = [
        {
            "role": "user",
            "content": "다음 글을 요약해주세요:\n\n{}".format(doc)
        }
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    outputs = pipe(
        prompt,
        do_sample=True,
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        add_special_tokens=True
    )
    print(outputs[0]["generated_text"][len(prompt):])


if __name__ == '__main__':
    train_data, test_data = dataset_loading()
    # fine_tuning(train_data)
    model_test(test_data)
    # sample_test()