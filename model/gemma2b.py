import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import load_dataset

def my_load_dataset():
    dataset = load_dataset("daekeun-ml/naver-news-summarization-ko")
    print(dataset)
    print(dataset['train'][0])
    
    return dataset

def generate_prompt(example):
    prompt_list = []
    for i in range(len(example['document'])):
        prompt_list.append(r"""<bos><start_of_turn>user
다음 글을 요약해주세요:

{}<end_of_turn>
<start_of_turn>model
{}<end_of_turn><eos>""".format(example['document'][i], example['summary'][i]))
    return prompt_list

def model_finetuning(train_data):

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
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_special_tokens=True)
    tokenizer.padding_side = 'right'

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        max_seq_length=512,
        args=TrainingArguments(
            output_dir="outputs",
            num_train_epochs = 1,
            max_steps=3000,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            optim="paged_adamw_8bit",
            warmup_steps=0.03,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=100,
            push_to_hub=False,
            report_to='none',
        ),
        peft_config=lora_config,
        formatting_func=generate_prompt,
    )

    trainer.train()

    model_finetuning()
    ADAPTER_MODEL = "lora_adapter"

    trainer.model.save_pretrained(ADAPTER_MODEL)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map='auto', torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, ADAPTER_MODEL, device_map='auto', torch_dtype=torch.float16)

    model = model.merge_and_unload()
    model.save_pretrained('gemma-2b-it-sum-ko')


if __name__ == '__main__':
    dataset = my_load_dataset()
    train_data = dataset['train']
    print(generate_prompt(train_data[:1])[0])

    print("-------------------------------------------------------------------------\n\n\n")
    model_finetuning(train_data)
