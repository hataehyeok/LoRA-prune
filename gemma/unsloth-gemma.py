from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from transformers import AutoModelForCausalLM, TrainingArguments
from transformers import TrainingArguments
from peft import PeftModel
from trl import SFTTrainer

def generate_prompt(example):
    prompt_list = []
    for i in range(len(example['document'])):
        prompt_list.append(r"""<bos><start_of_turn>user
다음 글을 요약해주세요:

{}<end_of_turn>
<start_of_turn>model
{}<end_of_turn><eos>""".format(example['document'][i], example['summary'][i]))
    return prompt_list

def unsloth_finetuning(train_data):
    BASE_MODEL = "unsloth/gemma-2b"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL,
        max_seq_length = 512,
        dtype = None,
        load_in_4bit = True, # use 4-bit quantization to reduce memory usage
        device_map=None,
        low_cpu_mem_usage=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 6,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 8,
        lora_dropout = 0.05,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = True, # support rank stabilized LoRA
        loftq_config = None,
        task_type="CAUSAL_LM"
    )

    print("------------------------------------------------------\n")
    model.print_trainable_parameters()
    print("------------------------------------------------------\n")

    args = TrainingArguments(
        output_dir="outputs",
        num_train_epochs = 0.5,
        max_steps=500,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_8bit",
        warmup_steps=5,
        learning_rate=2e-4,
        logging_steps=100,
        push_to_hub=False,
        report_to='none',
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,

    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        dataset_text_field = "text",
        max_seq_length=512,
        args=args,
        formatting_func=generate_prompt,
    )

    print(train_data)
    trainer.train()

    ADAPTER_MODEL = "lora_adapter"

    trainer.model.save_pretrained(ADAPTER_MODEL)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map='auto', torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, ADAPTER_MODEL, device_map='auto', torch_dtype=torch.float16)

    model = model.merge_and_unload()
    model.save_pretrained('unsloth_gemma-2b-it-sum-ko')