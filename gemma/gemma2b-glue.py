import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments, AutoModelForSequenceClassification
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from evaluate import evaluator
import evaluate
from torch.nn.utils import prune
import torch.nn as nn
from transformers.trainer_utils import set_seed
import os
from torch.utils.data import DataLoader


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def generate_prompt(example):
    """Prepare SST-2 prompt format for fine-tuning."""
    prompt_list = []
    for i in range(len(example['sentence'])):
        label = "positive" if example['label'][i] == 1 else "negative"
        prompt_list.append(r"""<bos><start_of_turn>user
The sentiment of the following text:

{}<end_of_turn>
<start_of_turn>model
{}<end_of_turn><eos>""".format(example['sentence'][i], label))
    return prompt_list

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def dataset_loading():
    """Load the GLUE SST-2 dataset."""
    dataset = load_dataset("glue", "sst2")
    train_data = dataset['train']
    valid_data = dataset['validation']
    test_data = dataset['test']

    return train_data, valid_data, test_data

def fine_tuning(train_data, test_data):
    """Fine-tune the model using LoRA configuration."""
    lora_config = LoraConfig(
        r=6,
        lora_alpha=8,
        lora_dropout=0.05,
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
        num_train_epochs=1,
        # max_steps=100,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        report_to='none',
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        # max_seq_length=128,
        tokenizer=tokenizer,
        args=args,
        peft_config=lora_config,
        formatting_func=generate_prompt,
    )

    trainer.train()

    ADAPTER_MODEL = "lora_adapter"
    trainer.model.save_pretrained(ADAPTER_MODEL)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map='auto', torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, ADAPTER_MODEL, device_map='auto', torch_dtype=torch.float16)

    model = model.merge_and_unload()
    model.save_pretrained('gemma-2b-it-sst2')

def model_test_print(test_data):
    """Test the fine-tuned model."""
    BASE_MODEL = "google/gemma-2b-it"
    FINETUNE_MODEL = "./gemma-2b-it-sst2"

    # finetune_model = AutoModelForCausalLM.from_pretrained(FINETUNE_MODEL, device_map={"": 0})
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Results of the fine-tuned model
    pipe_finetuned = pipeline("text-generation", model=base_model, tokenizer=tokenizer, max_new_tokens=64)
    test_sentence = test_data['sentence'][0]

    messages = [
        {
            "role": "user",
            "content": "The sentiment of the following text:\n\n{}".format(test_sentence)
        }
    ]

    prompt = pipe_finetuned.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe_finetuned(
        prompt,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        add_special_tokens=True
    )

    print('\nResults of the model:')
    print(outputs[0]["generated_text"][len(prompt):])

def model_eval(test_data):
    """Evaluate the fine-tuned model on a small subset of the test dataset."""

    BASE_MODEL = "google/gemma-2b-it"
    FINETUNE_MODEL = "./gemma-2b-it-sst2"

    finetune_model = AutoModelForCausalLM.from_pretrained(FINETUNE_MODEL, device_map={"": 0})
    # base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Results of the fine-tuned model
    pipe_finetuned = pipeline("text-generation", model=finetune_model, tokenizer=tokenizer, max_new_tokens=64)
    prompts = [
        f"<bos><start_of_turn>user\nThe sentiment of the following text:\n\n{example['sentence']}<end_of_turn>\n<start_of_turn>model\n"
        for example in test_data
    ]

    outputs = pipe_finetuned(prompts, do_sample=True, temperature=0.7, top_k=50, top_p=0.9, add_special_tokens=True)

    predictions = []
    for output, example in zip(outputs, test_data):
        generated_text = output[0]["generated_text"]
        prompt = f"<bos><start_of_turn>user\nThe sentiment of the following text:\n\n{example['sentence']}<end_of_turn>\n<start_of_turn>model\n"
        prediction_text = generated_text[len(prompt):].strip().lower()

        if "positive" in prediction_text:
            predictions.append(1)
        elif "negative" in prediction_text:
            predictions.append(0)
        else:
            predictions.append(-1)

    true_labels = [example['label'] for example in test_data]

    valid_predictions = [(pred, label) for pred, label in zip(predictions, true_labels) if pred != -1]
    if valid_predictions:
        valid_preds, valid_labels = zip(*valid_predictions)
        accuracy = accuracy_score(valid_labels, valid_preds)
        print(f"Accuracy on test cases: {accuracy * 100:.2f}%")
    else:
        print("No valid predictions to evaluate.")

def knowledge_distillation(pruned_model, teacher_model, dataloader, optimizer, num_epochs=3):
    loss_fn = nn.MSELoss()
    pruned_model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            inputs = batch["input_ids"].to(pruned_model.device)
            attention_mask = batch["attention_mask"].to(pruned_model.device)
            
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=inputs, attention_mask=attention_mask).logits
            
            student_outputs = pruned_model(input_ids=inputs, attention_mask=attention_mask).logits
            loss = loss_fn(student_outputs, teacher_outputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    return pruned_model

def prune_process(test_data):
    """Prune the fine-tuned model using LoRA."""
    set_seed(42)
    
    BASE_MODEL = "google/gemma-2b-it"
    ADAPTER_MODEL = "lora_adapter"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    lora_model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL, device_map="auto", torch_dtype=torch.float16)
    
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)
    
    tokenized_train = test_data.map(tokenize_function, batched=True)
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dataloader = DataLoader(tokenized_train, batch_size=8, shuffle=True)
    
    print("Pruning the model...]\n\n")

    # pruning the model
    sparsity = 0.5

    for name, module in lora_model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=sparsity)
            prune.remove(module, "weight")
    pruned_lora_model = lora_model

    # knowledge distillation
    optimizer = torch.optim.AdamW(pruned_lora_model.parameters(), lr=5e-5)
    pruned_lora_model = knowledge_distillation(pruned_lora_model, lora_model, dataloader, optimizer, num_epochs=3)

    PRUNED_ADAPTER_MODEL = "pruned_lora_adapter"
    pruned_lora_model.save_pretrained(PRUNED_ADAPTER_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype=torch.float16)
    final_model = PeftModel.from_pretrained(base_model, PRUNED_ADAPTER_MODEL)
    final_model.save_pretrained("gemma-2b-it-sst2-pruned")


if __name__ == '__main__':
    train_data, valid_data, test_data = dataset_loading()  # validation dataset is used as a test dataset
    # fine_tuning(train_data)
    # model_test_print(valid_data)
    # model_eval(valid_data)
    prune_process(test_data)

