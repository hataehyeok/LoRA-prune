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
import os


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
    test_data = dataset['test']

    return train_data, test_data

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
        max_steps=100,
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
        max_seq_length=128,
        tokenizer=tokenizer,
        args=args,
        peft_config=lora_config,
        formatting_func=generate_prompt,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    ADAPTER_MODEL = "lora_adapter"
    trainer.model.save_pretrained(ADAPTER_MODEL)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map='auto', torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, ADAPTER_MODEL, device_map='auto', torch_dtype=torch.float16)

    # model = model.merge_and_unload()
    # model.save_pretrained('gemma-2b-it-sst2_test')

    predictions = trainer.predict(test_data)
    print(predictions.predictions.shape, predictions.label_ids.shape)

    logits = predictions.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_true = predictions.label_ids

    # Classification Report
    class_report = classification_report(y_true, y_pred)
    print("Classification Report:")
    print(class_report)


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
    
def eval_model(test_data):
    """Evaluate the fine-tuned model."""
    BASE_MODEL = "google/gemma-2b-it"
    FINETUNE_MODEL = "./gemma-2b-it-sst2"

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    pipe = pipeline("text-classification", model=base_model, tokenizer=tokenizer)    # test_data = test_data.select(range(10))
    # data = test_data
    data = load_dataset("glue", "sst2")['test'].shuffle().select(range(10))
    metric = evaluate.load("accuracy")
    
    task_evaluator = evaluator("text-generation")
    results = task_evaluator.compute(
        model_or_pipeline=pipe,
        data=data,
        metric=metric,
        input_column="sentence",
        label_column="label",
        label_mapping={"negative": 0, "positive": 1}
    )
    print(results)

def fi_eval_model():
    BASE_MODEL = "google/gemma-2b-it"

    # Load model and tokenizer for classification
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Classification pipeline
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Load dataset
    data = load_dataset("glue", "sst2")['test']
    print(data)
    data = data.select(range(20))

    input_column = "sentence"
    label_column = "label"

    # Load accuracy metric
    metric = evaluate.load("accuracy")

    # Evaluate
    predictions = []
    references = []

    for example in data:
        print(example)
        print('\n')

    for example in data:
        # Get model prediction
        prediction = pipe(example[input_column])[0]['label']
        # Convert prediction from 'LABEL_0'/'LABEL_1' to 0/1
        predicted_label = int(prediction.split('_')[-1])
        predictions.append(predicted_label)

        # Append true label (ensure it's correctly accessed)
        references.append(example[label_column])

    # Debugging: Print predictions and references
    print('\n\nPredictions:', predictions)
    print('References:', references, '\n\n')

    # Compute metric
    results = metric.compute(predictions=predictions, references=references)
    print(results)

def prune_lora_adapter(lora_model, sparsity=0.5):
    """
    Prunes LoRA adapter layers to the given sparsity level.
    
    Args:
        lora_model: The model with LoRA adapters.
        sparsity: The sparsity level (percentage of weights to prune).
    """
    for name, module in lora_model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=sparsity)
            prune.remove(module, "weight")

    return lora_model

def knowledge_distillation(pruned_model, teacher_model, dataloader, optimizer, num_epochs=3):
    loss_fn = nn.MSELoss()
    pruned_model.train()
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs = batch["input_ids"].to(pruned_model.device)
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs).logits
            
            student_outputs = pruned_model(inputs).logits
            loss = loss_fn(student_outputs, teacher_outputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return pruned_model

def prune_process():
    """Prune the fine-tuned model using LoRA."""
    
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
        device_map=None,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    lora_model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL, device_map="auto")

    sparsity = 0.5
    pruned_lora_model = prune_lora_adapter(lora_model, sparsity=sparsity)
    # pruned_lora_model = knowledge_distillation(pruned_lora_model, lora_model, dataloader, optimizer, num_epochs=3)

    PRUNED_ADAPTER_MODEL = "pruned_lora_adapter"
    pruned_lora_model.save_pretrained(PRUNED_ADAPTER_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype=torch.float16)
    final_model = PeftModel.from_pretrained(base_model, PRUNED_ADAPTER_MODEL)
    final_model.save_pretrained("gemma-2b-it-sst2-pruned")

        
    PRUNED_ADAPTER_MODEL = "pruned_lora_adapter"
    pruned_lora_model.save_pretrained(PRUNED_ADAPTER_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype=torch.float16)
    final_model = PeftModel.from_pretrained(base_model, PRUNED_ADAPTER_MODEL)
    final_model.save_pretrained("gemma-2b-it-sst2-pruned")


def sst2_dataset_view():
    """View the SST-2 dataset."""
    dataset = load_dataset("glue", "sst2")
    print(dataset)

    train_data = dataset['train']
    valid_data = dataset['validation']
    test_data = dataset['test']

    print(train_data)
    print(valid_data)
    print(test_data)

    train_data = train_data.select(range(10))
    valid_data = valid_data.select(range(10))
    test_data = test_data.select(range(10))

    for example in train_data:
        print(example)
        print('\n')
    for example in valid_data:
        print(example)
        print('\n')
    for example in test_data:
        print(example)
        print('\n')




if __name__ == '__main__':
    # train_data, test_data = dataset_loading()
    # fine_tuning(train_data, test_data)
    # model_test_print(test_data)
    # fi_eval_model()
    sst2_dataset_view()
