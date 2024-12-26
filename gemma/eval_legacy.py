import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments
from sklearn.metrics import accuracy_score

def erer_model_eval(test_data):
    """Evaluate the fine-tuned model on the test dataset."""

    BASE_MODEL = "google/gemma-2b-it"
    FINETUNE_MODEL = "./gemma-2b-it-sst2"

    finetune_model = AutoModelForCausalLM.from_pretrained(FINETUNE_MODEL, device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    pipe_finetuned = pipeline("text-generation", model=finetune_model, tokenizer=tokenizer, max_new_tokens=64)

    predictions = []
    true_labels = []

    for example in test_data:
        sentence = example['sentence']
        true_label = example['label']
        true_labels.append(true_label)

        # Create prompt for evaluation
        prompt = f"<bos><start_of_turn>user\nThe sentiment of the following text:\n\n{sentence}<end_of_turn>\n<start_of_turn>model\n"

        # Generate model output
        outputs = pipe_finetuned(
            prompt,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            add_special_tokens=True
        )

        # Extract the model's predicted sentiment
        generated_text = outputs[0]["generated_text"]
        prediction = generated_text[len(prompt):].strip().lower()

        # Map prediction to label (1 = positive, 0 = negative)
        if "positive" in prediction:
            predictions.append(1)
        elif "negative" in prediction:
            predictions.append(0)
        else:
            predictions.append(-1)  # Unknown/invalid prediction

    # Calculate accuracy, ignoring invalid predictions
    valid_predictions = [(pred, label) for pred, label in zip(predictions, true_labels) if pred != -1]
    valid_preds, valid_labels = zip(*valid_predictions)

    accuracy = accuracy_score(valid_labels, valid_preds)
    print(f"Accuracy on test dataset: {accuracy * 100:.2f}%")


def model_eval_2(test_data):
    """Evaluate the fine-tuned model on the test dataset with batch processing."""

    BASE_MODEL = "google/gemma-2b-it"
    FINETUNE_MODEL = "./gemma-2b-it-sst2"

    # Load the fine-tuned model and tokenizer
    finetune_model = AutoModelForCausalLM.from_pretrained(FINETUNE_MODEL, device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    pipe_finetuned = pipeline("text-generation", model=finetune_model, tokenizer=tokenizer, max_new_tokens=64)

    # Prepare prompts for the test dataset
    prompts = [
        f"<bos><start_of_turn>user\nThe sentiment of the following text:\n\n{example['sentence']}<end_of_turn>\n<start_of_turn>model\n"
        for example in test_data
    ]

    # Generate predictions in batches
    outputs = pipe_finetuned(prompts, do_sample=True, temperature=0.7, top_k=50, top_p=0.9, add_special_tokens=True)

    # Process outputs and map to labels
    predictions = []
    for output, example in zip(outputs, test_data):
        generated_text = output["generated_text"]
        prompt = f"<bos><start_of_turn>user\nThe sentiment of the following text:\n\n{example['sentence']}<end_of_turn>\n<start_of_turn>model\n"
        prediction_text = generated_text[len(prompt):].strip().lower()

        if "positive" in prediction_text:
            predictions.append(1)
        elif "negative" in prediction_text:
            predictions.append(0)
        else:
            predictions.append(-1)  # Unknown/invalid prediction

    # Extract true labels
    true_labels = [example['label'] for example in test_data]

    # Calculate accuracy, ignoring invalid predictions
    valid_predictions = [(pred, label) for pred, label in zip(predictions, true_labels) if pred != -1]
    if valid_predictions:
        valid_preds, valid_labels = zip(*valid_predictions)
        accuracy = accuracy_score(valid_labels, valid_preds)
        print(f"Accuracy on test dataset: {accuracy * 100:.2f}%")
    else:
        print("No valid predictions to evaluate.")

def model_eval_10case(test_data):
    """Evaluate the fine-tuned model on a small subset of the test dataset."""

    BASE_MODEL = "google/gemma-2b-it"
    FINETUNE_MODEL = "./gemma-2b-it-sst2"

    # Reshape test_data into a list of dictionaries
    test_data_sample = test_data.select(range(100))

    # Load the fine-tuned model and tokenizer
    finetune_model = AutoModelForCausalLM.from_pretrained(FINETUNE_MODEL, device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    pipe_finetuned = pipeline("text-generation", model=finetune_model, tokenizer=tokenizer, max_new_tokens=64)

    # Prepare prompts for the test dataset
    prompts = [
        f"<bos><start_of_turn>user\nThe sentiment of the following text:\n\n{example['sentence']}<end_of_turn>\n<start_of_turn>model\n"
        for example in test_data_sample
    ]

    # Generate predictions in batches
    outputs = pipe_finetuned(prompts, do_sample=True, temperature=0.7, top_k=50, top_p=0.9, add_special_tokens=True)

    # import pdb; pdb.set_trace()

    # Process outputs and map to labels
    predictions = []
    for output, example in zip(outputs, test_data_sample):
        generated_text = output[0]["generated_text"]
        prompt = f"<bos><start_of_turn>user\nThe sentiment of the following text:\n\n{example['sentence']}<end_of_turn>\n<start_of_turn>model\n"
        prediction_text = generated_text[len(prompt):].strip().lower()

        if "positive" in prediction_text:
            predictions.append(1)
        elif "negative" in prediction_text:
            predictions.append(0)
        else:
            predictions.append(-1)  # Unknown/invalid prediction

    # Extract true labels
    true_labels = [example['label'] for example in test_data_sample]

    # Calculate accuracy, ignoring invalid predictions
    valid_predictions = [(pred, label) for pred, label in zip(predictions, true_labels) if pred != -1]
    if valid_predictions:
        valid_preds, valid_labels = zip(*valid_predictions)
        accuracy = accuracy_score(valid_labels, valid_preds)
        print(f"Accuracy on first 10 test cases: {accuracy * 100:.2f}%")
    else:
        print("No valid predictions to evaluate.")






def model_eval_lg(test_data):
    """Evaluate the fine-tuned model on a small subset of the test dataset."""

    BASE_MODEL = "google/gemma-2b-it"
    FINETUNE_MODEL = "./gemma-2b-it-sst2"

    test_data = test_data.select(range(500))

    # finetune_model = AutoModelForCausalLM.from_pretrained(FINETUNE_MODEL, device_map={"": 0})
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Results of the fine-tuned model
    pipe_finetuned = pipeline("text-generation", model=base_model, tokenizer=tokenizer, max_new_tokens=64)
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



# Use huggingface evaluate module but didn't work
def eval_model(test_data):
    """Evaluate the fine-tuned model."""
    BASE_MODEL = "google/gemma-2b-it"
    FINETUNE_MODEL = "./gemma-2b-it-sst2"

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    pipe = pipeline("text-generation", model=base_model, tokenizer=tokenizer, max_new_tokens = 64)
    data = test_data
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