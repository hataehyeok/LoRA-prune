import random
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments, AutoModelForSequenceClassification
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from transformers.trainer_utils import set_seed
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from helper import print_batch_info, print_MHA_info, print_MLP_info, print_LN_info


# def setup_optimizer_and_scheduler(model, learning_rate=2e-4, min_lr=4.5e-7, total_epochs=10):
#     """
#     Setup optimizer and cosine LR scheduler for the model.
    
#     Args:
#         model (nn.Module): The model to optimize.
#         learning_rate (float): Initial learning rate.
#         min_lr (float): Minimum learning rate for cosine decay.
#         total_epochs (int): Total number of epochs for the scheduler.
    
#     Returns:
#         optimizer: The Adam optimizer.
#         scheduler: The cosine LR scheduler.
#     """
#     # Setup optimizer
#     optimizer = Adam(model.parameters(), lr=learning_rate)
    
#     # Setup cosine LR scheduler
#     scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)
    
#     return optimizer, scheduler

# # Assume train_data, valid_data, test_data = dataset_loading()
# calibration_loader = create_calibration_dataset(train_data, num_samples=1024, batch_size=32)

# # Assume `model` is your neural network model
# optimizer, scheduler = setup_optimizer_and_scheduler(
#     model, learning_rate=2e-4, min_lr=4.5e-7, total_epochs=10
# )

# # Iterate over calibration data
# for epoch in range(10):
#     for batch in calibration_loader:
#         # Forward pass, compute loss, backward pass, etc.
#         pass
#     scheduler.step()  # Adjust learning rate after each epoch

def create_calibration_dataset(train_data, tokenizer, num_samples=1024, batch_size=32):
    """
    Create a calibration dataset from the training data.
    """

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)
    
    tokenized_train = train_data.map(tokenize_function, batched=True)
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    random.seed(42)
    indices = random.sample(range(len(tokenized_train)), num_samples)
    calibration_subset = Subset(tokenized_train, indices)
    calibration_loader = DataLoader(calibration_subset, batch_size=batch_size, shuffle=False)
    
    return calibration_loader



# TODO
# compute importance scores with various methods (L2 norm, mean, variance)
# re-simulate minitron results    ->    is will be same results?
def compute_width_importance_scores(model, data_loader):
    """
    Compute importance scores for heads, neurons, and embedding channels.
    Choose L2 norm (batch) and Mean (sequence) and then summed up to compute the layer-wise importance for best zero-shot performance.

    Activations from the Model, input shape: (B, S, D), where:
        B: Batch size
        S: Sequence length (number of tokens in each sequence)
        D: Model dimensions (neurons, heads, or embedding channels)
    
    """
    importance_scores = {"heads": {}, "neurons": {}, "embedding_channels": {}}
    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(model.device)
            
            print_batch_info(input_ids, model)

            for name, module in model.named_modules():
                if isinstance(module, nn.MultiheadAttention):                    
                    query, key, value = module.q_proj(input_ids), module.k_proj(input_ids), module.v_proj(input_ids)
                    attn_output = torch.bmm(query, key.transpose(1, 2)) @ value
                    attn_over_seq = attn_output.mean(dim=1)
                    attn_over_batch = attn_over_seq.norm(p=2, dim=0)
                    head_importance = attn_over_batch.sum(dim=0)

                    print_MHA_info(name, query, key, value, attn_output, attn_over_seq, attn_over_batch, head_importance)
                    
                    if name not in importance_scores["heads"]:
                        importance_scores["heads"][name] = torch.zeros(head_importance.size(), device=head_importance.device)
                    importance_scores["heads"][name] += head_importance

                elif isinstance(module, nn.Linear):
                    neuron_output = module(input_ids)
                    neuron_over_seq = neuron_output.mean(dim=1)
                    neuron_over_batch = neuron_over_seq.norm(p=2, dim=0)
                    neuron_importance = neuron_over_batch.sum(dim=0)

                    print_MLP_info(neuron_output, neuron_over_seq, neuron_over_batch, neuron_importance)
                    
                    if name not in importance_scores["neurons"]:
                        importance_scores["neurons"][name] = torch.zeros(neuron_importance.size(), device=neuron_importance.device)
                    importance_scores["neurons"][name] += neuron_importance

                elif isinstance(module, nn.LayerNorm):
                    emb_output = module(input_ids)
                    emb_over_seq = emb_output.mean(dim=1)
                    emb_over_batch = emb_over_seq.norm(p=2, dim=0)
                    emb_importance = emb_over_batch.sum(dim=0)

                    print_LN_info(name, emb_output, emb_over_seq, emb_over_batch, emb_importance)

                    if name not in importance_scores["embedding_channels"]:
                        importance_scores["embedding_channels"][name] = torch.zeros(emb_importance.size(), device=emb_importance.device)
                    importance_scores["embedding_channels"][name] += emb_importance
                
                else:
                    print("\n\n\n---------------------------------------------------------\n")
                    print(f"Skipping module: {name}")
                    print("\n---------------------------------------------------------\n\n\n")
    
    return importance_scores

def compute_perplexity(model, dataset):
    """Compute perplexity (PPL) of the model on a dataset."""
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for input_data in dataset:
            output = model(input_data)
            logits = output.logits  # Assuming logits are returned
            labels = input_data["labels"]  # Adjust based on your dataset
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
            count += 1
    return torch.exp(torch.tensor(total_loss / count))

def compute_block_importance(model, dataset):
    """Compute block importance (BI) for layers."""
    activations = {}
    model.eval()
    with torch.no_grad():
        for input_data in dataset:
            output = model(input_data)
            for name, module in model.named_modules():
                if isinstance(module, nn.ModuleList):
                    if name not in activations:
                        activations[name] = {"input": [], "output": []}
                    for layer in module:
                        # Capture input and output activations for each block
                        input_act = layer(input_data)  # Adjust based on your forward logic
                        activations[name]["input"].append(input_act.cpu())
                        activations[name]["output"].append(output.cpu())
    bi_scores = {}
    for name, act in activations.items():
        X_i = torch.cat(act["input"], dim=0)
        X_i_plus_1 = torch.cat(act["output"], dim=0)
        cosine_similarity = (X_i * X_i_plus_1).sum(dim=-1) / (
            torch.norm(X_i, dim=-1) * torch.norm(X_i_plus_1, dim=-1)
        )
        bi_scores[name] = 1 - cosine_similarity.mean().item()
    return bi_scores

def structured_pruning(
    lora_model,
    calibration_dataset,
    sparsity={"width": 0.5, "depth": 0.5},
    pruning_axes=["width", "depth"],
    use_bi_for_depth=True
):
    """
    Perform structured pruning on the model.
        - Prunes MLP, ATT, and EMB layers
        - Uses *activation-based importance estimation strategy*
        - Handles width (head, neuron, embedding channel) and depth (layer/block) pruning
        - Supports Perplexity (PPL) and Block Importance (BI) for depth pruning
    """

    if "width" in pruning_axes:
        importance_scores = compute_width_importance_scores(lora_model, calibration_dataset)

        for axis, scores in importance_scores.items():
            for name, score in scores.items():
                threshold = torch.quantile(torch.tensor(score), sparsity["width"])
                mask = torch.tensor(score) >= threshold
                if axis == "heads":
                    module = dict(lora_model.named_modules())[name]
                    module.prune_heads(mask)
                elif axis == "neurons":
                    module = dict(lora_model.named_modules())[name]
                    module.weight.data *= mask.unsqueeze(1)
                elif axis == "embedding_channels":
                    module = dict(lora_model.named_modules())[name]
                    module.weight.data *= mask

    if "depth" in pruning_axes:
        if use_bi_for_depth:
            bi_scores = compute_block_importance(lora_model, calibration_dataset)
            for name, score in bi_scores.items():
                if score < sparsity["depth"]:
                    module = dict(lora_model.named_modules())[name]
                    module = nn.Identity()  # Replace block with no-op
        else:
            # Use Perplexity (PPL)
            original_perplexity = compute_perplexity(lora_model, calibration_dataset)
            layer_importance = {}
            for name, module in lora_model.named_modules():
                if isinstance(module, nn.ModuleList):
                    for i, layer in enumerate(module):
                        temp_model = lora_model.clone()
                        temp_model.module[i] = nn.Identity()
                        perplexity = compute_perplexity(temp_model, calibration_dataset)
                        layer_importance[f"{name}.{i}"] = perplexity - original_perplexity

            # Rank layers and prune
            ranked_layers = sorted(layer_importance, key=layer_importance.get, reverse=True)
            num_layers_to_prune = int(len(ranked_layers) * sparsity["depth"])
            for layer in ranked_layers[:num_layers_to_prune]:
                name, idx = layer.split(".")
                idx = int(idx)
                dict(lora_model.named_modules())[name].module[int(idx)] = nn.Identity()

    return lora_model



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

def prune_and_knowledge_distillation(train_data, valid_data, test_data):
    """Prune the fine-tuned model and perform knowledge distillation."""
    
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
    calibration_dataset = create_calibration_dataset(train_data, tokenizer, num_samples=1024, batch_size=32)

    # Structured pruning
    print("\n\n\n------------------ model pruning ------------------\n\n\n")
    sparsity_config = {"width": 0.5, "depth": 0.3}
    pruned_model = structured_pruning(
        lora_model,
        calibration_dataset,
        sparsity=sparsity_config,
        pruning_axes=["width", "depth"],
        use_bi_for_depth=True
    )

    # Knowledge distillation setup
    teacher_model = lora_model
    optimizer = torch.optim.AdamW(pruned_lora_model.parameters(), lr=5e-5)
    loss_fn = nn.MSELoss()
    pruned_model = pruned_lora_model
    pruned_model.train()
    
    # Training the pruned model
    num_epochs = 3
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_data:
            inputs = batch["input_ids"].to(pruned_model.device)
            attention_mask = batch["attention_mask"].to(pruned_model.device)
            
            # Teacher outputs
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=inputs, attention_mask=attention_mask).logits
            
            # Student outputs
            student_outputs = pruned_model(input_ids=inputs, attention_mask=attention_mask).logits
            loss = loss_fn(student_outputs, teacher_outputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    pruned_lora_model = pruned_model

    print("\n\-----------Done KD-----------\n")


    # Save the pruned model
    PRUNED_ADAPTER_MODEL = "pruned_lora_adapter"
    pruned_lora_model.save_pretrained(PRUNED_ADAPTER_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype=torch.float16)
    final_model = PeftModel.from_pretrained(base_model, PRUNED_ADAPTER_MODEL)
    final_model.save_pretrained("gemma-2b-it-sst2-pruned")