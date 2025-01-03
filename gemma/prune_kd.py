import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TrainingArguments, AutoModelForSequenceClassification
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from transformers.trainer_utils import set_seed

# calibration_dataset

def compute_width_importance_scores(model, data_loader, sparsity=0.5):
    """
    Compute importance scores for heads, neurons, and embedding channels.
    """
    importance_scores = {"heads": {}, "neurons": {}, "embeddings": {}}
    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            for name, module in model.named_modules():
                if isinstance(module, nn.MultiheadAttention):
                    # Compute importance for heads
                    query, key, value = module.q_proj, module.k_proj, module.v_proj
                    attn_output = torch.bmm(query(batch), key(batch).transpose(1, 2)) @ value(batch)
                    head_importance = attn_output.norm(dim=(0, 1))  # Aggregating across batch, seq
                    importance_scores["heads"][name] = head_importance

                elif isinstance(module, nn.Linear):
                    # Compute importance for neurons
                    weight = module.weight
                    neuron_importance = weight.norm(dim=1)  # L2 norm across weights of each neuron
                    importance_scores["neurons"][name] = neuron_importance

                elif isinstance(module, nn.LayerNorm):
                    # Compute importance for embedding channels
                    ln_output = module(batch)
                    embedding_importance = ln_output.norm(dim=0)  # L2 norm across batch, seq
                    importance_scores["embeddings"][name] = embedding_importance

    # Aggregate scores across layers
    for key in importance_scores.keys():
        for layer_name, scores in importance_scores[key].items():
            importance_scores[key][layer_name] = scores.mean().item()  # Example aggregation

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
    use_bi_for_depth=False
):
    """
    Perform structured pruning on the model.
        - Prunes MLP, ATT, and EMB layers
        - Uses activation-based importance estimation strategy
        - Handles width (head, neuron, embedding channel) and depth (layer/block) pruning
        - Supports Perplexity (PPL) and Block Importance (BI) for depth pruning
    """

    # Compute width importance scores if width pruning is selected
    if "width" in pruning_axes:
        importance_scores = compute_width_importance_scores(
            lora_model, calibration_dataset, sparsity=sparsity["width"]
        )

        # Prune heads, neurons, and embedding channels based on scores
        for axis, scores in importance_scores.items():
            for name, score in scores.items():
                # Prune based on sparsity threshold
                threshold = torch.quantile(torch.tensor(score), sparsity["width"])
                mask = torch.tensor(score) >= threshold
                if axis == "heads":
                    # Prune heads in MultiheadAttention
                    module = dict(lora_model.named_modules())[name]
                    module.prune_heads(mask)
                elif axis == "neurons":
                    # Prune neurons in Linear layers
                    module = dict(lora_model.named_modules())[name]
                    module.weight.data *= mask.unsqueeze(1)
                elif axis == "embeddings":
                    # Prune embedding channels in LayerNorm
                    module = dict(lora_model.named_modules())[name]
                    module.weight.data *= mask

    # Compute depth importance scores if depth pruning is selected
    if "depth" in pruning_axes:
        if use_bi_for_depth:
            # Use Block Importance (BI)
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

def prune_and_knowledge_distillation(valid_data, test_data):
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
    
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)
    
    # dataloader setup from the validation dataset
    tokenized_train = valid_data.map(tokenize_function, batched=True)
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dataloader = DataLoader(tokenized_train, batch_size=2, shuffle=True)

    # Manual pruning
    print("\n\n\n------------------ model pruning ------------------\n\n\n")
    sparsity_config = {"width": 0.5, "depth": 0.3}
    pruned_model = structured_pruning(
        lora_model,
        dataloader,
        sparsity=sparsity_config,
        pruning_axes=["width", "depth"],
        use_bi_for_depth=True  # Use Block Importance for depth pruning
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
        for batch in dataloader:
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