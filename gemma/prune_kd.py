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
from ft import dataset_loading, fine_tuning

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
# importance_iter
def compute_width_importance_scores(model, data_loader, importance_iter):
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

            for name, module in model.named_modules():
                if isinstance(module, nn.MultiheadAttention):                    
                    query, key, value = module.q_proj(input_ids), module.k_proj(input_ids), module.v_proj(input_ids)
                    attn_output = torch.bmm(query, key.transpose(1, 2)) @ value
                    attn_over_seq = attn_output.mean(dim=1)
                    attn_over_batch = attn_over_seq.norm(p=2, dim=0)
                    head_importance = attn_over_batch.sum(dim=0)
                    
                    if name not in importance_scores["heads"]:
                        importance_scores["heads"][name] = torch.zeros(head_importance.size(), device=head_importance.device)
                    importance_scores["heads"][name] += head_importance

                elif isinstance(module, nn.Linear):
                    neuron_output = module(input_ids)
                    neuron_over_seq = neuron_output.mean(dim=1)
                    neuron_over_batch = neuron_over_seq.norm(p=2, dim=0)
                    neuron_importance = neuron_over_batch.sum(dim=0)
                    
                    if name not in importance_scores["neurons"]:
                        importance_scores["neurons"][name] = torch.zeros(neuron_importance.size(), device=neuron_importance.device)
                    importance_scores["neurons"][name] += neuron_importance

                elif isinstance(module, nn.LayerNorm):
                    emb_output = module(input_ids)
                    emb_over_seq = emb_output.mean(dim=1)
                    emb_over_batch = emb_over_seq.norm(p=2, dim=0)
                    emb_importance = emb_over_batch.sum(dim=0)

                    if name not in importance_scores["embedding_channels"]:
                        importance_scores["embedding_channels"][name] = torch.zeros(emb_importance.size(), device=emb_importance.device)
                    importance_scores["embedding_channels"][name] += emb_importance
                
                else:
                    raise ValueError(f"Skipping module: {name}")
    
    return importance_scores

# TODO
# perplexity vs. block importance
# importance_iter
def compute_block_importance(model, data_loader, importance_iter):
    """
    Compute block importance (BI) for layers.
    BI is computed as 1 - cosine similarity between the input and output of the layer.

    Iterates over batches
        input: a batch of input data from the DataLoader
        output: is used to compute the layer activation

    """
    activation = {}
    model.eval()
    with torch.no_grad():
        for input in data_loader:
            output = model(input)
            for name, module in model.named_modules():
                if isinstance(module, nn.ModuleList):
                    if name not in activation:
                        activation[name] = {"input": [], "output": []}
                    for layer in module:
                        input_act = layer(input)
                        activation[name]["input"].append(input_act.cpu())
                        activation[name]["output"].append(output.cpu())
    bi_scores = {}
    for name, act in activation.items():
        X_i = torch.cat(act["input"], dim=0)
        X_i_plus_1 = torch.cat(act["output"], dim=0)
        cosine_similarity = (X_i * X_i_plus_1).mean(dim=-1) / (
            torch.norm(X_i, dim=-1) * torch.norm(X_i_plus_1, dim=-1)
        )
        bi_scores[name] = 1 - cosine_similarity
    return bi_scores

def width_pruning(lora_adapter, calibration_dataset, width_sparsity, importance_iter):
    """
    Perform width pruning on the model, directly modifying its layers (heads, neurons, embedding channels).
    """
    if width_sparsity == 0:
        return lora_adapter
    
    importance_scores = compute_width_importance_scores(lora_adapter, calibration_dataset, importance_iter)
    
    for axis, scores in importance_scores.items():
        threshold = torch.quantile(torch.tensor(list(scores.values())), width_sparsity)
        
        for name, score in scores.items():
            if score < threshold:
                if "." in name:
                    parent_name, layer_name = name.rsplit(".", 1)
                    parent_module = dict(lora_adapter.named_modules())[parent_name]
                else:
                    parent_name, layer_name = "", name
                    parent_module = lora_adapter

                module = getattr(parent_module, layer_name)

                if axis == "heads" and hasattr(module, "prune_heads"):
                    heads_to_keep = torch.tensor([score >= threshold for score in scores.values()])
                    module.prune_heads(heads_to_keep)
                elif axis == "neurons":
                    mask = score >= threshold
                    module.weight.data *= mask.unsqueeze(1)
                elif axis == "embedding_channels":
                    mask = score >= threshold
                    module.weight.data *= mask
                else:
                    raise ValueError(f"Invalid axis or pruning not supported for module: {name}")

                if axis in ["heads", "neurons", "embedding_channels"]:
                    setattr(parent_module, layer_name, module)

    return lora_adapter

def depth_pruning(lora_adapter, calibration_dataset, depth_sparsity, importance_iter):
    if depth_sparsity == 0:
        return lora_adapter
    
    bi_scores = compute_block_importance(lora_adapter, calibration_dataset, importance_iter)
    threshold = torch.quantile(torch.tensor(list(bi_scores.values())), depth_sparsity)

    for name, act in bi_scores.items():
        if act < threshold:
            parent_name, layer_name = name.rsplit('.', 1)
            parent_module = dict(lora_adapter.named_modules())[parent_name]
            setattr(parent_module, layer_name, nn.Identity())
    
    return lora_adapter

def structured_pruning(lora_adapter, calibration_dataset, sparsity, importance_iter):
    """
    Perform structured pruning on the model.
        - Prunes MLP, ATT, and EMB layers
        - Uses *activation-based importance estimation strategy*
        - Handles width (head, neuron, embedding channel) and depth (layer/block) pruning
        - Supports Perplexity (PPL) and Block Importance (BI) for depth pruning
    """

    width_pruned_lora_adapter = width_pruning(lora_adapter, calibration_dataset, sparsity["width"], importance_iter)
    depth_pruned_lora_adapter = depth_pruning(width_pruned_lora_adapter, calibration_dataset, sparsity["depth"], importance_iter)

    return depth_pruned_lora_adapter

def conventional_retraining(lora_adapter, train_data, valid_data):
    #TODO but not to now
    return lora_adapter

def logit_kd_loss(teacher_logits, student_logits, temperature=1.0):
    """
    Compute logit-based KD loss averaged over the sequence length (L_logits).

    Args:
        teacher_logits (torch.Tensor): Teacher logits (B, S, V).
        student_logits (torch.Tensor): Student logits (B, S, V).
        temperature (float): Softmax temperature.

    Returns:
        torch.Tensor: Logit-based KD loss (averaged over sequence length).
    """
    # Compute teacher and student probabilities with temperature scaling
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)  # (B, S, V)
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)  # (B, S, V)

    # Compute KL divergence loss per token (across vocabulary)
    kl_div_loss = F.kl_div(student_probs, teacher_probs, reduction="none")  # (B, S, V)

    # Sum over vocabulary (dim=-1) to get loss per token, then average over sequence length
    token_losses = kl_div_loss.sum(dim=-1).mean(dim=1)  # (B,)
    sequence_loss = token_losses.mean()  # Scalar (averaged over batch)

    return sequence_loss

def intermediate_state_kd_loss(teacher_hidden_states, student_hidden_states, linear_mapping, selected_layers):
    """
    Compute intermediate state KD loss (L_is) averaged over the sequence length.

    Args:
        teacher_hidden_states (List[torch.Tensor]): Teacher hidden states (B, S, D_t).
        student_hidden_states (List[torch.Tensor]): Student hidden states (B, S, D_s).
        linear_mapping (torch.nn.Linear): Linear transformation for student states.
        selected_layers (List[int]): Indices of layers to use in the loss.

    Returns:
        torch.Tensor: Intermediate state KD loss (averaged over sequence length).
    """
    total_loss = 0.0
    sequence_length = teacher_hidden_states[0].size(1)
    batch_size = teacher_hidden_states[0].size(0)

    for k in selected_layers:
        # Select hidden states for the kth layer
        teacher_state = teacher_hidden_states[k]  # (B, S, D_t)
        student_state = student_hidden_states[k]  # (B, S, D_s)

        # Transform student state to match teacher's dimensionality
        transformed_student_state = linear_mapping(student_state)  # (B, S, D_t)

        # Compute loss per token and sum over sequence
        token_losses = F.mse_loss(transformed_student_state, teacher_state, reduction="none")  # (B, S, D_t)
        token_losses = token_losses.sum(dim=-1).mean(dim=1)  # Sum over dimensions, average over sequence (B,)

        # Accumulate the layer loss
        total_loss += token_losses.mean()  # Average over batch

    # Average over sequence length and number of layers
    total_loss /= (len(selected_layers) * sequence_length)

    return total_loss

def total_kd_loss(teacher_model, student_model, batch, linear_mapping, selected_layers, alpha=0.5, temperature=1.0):
    """
    Compute the total KD loss.

    Args:
        teacher_model: Teacher model.
        student_model: Student model.
        batch: Input batch with 'input_ids', 'attention_mask', and 'labels'.
        linear_mapping: Linear transformation layer.
        selected_layers: Layers to use for intermediate state loss.
        alpha: Weight for intermediate state KD loss.
        temperature: Temperature for logit-based KD loss.

    Returns:
        torch.Tensor: Total KD loss.
    """
    inputs = batch["input_ids"].to(student_model.device)
    attention_mask = batch["attention_mask"].to(student_model.device)
    labels = batch["labels"].to(student_model.device)

    # Teacher outputs
    with torch.no_grad():
        teacher_outputs = teacher_model(input_ids=inputs, attention_mask=attention_mask, output_hidden_states=True)
        teacher_logits = teacher_outputs.logits
        teacher_hidden_states = teacher_outputs.hidden_states

    # Student outputs
    student_outputs = student_model(input_ids=inputs, attention_mask=attention_mask, output_hidden_states=True)
    student_logits = student_outputs.logits
    student_hidden_states = student_outputs.hidden_states

    # Loss components
    loss_clm = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
    loss_logits = logit_kd_loss(teacher_logits, student_logits, temperature)
    loss_intermediate = intermediate_state_kd_loss(teacher_hidden_states, student_hidden_states, linear_mapping, selected_layers)

    # Total loss
    total_loss = loss_clm + loss_logits + alpha * loss_intermediate
    return total_loss

# TODO
# alpha value
def knowledge_distill(lora_adapter, pruned_adapter, train_data, alpha=0.5, temperature=1.0, selected_layers=None):
    """
    Perform knowledge distillation with logit and intermediate state KD losses.

    Args:
        lora_adapter: Teacher model.
        pruned_adapter: Student model.
        train_data: Training dataset.
        alpha: Weighting coefficient for intermediate state loss.
        temperature: Temperature for logit-based KD loss.
        selected_layers: Layers to use for intermediate state loss.

    Returns:
        pruned_adapter: The distilled student model.
    """
    teacher_model = lora_adapter
    student_model = pruned_adapter
    optimizer = Adam(student_model.parameters(), lr=5e-5)
    linear_mapping = nn.Linear(student_model.config.hidden_size, teacher_model.config.hidden_size).to(student_model.device)

    student_model.train()
    teacher_model.eval()

    for batch in DataLoader(train_data, batch_size=32):
        total_loss = total_kd_loss(teacher_model, student_model, batch, linear_mapping, selected_layers, alpha, temperature)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return student_model

# TODO
# setup the vary sparsity
#   Instead of performing one-shot pruning (pruning everything in a single step)
#   , iterative pruning splits the process into multiple iterations (T) to allow the model to gradually adapt.
def prune_and_knowledge_distillation(train_data, valid_data, test_data, is_iter_prune = False, is_process_kd = True, iter = 4):
    """
    Prune the fine-tuned model and perform knowledge distillation.
    
    Iterative Importance Estimation and Pruning
        - iter num -> 4
        - measure the initial validation loss and final validation loss with various iteration

    Retraining
        - conventional retraining, leveraging ground-truth labels
        - knowledge distillation using supervision from the unpruned model (teacher)
    """
    
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
    lora_adapter = PeftModel.from_pretrained(base_model, ADAPTER_MODEL, device_map="auto", torch_dtype=torch.float16)
    calibration_dataset = create_calibration_dataset(train_data, tokenizer, num_samples=1024, batch_size=32)
    
    

    if is_iter_prune:
        for i in range(iter):
            d_s = 1.0
            d_t = 0.5
            importance_iter = d_s - (i * ((d_s - d_t) / iter))
            prune_iter = d_s - (i + 1) * ((d_s - d_t) / iter)
            #TODO
            # sparsity have to have the list of iterated sparsity
            sparsity = {"width": prune_iter, "depth": prune_iter}
            pruned_adapter = structured_pruning(lora_adapter, calibration_dataset, sparsity, importance_iter)
            if is_process_kd:
                kd_adapter = knowledge_distill(lora_adapter, pruned_adapter, train_data)
            else:
                kd_adapter = conventional_retraining(pruned_adapter, train_data, valid_data)
            
            lora_adapter = kd_adapter
    else:
        sparsity = {"width": 0.5, "depth": 0}
        pruned_adapter = structured_pruning(lora_adapter, calibration_dataset, sparsity, 1)
        if is_process_kd:
            kd_adapter = knowledge_distill(lora_adapter, pruned_adapter, train_data)
        else:
            kd_adapter = conventional_retraining(pruned_adapter, train_data, valid_data)
    
    PRUNED_ADAPTER_MODEL = "pruned_lora_adapter"
    kd_adapter.save_pretrained(PRUNED_ADAPTER_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype=torch.float16)
    final_model = PeftModel.from_pretrained(base_model, PRUNED_ADAPTER_MODEL)
    final_model.save_pretrained("gemma-2b-it-sst2-pruned")
