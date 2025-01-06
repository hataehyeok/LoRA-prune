import torch
from itertools import product
from transformers import AutoTokenizer, AutoModelForCausalLM
from prune_kd import structured_pruning, knowledge_distill, create_calibration_dataset
from ft import dataset_loading


def lightweight_nas(search_space, parameter_budget, base_model, train_data, valid_data, tokenizer, num_samples=1024, batch_size=32):
    """
    Perform Lightweight Neural Architecture Search (NAS) to find optimal configurations.

    Args:
        search_space (dict): Search space for architecture dimensions.
            Example: {
                "embedding_dim": [128, 256, 512],
                "num_heads": [4, 8, 16],
                "num_neurons": [128, 256, 512]
            }
        parameter_budget (int): Maximum parameter count allowed for candidates.
        base_model: Pre-trained base model to modify for NAS.
        train_data: Training dataset.
        valid_data: Validation dataset.
        tokenizer: Tokenizer for input data.
        num_samples (int): Number of samples for the calibration dataset.
        batch_size (int): Batch size for evaluation.

    Returns:
        dict: Best configuration and validation loss.
    """
    # Step 1: Generate feasible architectures
    architectures = list(product(*search_space.values()))
    candidates = []
    for arch in architectures:
        model_size = estimate_model_size(arch, base_model)
        if model_size <= parameter_budget:
            candidates.append(arch)
    print(f"Generated {len(candidates)} candidates within the parameter budget.")

    # Step 2: Evaluate each architecture
    calibration_dataset = create_calibration_dataset(train_data, tokenizer, num_samples, batch_size)
    best_loss = float("inf")
    best_architecture = None

    for arch in candidates:
        print(f"Evaluating architecture: {arch}")
        modified_model = modify_model_architecture(base_model, arch)
        pruned_model = iterative_pruning(modified_model, calibration_dataset, train_data, iter=4)
        val_loss = evaluate_model(pruned_model, valid_data, tokenizer)

        print(f"Validation Loss for {arch}: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_architecture = arch

    print(f"Best Architecture: {best_architecture}, Validation Loss: {best_loss:.4f}")
    return {"best_architecture": best_architecture, "validation_loss": best_loss}

def estimate_model_size(architecture, base_model):
    """
    Estimate the number of parameters for a given architecture.
    """
    embedding_dim, num_heads, num_neurons = architecture
    return embedding_dim * num_heads + num_neurons * base_model.config.hidden_size

def modify_model_architecture(base_model, architecture):
    """
    Modify the base model to match the given architecture.
    """
    embedding_dim, num_heads, num_neurons = architecture
    base_model.config.hidden_size = embedding_dim
    base_model.config.num_attention_heads = num_heads
    base_model.config.intermediate_size = num_neurons
    base_model.resize_token_embeddings(embedding_dim)
    return base_model

def iterative_pruning(model, calibration_dataset, train_data, iter=4):
    """
    Perform iterative pruning and importance estimation on the model.
    """
    for i in range(iter):
        d_s = 1.0
        d_t = 0.5
        importance_iter = d_s - (i * ((d_s - d_t) / iter))
        prune_iter = d_s - (i + 1) * ((d_s - d_t) / iter)
        sparsity = {"width": prune_iter, "depth": prune_iter}
        model = structured_pruning(model, calibration_dataset, sparsity, importance_iter)
        model = knowledge_distill(model, model, train_data)
    return model

def evaluate_model(model, valid_data, tokenizer):
    """
    Evaluate the model on validation data and return loss.
    """
    model.eval()
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in DataLoader(valid_data, batch_size=32):
            inputs = tokenizer(batch["sentence"], return_tensors="pt", padding=True, truncation=True, max_length=128)
            labels = batch["label"]
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
    return total_loss / len(valid_data)

# Example Usage
if __name__ == "__main__":
    search_space = {
        "embedding_dim": [128, 256, 512],
        "num_heads": [4, 8, 16],
        "num_neurons": [128, 256, 512]
    }
    parameter_budget = 1e9
    train_data, valid_data, _ = dataset_loading()
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")

    best_architecture = lightweight_nas(search_space, parameter_budget, base_model, train_data, valid_data, tokenizer)
    print(best_architecture)