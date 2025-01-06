import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
import random

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU()
        )
        self.layer2 = nn.Linear(20, 10)


def create_mock_calibration_dataset(batch_size=2, num_samples=10):
    """
    Create a mock calibration dataset with random data for testing.
    """
    dataset = []
    for _ in range(num_samples):
        dataset.append({
            "input_ids": torch.randn(1, 10),  # Simulating tokenized input (10 features)
            "attention_mask": torch.ones(1, 10)  # Dummy attention mask
        })
    return DataLoader(dataset, batch_size=batch_size)

# Mock BI Scores for Testing
mock_bi_scores = {
    "layer1.0": 0.2,  # First layer in Sequential (Linear)
    "layer2": 0.8     # Second Linear layer
}

# Mock Block Importance Function
def mock_compute_block_importance(model, data_loader):
    """
    Mock block importance computation for testing.
    """
    return mock_bi_scores

# Mock Pruning Function for Depth
def depth_pruning(model, bi_scores, sparsity):
    """
    Perform depth pruning on the model using mock BI scores.
    """
    threshold = torch.quantile(torch.tensor(list(bi_scores.values())), sparsity)
    print(f"Pruning threshold: {threshold}")
    
    for name, bi_score in bi_scores.items():
        if bi_score < threshold:
            # Split name to get parent and layer
            parent_name, layer_name = name.rsplit(".", 1)
            parent_module = dict(model.named_modules())[parent_name]
            setattr(parent_module, layer_name, nn.Identity())
            print(f"Pruned layer: {name}")
    return model

# Test the Entire Workflow
def test_workflow():
    # Initialize model
    lora_model = MyModel()
    print("Original Model Structure:")
    print(lora_model)
    
    # Create a mock calibration dataset
    calibration_dataset = create_mock_calibration_dataset()
    
    # Use mock block importance function
    bi_scores = mock_compute_block_importance(lora_model, calibration_dataset)
    print("\nMock BI Scores:")
    print(bi_scores)
    
    # Perform depth pruning
    sparsity = 0.5  # Prune 50% of the least important layers
    pruned_model = depth_pruning(lora_model, bi_scores, sparsity)
    
    print("\nPruned Model Structure:")
    print(pruned_model)

# Run the Test
if __name__ == "__main__":
    test_workflow()