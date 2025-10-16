
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
from avalanche.benchmarks.classic import SplitMNIST


class SimpleMLP(nn.Module):
    """
    Simple MLP with shared parameters for all tasks
    """
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super(SimpleMLP, self).__init__()
        
        # Simple MLP architecture
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor
        """
        x = x.view(x.size(0), -1)  # Flatten input
        return self.model(x)


def compute_empirical_fisher_information(model, dataset, device='cpu', num_samples=200):
    """
    Compute Fisher Information Matrix for the model
    
    Args:
        model: SimpleMLP model
        dataset: Dataset for the task
        device: Device to run on
        num_samples: Number of samples to use
        
    Returns:
        fisher_dict: Dictionary of Fisher information per parameter
    """
    print(f"Computing Fisher Information with {num_samples} samples...")
    
    model = model.to(device)
    model.eval()
    
    # Initialize Fisher Information dictionary
    fisher_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param.data).to(device)
    
    # Create DataLoader with limited samples
    if num_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:num_samples]
        from torch.utils.data import Subset
        subset_dataset = Subset(dataset, indices.tolist())
        dataloader = DataLoader(subset_dataset, batch_size=1, shuffle=False, num_workers=0)
    else:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    processed_samples = 0
    
    # Process samples
    for batch_idx, batch in enumerate(dataloader):
        # Handle Avalanche dataset format
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch[0], batch[1]
        else:
            continue
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero gradients
        model.zero_grad()
        
        # Forward pass
        output = model(inputs)
        
        # Compute log probability loss
        log_prob = F.log_softmax(output, dim=1)
        loss = F.nll_loss(log_prob, targets)
        
        # Backward pass
        loss.backward()
        
        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data ** 2
        
        processed_samples += 1
        
        if processed_samples % 50 == 0:
            print(f"  Processed {processed_samples} samples...")
    
    # Average over all samples
    for name in fisher_dict:
        fisher_dict[name] /= processed_samples
    
    print(f"Fisher Information computed for {processed_samples} samples!")
    return fisher_dict


def save_important_weights(model, fisher_dict, top_n_percent=10.0, save_path="important_weights.pt"):
    """
    Save top N% important weights based on Fisher information
    
    Args:
        model: PyTorch model
        fisher_dict: Fisher information dictionary  
        top_n_percent: Percentage of top weights to save
        save_path: Path to save weights
        
    Returns:
        important_weights_data: Dictionary with masks and values
    """
    print(f"Saving top {top_n_percent}% important weights...")
    
    if top_n_percent == 0:
        # Save all weights
        print("Saving all weights (top_n_percent=0)")
        important_weights_data = {}
        total_weights = 0
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in fisher_dict:
                    # Save all weights
                    mask = torch.ones_like(fisher_dict[name], dtype=torch.bool)
                    important_values = param.data.clone()
                    
                    important_weights_data[name] = {
                        'mask': mask.cpu(),
                        'values': important_values.cpu()
                    }
                    total_weights += important_values.numel()
        
        torch.save(important_weights_data, save_path)
        print(f"Saved all {total_weights:,} weights to {save_path}")
        
    else:
        # Find global threshold
        all_scores = torch.cat([f.view(-1) for f in fisher_dict.values()])
        threshold_quantile = 1.0 - (top_n_percent / 100.0)
        threshold = torch.quantile(all_scores, threshold_quantile)
        
        important_weights_data = {}
        total_important_weights = 0
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in fisher_dict:
                    # Create mask for important weights
                    mask = fisher_dict[name] >= threshold
                    important_values = param.data[mask]
                    
                    important_weights_data[name] = {
                        'mask': mask.cpu(),
                        'values': important_values.cpu()
                    }
                    total_important_weights += important_values.numel()
        
        torch.save(important_weights_data, save_path)
        print(f"Saved {total_important_weights:,} important weights to {save_path}")
    
    return important_weights_data


def restore_important_weights(model, important_weights_data, device):
    """
    Temporarily restore important weights for inference
    
    Args:
        model: PyTorch model
        important_weights_data: Dictionary with masks and values
        device: Device to restore weights on
        
    Returns:
        original_weights: Dictionary to store original weights for restoration
    """
    # Store original weights
    original_weights = {}
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in important_weights_data:
                # Store original values
                data = important_weights_data[name]
                mask = data['mask'].to(device)
                original_weights[name] = param.data[mask].clone()
                
                # Restore important weights
                saved_values = data['values'].to(device)
                param.data[mask] = saved_values
    
    return original_weights


def revert_to_original_weights(model, original_weights, important_weights_data, device):
    """
    Revert model back to original weights after task-specific inference
    
    Args:
        model: PyTorch model
        original_weights: Dictionary with original weight values
        important_weights_data: Dictionary with masks
        device: Device
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_weights:
                mask = important_weights_data[name]['mask'].to(device)
                param.data[mask] = original_weights[name]


def evaluate_task_with_specific_weights(model, test_experience, important_weights_path, device):
    """
    Evaluate a specific task using its important weights
    
    Args:
        model: SimpleMLP model
        test_experience: Avalanche test experience
        important_weights_path: Path to saved important weights
        device: Device
        
    Returns:
        accuracy: Task accuracy
    """
    # Load important weights for this task
    important_weights_data = torch.load(important_weights_path, map_location='cpu')
    
    # Temporarily restore task-specific important weights
    original_weights = restore_important_weights(model, important_weights_data, device)
    
    # Evaluate on the task
    model.eval()
    correct = 0
    total = 0
    
    test_loader = DataLoader(test_experience.dataset, batch_size=100, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                continue
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100.0 * correct / total
    
    # Revert back to original weights
    revert_to_original_weights(model, original_weights, important_weights_data, device)
    
    return accuracy


def create_split_mnist_benchmark():
    """
    Create SplitMNIST benchmark
    
    Returns:
        benchmark: SplitMNIST benchmark
    """
    return SplitMNIST(n_experiences=5, return_task_id=False, seed=42, shuffle=False)
