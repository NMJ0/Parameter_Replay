import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def compute_empirical_fisher_information(model, dataset, device='cpu', num_samples=None, batch_size=1):
    """
    Compute the Empirical Fisher Information Matrix for given model and dataset.
    
    This implementation follows the empirical Fisher approach where we compute:
    F_ii = (1/N) * sum_n (∂log p(y_n|x_n) / ∂θ_i)^2
    
    where y_n is the ground-truth label for sample x_n.
    
    Args:
        model: PyTorch model
        dataset: PyTorch Dataset or Avalanche dataset (experience.dataset)
        device: Device to run computations on
        num_samples: Maximum number of samples to use (None for all samples in dataset)
        batch_size: Batch size for DataLoader (default=1 for per-sample gradients)
        
    Returns:
        fisher_dict: Dictionary mapping parameter names to their Fisher Information values
    """
    
    # Ensure model is on the correct device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize Fisher Information dictionary
    fisher_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param.data).to(device)
    
    # Create DataLoader from dataset
    # Limit samples if specified
    if num_samples is not None and num_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:num_samples]
        from torch.utils.data import Subset
        subset_dataset = Subset(dataset, indices.tolist())
        dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        total_samples = num_samples
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        total_samples = len(dataset)
    
    print(f"Computing Empirical Fisher Information for {total_samples} samples...")
    
    processed_samples = 0
    
    # Process samples
    for batch_idx, batch in enumerate(dataloader):
        # Handle different batch formats (Avalanche datasets might return tuples)
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:
                print(f"Unexpected batch format: {batch}")
                continue
        else:
            print(f"Unexpected batch format: {type(batch)}")
            continue
        
        # Move to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        current_batch_size = inputs.size(0)
        
        # Process each sample in the batch individually for true empirical Fisher
        for i in range(current_batch_size):
            # Zero gradients
            model.zero_grad()
            
            # Get single sample
            x_i = inputs[i:i+1]  # Keep batch dimension
            y_i = targets[i:i+1]
            
            # Forward pass
            output = model(x_i)
            
            # Compute log probability of ground-truth class
            log_prob = F.log_softmax(output, dim=1)
            loss = F.nll_loss(log_prob, y_i)
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Accumulate squared gradients (Fisher Information)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2
            
            processed_samples += 1
            
            if processed_samples % 100 == 0:
                print(f"Processed {processed_samples}/{total_samples} samples...")
    
    # Average over all samples
    for name in fisher_dict:
        fisher_dict[name] /= processed_samples
    
    print(f"Empirical Fisher Information computed successfully for {processed_samples} samples!")
    return fisher_dict


def compute_empirical_fisher_information_batched(model, dataset, device='cpu', num_samples=None, batch_size=32):
    """
    Batched version of empirical Fisher Information computation.
    This is faster but less accurate than the per-sample version above.
    
    Args:
        model: PyTorch model
        dataset: PyTorch Dataset or Avalanche dataset (experience.dataset)
        device: Device to run computations on
        num_samples: Maximum number of samples to use (None for all samples in dataset)
        batch_size: Batch size for processing
        
    Returns:
        fisher_dict: Dictionary mapping parameter names to their Fisher Information values
    """
    
    # Ensure model is on the correct device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize Fisher Information dictionary
    fisher_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param.data).to(device)
    
    # Create DataLoader from dataset
    if num_samples is not None and num_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:num_samples]
        from torch.utils.data import Subset
        subset_dataset = Subset(dataset, indices.tolist())
        dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        total_samples = num_samples
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        total_samples = len(dataset)
    
    print(f"Computing Batched Empirical Fisher Information for {total_samples} samples...")
    
    num_batches = 0
    
    # Process in batches
    for batch_idx, batch in enumerate(dataloader):
        # Handle different batch formats (Avalanche datasets might return tuples)
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:
                print(f"Unexpected batch format: {batch}")
                continue
        else:
            print(f"Unexpected batch format: {type(batch)}")
            continue
        
        # Move to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero gradients
        model.zero_grad()
        
        # Forward pass
        output = model(inputs)
        
        # Compute log probability of ground-truth classes
        log_prob = F.log_softmax(output, dim=1)
        loss = F.nll_loss(log_prob, targets)
        
        # Backward pass to compute gradients
        loss.backward()
        
        # Accumulate squared gradients (Fisher Information)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data ** 2
        
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed batch {batch_idx + 1}...")
    
    # Average over all batches
    for name in fisher_dict:
        fisher_dict[name] /= num_batches
    
    print(f"Batched Empirical Fisher Information computed successfully for {num_batches} batches!")
    return fisher_dict


def plot_fisher_information_heatmap(fisher_dict, model=None, save_path=None, figsize=(12, 8), 
                                   top_n_params=None, metric='mean', log_scale=True):
    """
    Plot a heatmap of Fisher Information importance values.
    
    Args:
        fisher_dict: Dictionary mapping parameter names to their Fisher Information tensors
        model: PyTorch model (optional, used for getting parameter shapes and info)
        save_path: Path to save the plot (optional)
        figsize: Figure size tuple (width, height)
        top_n_params: Show only top N most important parameters (None for all)
        metric: Metric to use for importance ('mean', 'max', 'sum', 'std')
        log_scale: Whether to use log scale for color mapping
        
    Returns:
        fig: matplotlib figure object
    """
    
    if len(fisher_dict) == 0:
        print("Error: fisher_dict is empty!")
        return None
    
    # Extract importance values for each parameter
    param_data = []
    
    for param_name, fisher_tensor in fisher_dict.items():
        # Move to CPU if on GPU
        fisher_cpu = fisher_tensor.cpu() if fisher_tensor.is_cuda else fisher_tensor
        
        # Calculate different metrics
        if metric == 'mean':
            importance = fisher_cpu.mean().item()
        elif metric == 'max':
            importance = fisher_cpu.max().item()
        elif metric == 'sum':
            importance = fisher_cpu.sum().item()
        elif metric == 'std':
            importance = fisher_cpu.std().item()
        else:
            importance = fisher_cpu.mean().item()  # default to mean
        
        # Get parameter info
        param_shape = fisher_tensor.shape
        param_size = fisher_tensor.numel()
        
        # Determine layer type from parameter name
        if 'weight' in param_name:
            param_type = 'Weight'
        elif 'bias' in param_name:
            param_type = 'Bias'
        else:
            param_type = 'Other'
        
        param_data.append({
            'Parameter': param_name,
            'Importance': importance,
            'Type': param_type,
            'Shape': str(param_shape),
            'Size': param_size
        })
    
    # Create DataFrame
    df = pd.DataFrame(param_data)
    
    # Sort by importance
    df = df.sort_values('Importance', ascending=False)
    
    # Limit to top N parameters if specified
    if top_n_params is not None:
        df = df.head(top_n_params)
        title_suffix = f" (Top {top_n_params})"
    else:
        title_suffix = ""
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Horizontal bar chart
    y_pos = np.arange(len(df))
    bars = ax1.barh(y_pos, df['Importance'], color=['skyblue' if t == 'Weight' else 'lightcoral' for t in df['Type']])
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([name[:25] + '...' if len(name) > 25 else name for name in df['Parameter']], fontsize=10)
    ax1.set_xlabel(f'Fisher Information ({metric.capitalize()})', fontsize=12)
    ax1.set_title(f'Parameter Importance{title_suffix}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    if log_scale and df['Importance'].min() > 0:
        ax1.set_xscale('log')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, df['Importance'])):
        ax1.text(value, bar.get_y() + bar.get_height()/2, f'{value:.2e}', 
                ha='left', va='center', fontsize=8, fontweight='bold')
    
    # Plot 2: Heatmap-style visualization
    # Create a matrix for heatmap (reshape data for better visualization)
    importance_values = df['Importance'].values.reshape(-1, 1)
    
    # Create heatmap
    im = ax2.imshow(importance_values, cmap='viridis', aspect='auto')
    
    # Set ticks and labels
    ax2.set_yticks(range(len(df)))
    ax2.set_yticklabels([name[:20] + '...' if len(name) > 20 else name for name in df['Parameter']], fontsize=10)
    ax2.set_xticks([])
    ax2.set_title(f'Fisher Information Heatmap{title_suffix}', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label(f'Fisher Information ({metric.capitalize()})', fontsize=12)
    
    # Add text annotations on heatmap
    for i in range(len(df)):
        text = f'{importance_values[i, 0]:.1e}'
        ax2.text(0, i, text, ha='center', va='center', fontweight='bold', 
                color='white' if importance_values[i, 0] > importance_values.mean() else 'black')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Fisher Information heatmap saved to {save_path}")
    
    plt.show()
    return fig


def plot_fisher_information_detailed(fisher_dict, model=None, save_path=None, figsize=(15, 10)):
    """
    Create a comprehensive detailed visualization of Fisher Information.
    
    Args:
        fisher_dict: Dictionary mapping parameter names to their Fisher Information tensors
        model: PyTorch model (optional)
        save_path: Path to save the plot (optional)
        figsize: Figure size tuple
        
    Returns:
        fig: matplotlib figure object
    """
    
    if len(fisher_dict) == 0:
        print("Error: fisher_dict is empty!")
        return None
    
    # Prepare data
    param_stats = []
    for param_name, fisher_tensor in fisher_dict.items():
        fisher_cpu = fisher_tensor.cpu() if fisher_tensor.is_cuda else fisher_tensor
        
        param_stats.append({
            'Parameter': param_name,
            'Mean': fisher_cpu.mean().item(),
            'Max': fisher_cpu.max().item(),
            'Min': fisher_cpu.min().item(),
            'Std': fisher_cpu.std().item(),
            'Size': fisher_tensor.numel(),
            'Shape': str(fisher_tensor.shape)
        })
    
    df = pd.DataFrame(param_stats)
    df = df.sort_values('Mean', ascending=False)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Mean importance
    axes[0, 0].barh(range(len(df)), df['Mean'], color='steelblue')
    axes[0, 0].set_yticks(range(len(df)))
    axes[0, 0].set_yticklabels([name[:15] + '...' if len(name) > 15 else name for name in df['Parameter']], fontsize=9)
    axes[0, 0].set_xlabel('Mean Fisher Information')
    axes[0, 0].set_title('Mean Parameter Importance', fontweight='bold')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Max importance
    axes[0, 1].barh(range(len(df)), df['Max'], color='coral')
    axes[0, 1].set_yticks(range(len(df)))
    axes[0, 1].set_yticklabels([name[:15] + '...' if len(name) > 15 else name for name in df['Parameter']], fontsize=9)
    axes[0, 1].set_xlabel('Max Fisher Information')
    axes[0, 1].set_title('Maximum Parameter Importance', fontweight='bold')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot - Mean vs Std
    scatter = axes[1, 0].scatter(df['Mean'], df['Std'], c=df['Size'], cmap='viridis', alpha=0.7, s=100)
    axes[1, 0].set_xlabel('Mean Fisher Information')
    axes[1, 0].set_ylabel('Std Fisher Information')
    axes[1, 0].set_title('Mean vs Standard Deviation', fontweight='bold')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add colorbar for parameter size
    cbar = plt.colorbar(scatter, ax=axes[1, 0])
    cbar.set_label('Parameter Count')
    
    # Plot 4: Distribution of importance values
    all_importances = []
    for fisher_tensor in fisher_dict.values():
        fisher_cpu = fisher_tensor.cpu() if fisher_tensor.is_cuda else fisher_tensor
        all_importances.extend(fisher_cpu.flatten().tolist())
    
    axes[1, 1].hist(all_importances, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].set_xlabel('Fisher Information Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Fisher Information Values', fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Detailed Fisher Information plot saved to {save_path}")
    
    plt.show()
    return fig


# Test function to verify dataset compatibility
def test_dataset_access(dataset, device='cpu'):
    """
    Test function to verify we can properly access data from a dataset.
    """
    print(f"Testing dataset access...")
    print(f"Dataset type: {type(dataset)}")
    print(f"Dataset length: {len(dataset)}")
    
    try:
        # Try to create a DataLoader
        test_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        
        # Try to get one batch
        for batch_idx, batch in enumerate(test_loader):
            print(f"Successfully loaded batch {batch_idx}")
            print(f"Batch type: {type(batch)}")
            
            if isinstance(batch, (list, tuple)):
                print(f"Batch length: {len(batch)}")
                if len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                    print(f"Inputs shape: {inputs.shape}")
                    print(f"Targets shape: {targets.shape}")
                    print(f"Inputs device: {inputs.device}")
                    print(f"Targets device: {targets.device}")
                    print("✅ Dataset access test passed!")
                    return True
            break
    except Exception as e:
        print(f"❌ Dataset access test failed: {e}")
        return False
    
    return True


# Example usage:
if __name__ == "__main__":
    # Example with Avalanche
    from avalanche.benchmarks.classic import SplitMNIST
    from avalanche.models import SimpleMLP
    
    # Create benchmark and model
    benchmark = SplitMNIST(n_experiences=5, return_task_id=False, seed=42)
    model = SimpleMLP(num_classes=benchmark.n_classes)
    
    # Get first experience
    first_experience = next(iter(benchmark.train_stream))
    
    # Test dataset access
    test_dataset_access(first_experience.dataset)
    
    # Compute Fisher Information using dataset
    fisher_dict = compute_empirical_fisher_information(
        model=model, 
        dataset=first_experience.dataset, 
        device='cpu',
        num_samples=100  # Limit to 100 samples for testing
    )
    
    # Print Fisher Information statistics
    for name, fisher_values in fisher_dict.items():
        print(f"{name}: mean={fisher_values.mean().item():.2e}, max={fisher_values.max().item():.2e}")
    
    # Plot Fisher Information heatmap
    plot_fisher_information_heatmap(fisher_dict, model=model, save_path='fisher_heatmap.png', top_n_params=10)
    
    # Plot detailed Fisher Information analysis
    plot_fisher_information_detailed(fisher_dict, model=model, save_path='fisher_detailed.png')