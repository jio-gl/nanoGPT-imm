import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import GPT as StandardGPT
from model_imm import GPT as IMMGPT, GPTConfig

# Load config from training
config = {
    'out_dir': 'out-shakespeare-char',
    'batch_size': 32,
    'block_size': 128,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'vocab_size': 65,
    'n_layer': 4,
    'n_head': 4,
    'n_embd': 256,
    'dropout': 0.2,
    'eval_iters': 4000,  # Number of inference steps
    'dataset': 'shakespeare_char',
}

def get_fixed_validation_data():
    """Get a fixed set of validation data that will be reused for all inference steps"""
    data_dir = os.path.join('data', config['dataset'])
    data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # Use a fixed seed to get the same validation batches every time
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate multiple batches for the inference iterations
    batches = []
    for _ in range(config['eval_iters']):
        ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],))
        x = torch.stack([torch.from_numpy((data[i:i+config['block_size']]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+config['block_size']]).astype(np.int64)) for i in ix])
        x, y = x.to(config['device']), y.to(config['device'])
        batches.append((x, y))
    
    return batches

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_memory_module(model):
    """Unfreeze only the memory module parameters for IMM model"""
    for name, param in model.named_parameters():
        if 'memory' in name:
            param.requires_grad = True
            print(f"Unfrozen memory parameter: {name}")

def moving_average(data, window_size=50):
    """Calculate moving average with specified window size"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def run_inference(model, validation_batches, is_imm=False):
    model.eval()
    losses = []
    
    if is_imm:
        # For IMM, enable training mode only for memory module
        model.train()
        # But freeze everything except memory
        freeze_model(model)
        unfreeze_memory_module(model)
        
        # Setup optimizer for memory module only
        memory_params = [p for name, p in model.named_parameters() if 'memory' in name and p.requires_grad]
        if memory_params:
            optimizer = torch.optim.Adam(memory_params, lr=1e-4)
            print(f"Memory optimizer created with {len(memory_params)} parameters")
        else:
            print("Warning: No memory parameters found for optimization")
            optimizer = None
    
    for i, (x, y) in enumerate(validation_batches):
        if is_imm and optimizer is not None:
            # IMM: Update memory module
            optimizer.zero_grad()
            logits, loss = model(x, y, update_memory=True)
            loss.backward()
            optimizer.step()
        else:
            # Standard GPT: No updates
            with torch.no_grad():
                logits, loss = model(x, y)
        
        losses.append(loss.item())
        
        # Print progress every 100 iterations
        if (i + 1) % 100 == 0:
            print(f"  Iteration {i + 1}/{config['eval_iters']}, Loss: {loss.item():.4f}")
    
    return losses

def plot_inference_loss(standard_losses, imm_losses):
    window_size = 50
    
    # Calculate moving averages
    standard_ma = moving_average(standard_losses, window_size)
    imm_ma = moving_average(imm_losses, window_size)
    
    # Create x-axis for moving averages - ensure same length as moving average arrays
    ma_x = np.arange(len(standard_ma))
    
    plt.figure(figsize=(12, 8))
    
    # Plot raw data with transparency
    plt.subplot(2, 1, 1)
    plt.plot(standard_losses, alpha=0.3, color='blue', label='Standard GPT (raw)')
    plt.plot(imm_losses, alpha=0.3, color='red', label='IMM GPT (raw)')
    plt.plot(ma_x + window_size//2, standard_ma, color='blue', linewidth=2, label='Standard GPT (MA-50)')
    plt.plot(ma_x + window_size//2, imm_ma, color='red', linewidth=2, label='IMM GPT (MA-50)')
    plt.xlabel('Inference Iteration')
    plt.ylabel('Loss')
    plt.title('Inference Loss: Raw Data + Moving Average (50-step window)\nSame validation data - Standard GPT (frozen) vs IMM GPT (memory adapts)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot only moving averages for clarity
    plt.subplot(2, 1, 2)
    plt.plot(ma_x + window_size//2, standard_ma, color='blue', linewidth=2, label='Standard GPT (frozen)')
    plt.plot(ma_x + window_size//2, imm_ma, color='red', linewidth=2, label='IMM GPT (memory adapts)')
    plt.xlabel('Inference Iteration')
    plt.ylabel('Loss (Moving Average)')
    plt.title('Inference Loss: Moving Average Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['out_dir'], 'inference_adaptation_comparison.png'), dpi=300)
    plt.close()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Standard GPT - Initial loss: {np.mean(standard_losses[:10]):.4f}, Final loss: {np.mean(standard_losses[-10:]):.4f}")
    
    standard_initial = np.mean(standard_losses[:10])
    standard_final = np.mean(standard_losses[-10:])
    standard_change = standard_final - standard_initial
    standard_change_pct = (standard_change / standard_initial) * 100
    
    print(f"Standard GPT - Loss change: {standard_change:.4f} ({standard_change_pct:+.2f}%)")
    
    print(f"IMM GPT - Initial loss: {np.mean(imm_losses[:10]):.4f}, Final loss: {np.mean(imm_losses[-10:]):.4f}")
    
    imm_initial = np.mean(imm_losses[:10])
    imm_final = np.mean(imm_losses[-10:])
    imm_improvement = imm_initial - imm_final
    imm_improvement_pct = (imm_improvement / imm_initial) * 100
    
    print(f"IMM GPT - Loss improvement: {imm_improvement:.4f} ({imm_improvement_pct:+.2f}%)")
    
    if imm_improvement > 0:
        print(f"✓ IMM memory module successfully adapted and reduced loss!")
    else:
        print(f"✗ IMM memory module did not show improvement (may need more iterations or different settings)")

def main():
    # Get fixed validation data
    print('Preparing fixed validation dataset...')
    validation_batches = get_fixed_validation_data()
    print(f'Created {len(validation_batches)} validation batches')
    
    # Load Standard GPT
    print('Loading Standard GPT...')
    standard_gptconf = GPTConfig(
        n_layer=config['n_layer'], n_head=config['n_head'], n_embd=config['n_embd'],
        block_size=config['block_size'], bias=True, vocab_size=config['vocab_size'], dropout=config['dropout']
    )
    standard_model = StandardGPT(standard_gptconf)
    ckpt = torch.load(os.path.join(config['out_dir'], 'standard_model.pt'), map_location=config['device'], weights_only=False)
    standard_model.load_state_dict(ckpt['model'])
    standard_model.to(config['device'])
    freeze_model(standard_model)

    # Load IMM GPT
    print('Loading IMM GPT...')
    imm_gptconf = GPTConfig(
        n_layer=config['n_layer'], n_head=config['n_head'], n_embd=config['n_embd'],
        block_size=config['block_size'], bias=True, vocab_size=config['vocab_size'], dropout=config['dropout'], memory_slots=8
    )
    imm_model = IMMGPT(imm_gptconf)
    ckpt = torch.load(os.path.join(config['out_dir'], 'imm_model.pt'), map_location=config['device'], weights_only=False)
    imm_model.load_state_dict(ckpt['model'])
    imm_model.to(config['device'])

    # Inference
    print('Running inference on Standard GPT (all weights frozen)...')
    standard_losses = run_inference(standard_model, validation_batches, is_imm=False)
    print('Running inference on IMM GPT (only memory module updates)...')
    imm_losses = run_inference(imm_model, validation_batches, is_imm=True)

    # Plot
    plot_inference_loss(standard_losses, imm_losses)
    print('Inference adaptation comparison complete. Chart saved as inference_adaptation_comparison.png')

if __name__ == '__main__':
    main() 