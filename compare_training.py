import os
import time
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import GPT as StandardGPT  # Import standard GPT
from model_imm import GPT as IMMGPT, GPTConfig  # Import IMM GPT

# Shakespeare training configuration
config = {
    'out_dir': 'out-shakespeare-char',
    'eval_interval': 100,
    'eval_iters': 20,
    'log_interval': 10,
    'dataset': 'shakespeare_char',
    'gradient_accumulation_steps': 1,
    'batch_size': 32,
    'block_size': 128,
    'n_layer': 4,
    'n_head': 4,
    'n_embd': 256,
    'dropout': 0.2,
    'learning_rate': 1e-3,
    'max_iters': 2000,
    'lr_decay_iters': 2000,
    'min_lr': 1e-4,
    'beta2': 0.99,
    'warmup_iters': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'compile': False
}

def get_batch(split):
    data_dir = os.path.join('data', config['dataset'])
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],))
    x = torch.stack([torch.from_numpy((data[i:i+config['block_size']]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config['block_size']]).astype(np.int64)) for i in ix])
    x, y = x.to(config['device']), y.to(config['device'])
    return x, y

def train_model(model, is_imm=False):
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, config['beta2']),
        eps=1e-8
    )
    
    # Initialize tracking variables
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Training loop
    for iter_num in tqdm(range(config['max_iters']), desc=f"Training {'IMM' if is_imm else 'Standard'} model"):
        # Get batch
        x, y = get_batch('train')
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Track training loss
        train_losses.append(loss.item())
        
        # Evaluation
        if iter_num % config['eval_interval'] == 0:
            model.eval()
            with torch.no_grad():
                val_losses_batch = []
                for _ in range(config['eval_iters']):
                    x, y = get_batch('val')
                    _, loss = model(x, y)
                    val_losses_batch.append(loss.item())
                val_loss = np.mean(val_losses_batch)
                val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save checkpoint
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    torch.save(checkpoint, os.path.join(config['out_dir'], f"{'imm' if is_imm else 'standard'}_model.pt"))
            model.train()
    
    return train_losses, val_losses

def plot_comparison(standard_train_losses, standard_val_losses, imm_train_losses, imm_val_losses):
    plt.figure(figsize=(12, 6))
    
    # Plot training losses
    plt.subplot(1, 2, 1)
    plt.plot(standard_train_losses, label='Standard GPT')
    plt.plot(imm_train_losses, label='IMM GPT')
    plt.title('Training Loss Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot validation losses
    plt.subplot(1, 2, 2)
    plt.plot(standard_val_losses, label='Standard GPT')
    plt.plot(imm_val_losses, label='IMM GPT')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['out_dir'], 'training_comparison.png'))
    plt.close()

def main():
    # Create output directory
    os.makedirs(config['out_dir'], exist_ok=True)
    
    # Initialize models
    standard_model_args = dict(
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_embd=config['n_embd'],
        block_size=config['block_size'],
        bias=True,
        vocab_size=65,  # Shakespeare character-level vocabulary size
        dropout=config['dropout']
    )
    
    imm_model_args = dict(
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_embd=config['n_embd'],
        block_size=config['block_size'],
        bias=True,
        vocab_size=65,  # Shakespeare character-level vocabulary size
        dropout=config['dropout'],
        memory_slots=8  # Add memory slots for IMM model
    )
    
    # Train standard GPT
    print("Training standard GPT...")
    standard_gptconf = GPTConfig(**standard_model_args)
    standard_model = StandardGPT(standard_gptconf)  # Use StandardGPT
    standard_model.to(config['device'])
    standard_train_losses, standard_val_losses = train_model(standard_model, is_imm=False)
    
    # Train IMM GPT
    print("Training IMM GPT...")
    imm_gptconf = GPTConfig(**imm_model_args)
    imm_model = IMMGPT(imm_gptconf)  # Use IMMGPT
    imm_model.to(config['device'])
    imm_train_losses, imm_val_losses = train_model(imm_model, is_imm=True)
    
    # Plot comparison
    plot_comparison(standard_train_losses, standard_val_losses, imm_train_losses, imm_val_losses)
    
    # Save results
    results = {
        'standard_train_losses': standard_train_losses,
        'standard_val_losses': standard_val_losses,
        'imm_train_losses': imm_train_losses,
        'imm_val_losses': imm_val_losses,
        'config': config
    }
    torch.save(results, os.path.join(config['out_dir'], 'training_results.pt'))
    
    print("Training comparison complete. Results saved in", config['out_dir'])

if __name__ == '__main__':
    main() 