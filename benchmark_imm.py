"""
Benchmark script to measure Implicit Memory Module (IMM) effectiveness
on Shakespeare + modern text dataset.
"""

import os
import time
import math
import torch
import torch.nn as nn
import numpy as np
from model_imm import GPT, GPTConfig
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import wandb

class ShakespeareModernDataset(Dataset):
    def __init__(self, block_size, split='train'):
        self.block_size = block_size
        
        # Load Shakespeare data
        shakespeare_path = os.path.join('data', 'shakespeare_char', f'{split}.bin')
        shakespeare_data = np.memmap(shakespeare_path, dtype=np.uint16, mode='r')
        
        # Load modern text (you'll need to prepare this)
        modern_path = os.path.join('data', 'modern_text', f'{split}.bin')
        modern_data = np.memmap(modern_path, dtype=np.uint16, mode='r')
        
        # Combine datasets
        self.data = np.concatenate([shakespeare_data, modern_data])
        
    def __len__(self):
        return len(self.data) - self.block_size
        
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1:idx + 1 + self.block_size].astype(np.int64))
        return x, y

class MemoryBenchmark:
    def __init__(self, model, device, block_size=256):
        self.model = model
        self.device = device
        self.block_size = block_size
        
    def measure_memory_effectiveness(self, test_sequences, context_lengths=[32, 64, 128]):
        """Measure how well the model remembers context of different lengths"""
        results = []
        
        for context_len in context_lengths:
            print(f"\nTesting context length: {context_len}")
            accuracies = []
            for seq_idx, seq in enumerate(test_sequences):
                if len(seq) < context_len + 10:
                    print(f"Warning: Sequence {seq_idx} too short for context length {context_len}")
                    continue
                    
                # Get context and target
                context = seq[:context_len]
                target = seq[context_len:context_len + 10]  # Predict next 10 tokens
                
                # Generate with context
                with torch.no_grad():
                    x = torch.tensor(context, dtype=torch.long, device=self.device).unsqueeze(0)
                    y = self.model.generate(x, max_new_tokens=10, temperature=1.0)
                    pred = y[0, -10:].cpu().numpy()
                
                # Calculate accuracy
                target = target.astype(np.int64)  # Ensure same dtype
                matches = np.sum(pred == target)
                acc = matches / len(target)
                accuracies.append(acc)
                
                print(f"  Sequence {seq_idx + 1}:")
                print(f"    Context length: {len(context)}")
                print(f"    Predicted: {pred}")
                print(f"    Target: {target}")
                print(f"    Accuracy: {acc:.4f}")
            
            if accuracies:
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                print(f"Context length {context_len} - Mean accuracy: {mean_acc:.4f}, Std: {std_acc:.4f}")
                
                results.append({
                    'context_length': context_len,
                    'accuracy': mean_acc,
                    'std': std_acc
                })
            else:
                print(f"No valid sequences for context length {context_len}")
            
        return results
    
    def measure_modern_vs_shakespeare(self, modern_seqs, shakespeare_seqs):
        """Compare performance on modern vs Shakespeare text"""
        print("\nEvaluating modern text...")
        modern_acc = self.measure_memory_effectiveness(modern_seqs)
        
        print("\nEvaluating Shakespeare text...")
        shakespeare_acc = self.measure_memory_effectiveness(shakespeare_seqs)
        
        return {
            'modern': modern_acc,
            'shakespeare': shakespeare_acc
        }
    
    def measure_scaling(self, seq_lengths=[256, 512, 1024]):
        """Measure how performance scales with sequence length"""
        results = []
        
        for seq_len in seq_lengths:
            if seq_len > self.block_size:
                print(f"Warning: sequence length {seq_len} exceeds model's block size {self.block_size}")
                continue
                
            print(f"\nTesting sequence length: {seq_len}")
            # Generate test sequences
            test_seqs = [np.random.randint(0, 50257, seq_len) for _ in range(10)]
            
            # Measure memory effectiveness
            acc = self.measure_memory_effectiveness(test_seqs, [seq_len//2])[0]
            results.append({
                'sequence_length': seq_len,
                'accuracy': acc['accuracy'],
                'std': acc['std']
            })
            
        return results

def prepare_modern_text():
    """Prepare modern text dataset"""
    # This is a placeholder - you'll need to implement this
    # to create your modern text dataset
    pass

def train_model(model, train_dataset, val_dataset, device, config):
    """Train the model before benchmarking"""
    print("Training model...")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            if batch_idx >= config['max_train_steps']:
                break
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} - Train loss: {total_loss/(batch_idx+1):.4f}, Val loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    return model

def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'block_size': 256,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 3,
        'max_train_steps': 100  # Limit training steps for quick testing
    }
    
    print(f"Using device: {device}")
    print("Loading datasets...")
    
    # Create datasets
    train_dataset = ShakespeareModernDataset(config['block_size'], 'train')
    val_dataset = ShakespeareModernDataset(config['block_size'], 'val')
    
    print("Initializing model...")
    # Initialize model
    model_args = dict(
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=config['block_size'],
        bias=True,
        vocab_size=50257,
        dropout=0.1
    )
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)
    
    # Train model
    model = train_model(model, train_dataset, val_dataset, device, config)
    
    # Initialize benchmark
    benchmark = MemoryBenchmark(model, device, config['block_size'])
    
    # Run benchmarks
    print("Running memory effectiveness benchmark...")
    context_results = benchmark.measure_memory_effectiveness(
        [train_dataset[i][0].numpy() for i in range(10)]
    )
    
    print("Running modern vs Shakespeare benchmark...")
    modern_vs_shakespeare = benchmark.measure_modern_vs_shakespeare(
        [train_dataset[i][0].numpy() for i in range(5)],
        [val_dataset[i][0].numpy() for i in range(5)]
    )
    
    print("Running scaling benchmark...")
    scaling_results = benchmark.measure_scaling()
    
    # Log results
    # wandb.log({
    #     'context_results': context_results,
    #     'modern_vs_shakespeare': modern_vs_shakespeare,
    #     'scaling_results': scaling_results
    # })
    
    print("\nResults:")
    print("Context Results:", context_results)
    print("\nModern vs Shakespeare:", modern_vs_shakespeare)
    print("\nScaling Results:", scaling_results)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot context length vs accuracy
    plt.subplot(131)
    context_lengths = [r['context_length'] for r in context_results]
    accuracies = [r['accuracy'] for r in context_results]
    plt.plot(context_lengths, accuracies)
    plt.title('Context Length vs Accuracy')
    plt.xlabel('Context Length')
    plt.ylabel('Accuracy')
    
    # Plot modern vs Shakespeare
    plt.subplot(132)
    modern_acc = [r['accuracy'] for r in modern_vs_shakespeare['modern']]
    shakespeare_acc = [r['accuracy'] for r in modern_vs_shakespeare['shakespeare']]
    plt.plot(context_lengths, modern_acc, label='Modern')
    plt.plot(context_lengths, shakespeare_acc, label='Shakespeare')
    plt.title('Modern vs Shakespeare Performance')
    plt.xlabel('Context Length')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot scaling results
    plt.subplot(133)
    seq_lengths = [r['sequence_length'] for r in scaling_results]
    scaling_acc = [r['accuracy'] for r in scaling_results]
    plt.plot(seq_lengths, scaling_acc)
    plt.title('Scaling Performance')
    plt.xlabel('Sequence Length')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("\nPlots saved to benchmark_results.png")
    # wandb.log({'benchmark_plot': wandb.Image('benchmark_results.png')})

if __name__ == '__main__':
    main() 