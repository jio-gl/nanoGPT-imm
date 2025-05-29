import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import GPT, GPTConfig
from model_imm import GPT as GPTIMM
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import numpy as np
from torch.nn import functional as F
import json

def load_pretrained_model(model_type, device):
    """Load a pre-trained GPT-2 model"""
    if model_type == 'standard':
        model = GPT.from_pretrained('gpt2')
    else:
        model = GPTIMM.from_pretrained('gpt2')
    model = model.to(device)
    return model

def load_dataset(data_path):
    with open(data_path, 'r') as f:
        text = f.read()
    
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    
    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, encode, decode, vocab_size

def get_batch(split, train_data, val_data, batch_size=32, block_size=8, device='cpu'):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def traditional_finetune(model, train_data, val_data, device, num_steps=1000, batch_size=32, block_size=8, learning_rate=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []
    
    model.train()
    pbar = tqdm(range(num_steps), desc="Traditional fine-tuning")
    
    for step in pbar:
        # sample a batch of data
        xb, yb = get_batch('train', train_data, val_data, batch_size, block_size, device)
        
        # evaluate the loss
        logits, loss = model(xb, yb)
        losses.append(loss.item())
        
        # backprop and update the parameters
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return losses

def imm_finetune(model, train_data, val_data, device, num_steps=1000, batch_size=32, block_size=8):
    losses = []
    
    model.eval()  # IMM operates in eval mode
    pbar = tqdm(range(num_steps), desc="IMM pseudo fine-tuning")
    
    for step in pbar:
        # sample a batch of data
        xb, yb = get_batch('train', train_data, val_data, batch_size, block_size, device)
        
        # evaluate the loss
        logits, loss = model(xb, yb)
        losses.append(loss.item())
        
        # Update progress bar
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return losses

def evaluate_model(model, get_batch, device, num_batches=10):
    """Evaluate model on validation data"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch('val')
            logits, loss = model(x, y)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def plot_results(standard_losses, imm_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(standard_losses, label='Traditional Fine-tuning')
    plt.plot(imm_losses, label='IMM Pseudo Fine-tuning')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Comparison of Fine-tuning Methods')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def save_results(results, save_path):
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)

def compare_finetuning(data_path):
    # Set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Load and preprocess the dataset
    train_data, val_data, encode, decode, vocab_size = load_dataset(data_path)
    
    # Load pre-trained models
    print("\nLoading pre-trained models...")
    standard_model = GPT.from_pretrained('gpt2')
    imm_model = GPTIMM.from_pretrained('gpt2')
    
    standard_model = standard_model.to(device)
    imm_model = imm_model.to(device)
    
    # Perform traditional fine-tuning
    print("\nPerforming traditional fine-tuning...")
    standard_losses = traditional_finetune(standard_model, train_data, val_data, device)
    
    # Perform IMM pseudo fine-tuning
    print("\nPerforming IMM pseudo fine-tuning...")
    imm_losses = imm_finetune(imm_model, train_data, val_data, device)
    
    # Save results
    results = {
        'standard_losses': standard_losses,
        'imm_losses': imm_losses
    }
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save the results
    save_results(results, 'results/finetuning_comparison.json')
    plot_results(standard_losses, imm_losses, 'results/finetuning_comparison.png')
    
    print("\nResults have been saved to the 'results' directory.")

if __name__ == '__main__':
    data_path = 'data/shakespeare_char/input.txt'
    compare_finetuning(data_path) 