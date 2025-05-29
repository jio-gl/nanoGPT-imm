import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import GPT, GPTConfig
from model_imm import GPT as GPTIMM
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        data_size = len(data)
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

def get_data_loaders(data_path, block_size, batch_size, max_samples=5000, train_ratio=0.8, val_ratio=0.1):
    # Read the data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    # Limit the data size
    data = data[:max_samples * (block_size + 1)]
    
    # Create dataset
    dataset = CharDataset(data, block_size)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, dataset.vocab_size

def create_model(model_type, config):
    """Create either a standard GPT or IMM model"""
    if model_type == 'standard':
        return GPT(config)
    else:
        return GPTIMM(config)

def train_model(model, train_data, val_data, device, epochs=3):
    """Train the model and return training history"""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_data, desc=f'Epoch {epoch+1}'):
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_data)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_data:
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_data)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
    
    return train_losses, val_losses

def evaluate_inference_improvement(model, test_loader, device, num_steps=10):
    """Evaluate how model performance improves during inference"""
    model.eval()
    losses = []
    perplexities = []
    memory_usage = []  # Track memory usage if available
    
    # Convert test loader to list for multiple passes
    test_batches = list(test_loader)
    num_batches = len(test_batches)
    
    with torch.no_grad():
        # Reset memory only once at the start for IMM model
        if hasattr(model, 'imm'):
            model.imm.reset_memory(test_batches[0][0].size(0), device)
            
        for step in range(num_steps):
            total_loss = 0
            total_tokens = 0
            
            # Use different starting points in the test set for each step
            start_idx = step % num_batches
            for i in range(num_batches):
                batch_idx = (start_idx + i) % num_batches
                x, y = test_batches[batch_idx]
                x, y = x.to(device), y.to(device)
                
                # For IMM model, update memory during inference
                if hasattr(model, 'imm'):
                    # First pass without memory update to get baseline performance
                    logits, loss = model(x, y, update_memory=False)
                    # Second pass with memory update to improve future predictions
                    _, _ = model(x, targets=y, update_memory=True, logits=logits)
                else:
                    logits, loss = model(x, y)
                
                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()
            
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss)
            losses.append(avg_loss)
            perplexities.append(perplexity)
            
            # Try to get memory usage if available
            if hasattr(model, 'imm') and hasattr(model.imm, 'memory'):
                memory_size = model.imm.memory.element_size() * model.imm.memory.nelement()
                memory_usage.append(memory_size / (1024 * 1024))  # Convert to MB
            
            print(f'Inference Step {step+1}: Loss = {avg_loss:.4f}, Perplexity = {perplexity:.2f}')
            if memory_usage:
                print(f'Memory Usage: {memory_usage[-1]:.2f} MB')
    
    return losses, perplexities, memory_usage

def save_model(model, path):
    """Save model and its configuration"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config
    }, path)
    print(f"Model saved to {path}")

def load_model(model_type, path, device):
    """Load model from saved checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['config']
    
    # Create model with saved config
    if model_type == 'standard':
        model = GPT(config)
    else:
        model = GPTIMM(config)
    
    # For IMM model, we need to handle the memory buffers
    if model_type == 'imm':
        # Remove memory buffers from state dict as they will be reinitialized
        state_dict = checkpoint['model_state_dict']
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('imm.memory')}
        model.load_state_dict(state_dict, strict=False)
        # Initialize memory with correct batch size
        if hasattr(model, 'imm'):
            model.imm.reset_memory(1, device)  # Initialize with batch size 1, will be reset during evaluation
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    print(f"Model loaded from {path}")
    return model

def train_and_save_models(config, train_loader, val_loader, device):
    """Train both models and save them"""
    print("Training new models...")
    # Create models
    standard_model = GPT(config)
    imm_model = GPTIMM(config)
    
    print(f"Standard GPT parameters: {standard_model.get_num_params()/1e6:.2f}M")
    print(f"IMM GPT parameters: {imm_model.get_num_params()/1e6:.2f}M")
    
    # Train models
    print("\nTraining Standard GPT...")
    standard_train_losses, standard_val_losses = train_model(
        standard_model, train_loader, val_loader, device, epochs=1
    )
    
    print("\nTraining IMM GPT...")
    imm_train_losses, imm_val_losses = train_model(
        imm_model, train_loader, val_loader, device, epochs=1
    )
    
    # Save models
    os.makedirs('out', exist_ok=True)
    save_model(standard_model, 'out/standard_model.pt')
    save_model(imm_model, 'out/imm_model.pt')
    
    return standard_model, imm_model, standard_train_losses, standard_val_losses, imm_train_losses, imm_val_losses

def create_fact_pool(vocab_size):
    """Create a pool of different facts to learn"""
    # Use tokens from the first quarter of the vocabulary to avoid conflicts
    # and ensure we stay within vocabulary bounds
    base_idx = vocab_size // 4
    max_token_id = vocab_size - 1
    
    def safe_token(offset):
        """Ensure token index is within vocabulary bounds"""
        token = base_idx + offset
        return min(token, max_token_id)
    
    # Define special tokens for fact structure
    START_FACT = safe_token(1)
    CATEGORY = safe_token(2)
    RELATION = safe_token(3)
    VALUE = safe_token(4)
    END_FACT = safe_token(5)
    
    # Create facts with clear categories and relationships
    facts_pool = {
        'science': [
            # Format: [START_FACT, CATEGORY, category_id, RELATION, relation_id, VALUE, value_id, END_FACT]
            [START_FACT, CATEGORY, safe_token(10), RELATION, safe_token(11), VALUE, safe_token(12), END_FACT],  # Speed of light
            [START_FACT, CATEGORY, safe_token(10), RELATION, safe_token(13), VALUE, safe_token(14), END_FACT],  # Gravitational constant
            [START_FACT, CATEGORY, safe_token(10), RELATION, safe_token(15), VALUE, safe_token(16), END_FACT],  # Planck constant
        ],
        'math': [
            [START_FACT, CATEGORY, safe_token(20), RELATION, safe_token(21), VALUE, safe_token(22), END_FACT],  # Pi
            [START_FACT, CATEGORY, safe_token(20), RELATION, safe_token(23), VALUE, safe_token(24), END_FACT],  # Euler's number
            [START_FACT, CATEGORY, safe_token(20), RELATION, safe_token(25), VALUE, safe_token(26), END_FACT],  # Golden ratio
        ],
        'history': [
            [START_FACT, CATEGORY, safe_token(30), RELATION, safe_token(31), VALUE, safe_token(32), END_FACT],  # First computer
            [START_FACT, CATEGORY, safe_token(30), RELATION, safe_token(33), VALUE, safe_token(34), END_FACT],  # First internet
            [START_FACT, CATEGORY, safe_token(30), RELATION, safe_token(35), VALUE, safe_token(36), END_FACT],  # First AI
        ]
    }
    
    # Validate all tokens are within vocabulary bounds
    for category_facts in facts_pool.values():
        for fact in category_facts:
            for token in fact:
                if token >= vocab_size:
                    raise ValueError(f"Token {token} is outside vocabulary range [0, {vocab_size-1}]")
    
    return facts_pool

def create_pattern_from_facts(facts, pattern_length=128):
    """Create a pattern from a list of facts"""
    try:
        # Flatten facts into a single pattern
        pattern = torch.tensor([token for fact in facts for token in fact], dtype=torch.long)
        
        # Pad pattern if needed
        if len(pattern) < pattern_length:
            pattern = torch.cat([pattern, torch.zeros(pattern_length - len(pattern), dtype=torch.long)])
        elif len(pattern) > pattern_length:
            pattern = pattern[:pattern_length]
        
        return pattern
    except Exception as e:
        print(f"Error creating pattern: {e}")
        return None

def evaluate_inference_learning(model, test_loader, device, num_steps=50):
    """Evaluate if model can learn from new data during inference"""
    model.eval()
    results = {
        'category_losses': {},  # Track losses by category
        'general_losses': [],   # Track general performance
        'memory_usage': [],     # Track memory usage for IMM
        'learning_curves': {}   # Track learning progress per category
    }
    
    try:
        # Get vocabulary size from test data
        test_batches = list(test_loader)
        x, y = test_batches[0]
        vocab_size = x.max().item() + 1
        print(f"Vocabulary size: {vocab_size}")
        
        # Create pool of facts
        facts_pool = create_fact_pool(vocab_size)
        learned_facts = {category: [] for category in facts_pool.keys()}
        
        # Reset memory for IMM model
        if hasattr(model, 'imm'):
            model.imm.reset_memory(x.size(0), device)
            print("Reset IMM memory")
        
        with torch.no_grad():
            # Track best performance per category
            best_losses = {category: float('inf') for category in facts_pool.keys()}
            no_improvement = {category: 0 for category in facts_pool.keys()}
            
            for step in range(num_steps):
                print(f"\nStep {step + 1}/{num_steps}")
                
                # Introduce new facts from each category
                for category, facts in facts_pool.items():
                    if step < len(facts):
                        fact = facts[step]
                        print(f"\nProcessing {category} fact {step + 1}:")
                        print(f"Tokens: {fact}")
                        
                        learned_facts[category].append(fact)
                        print(f"Total {category} facts learned: {len(learned_facts[category])}")
                        
                        # Create pattern from current category's facts
                        pattern = create_pattern_from_facts(learned_facts[category])
                        if pattern is None:
                            continue
                        
                        print(f"Pattern shape before repeat: {pattern.shape}")
                        # Ensure pattern matches batch size
                        pattern = pattern.unsqueeze(0).repeat(x.size(0), 1)
                        print(f"Pattern shape after repeat: {pattern.shape}")
                        
                        x_pattern = pattern.to(device)
                        y_pattern = pattern.to(device)
                        
                        # Warm-up: expose model to current facts
                        print(f"Warm-up: Exposing model to {category} facts...")
                        for warm_up_step in range(3):
                            if hasattr(model, 'imm'):
                                _, warm_up_loss = model(x_pattern, targets=y_pattern, update_memory=True)
                            else:
                                _, warm_up_loss = model(x_pattern, y_pattern)
                            print(f"Warm-up step {warm_up_step + 1} loss: {warm_up_loss.item():.4f}")
                        
                        # Evaluate on current category
                        logits, loss = model(x_pattern, y_pattern)
                        category_loss = loss.item()
                        
                        # Track category-specific performance
                        if category not in results['category_losses']:
                            results['category_losses'][category] = []
                        results['category_losses'][category].append(category_loss)
                        
                        # Check if category performance improved
                        if category_loss < best_losses[category]:
                            best_losses[category] = category_loss
                            no_improvement[category] = 0
                            print(f"New best loss for {category}: {category_loss:.4f}")
                        else:
                            no_improvement[category] += 1
                            print(f"No improvement for {category} ({no_improvement[category]} steps)")
                        
                        # For IMM model, update memory with current facts
                        if hasattr(model, 'imm'):
                            _, _ = model(x_pattern, targets=y_pattern, update_memory=True, logits=logits)
                
                # Evaluate on regular test data
                test_batches = list(test_loader)
                total_loss = 0
                total_tokens = 0
                
                for x, y in test_batches:
                    x, y = x.to(device), y.to(device)
                    logits, loss = model(x, y)
                    total_loss += loss.item() * y.numel()
                    total_tokens += y.numel()
                
                avg_loss = total_loss / total_tokens
                perplexity = math.exp(avg_loss)
                results['general_losses'].append(avg_loss)
                
                # Track memory usage for IMM model
                if hasattr(model, 'imm') and hasattr(model.imm, 'memory'):
                    memory_size = model.imm.memory.element_size() * model.imm.memory.nelement()
                    results['memory_usage'].append(memory_size / (1024 * 1024))  # Convert to MB
                
                # Print progress
                print("\nCurrent Performance:")
                for category in facts_pool.keys():
                    if category in results['category_losses']:
                        print(f"{category.capitalize()} Loss: {results['category_losses'][category][-1]:.4f}")
                print(f"General Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
                if results['memory_usage']:
                    print(f"Memory Usage: {results['memory_usage'][-1]:.2f} MB")
                
                # Check if all categories have stopped improving
                if all(count >= 2 for count in no_improvement.values()):
                    print("\nStopping early: No improvement in any category for 2 steps")
                    break
        
        return results
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_models(standard_model, imm_model, test_loader, device, num_steps=50):
    """Evaluate both models and generate comparison plots"""
    print("\nEvaluating IMM GPT inference learning...")
    imm_results = evaluate_inference_learning(imm_model, test_loader, device, num_steps=num_steps)
    
    print("\nEvaluating Standard GPT inference learning...")
    standard_results = evaluate_inference_learning(standard_model, test_loader, device, num_steps=num_steps)
    
    if imm_results is None or standard_results is None:
        print("Error: Evaluation failed")
        return
    
    # Plot results
    plt.figure(figsize=(20, 15))
    
    # Plot category-specific learning curves
    for i, category in enumerate(imm_results['category_losses'].keys()):
        plt.subplot(3, 2, i+1)
        plt.plot(standard_results['category_losses'][category], label='Standard')
        plt.plot(imm_results['category_losses'][category], label='IMM')
        plt.title(f'{category.capitalize()} Learning Curve')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
    
    # Plot general performance
    plt.subplot(3, 2, 4)
    plt.plot(standard_results['general_losses'], label='Standard')
    plt.plot(imm_results['general_losses'], label='IMM')
    plt.title('General Performance')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot memory usage
    if imm_results['memory_usage']:
        plt.subplot(3, 2, 5)
        plt.plot(imm_results['memory_usage'], label='IMM Memory Usage')
        plt.title('IMM Memory Usage')
        plt.xlabel('Step')
        plt.ylabel('Memory (MB)')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('inference_learning_comparison.png')
    print("\nResults saved to inference_learning_comparison.png")
    
    # Print final comparison metrics
    print("\nFinal Comparison Metrics:")
    for category in imm_results['category_losses'].keys():
        imm_final = imm_results['category_losses'][category][-1]
        standard_final = standard_results['category_losses'][category][-1]
        improvement = (standard_final - imm_final) / standard_final * 100
        print(f"{category.capitalize()}:")
        print(f"  IMM GPT - Final Loss: {imm_final:.4f}")
        print(f"  Standard GPT - Final Loss: {standard_final:.4f}")
        print(f"  Improvement: {improvement:.2f}%")
    
    imm_final = imm_results['general_losses'][-1]
    standard_final = standard_results['general_losses'][-1]
    improvement = (standard_final - imm_final) / standard_final * 100
    print(f"\nGeneral Performance:")
    print(f"IMM GPT - Final Loss: {imm_final:.4f}")
    print(f"Standard GPT - Final Loss: {standard_final:.4f}")
    print(f"Improvement: {improvement:.2f}%")
    
    if imm_results['memory_usage']:
        print(f"Final Memory Usage: {imm_results['memory_usage'][-1]:.2f} MB")

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model paths
    standard_model_path = 'out/standard_model.pt'
    imm_model_path = 'out/imm_model.pt'
    
    # Load data with minimal dataset
    data_path = 'data/shakespeare_char/input.txt'
    block_size = 128
    batch_size = 64
    max_samples = 200
    train_loader, val_loader, test_loader, vocab_size = get_data_loaders(
        data_path, block_size, batch_size, max_samples=max_samples
    )
    
    config = GPTConfig(
        n_layer=4,
        n_head=4,
        n_embd=256,
        block_size=block_size,
        vocab_size=vocab_size,
        dropout=0.0,
        bias=True
    )
    
    # Check if models exist
    if not (os.path.exists(standard_model_path) and os.path.exists(imm_model_path)):
        standard_model, imm_model, *_ = train_and_save_models(config, train_loader, val_loader, device)
    else:
        print("Loading existing models...")
        standard_model = load_model('standard', standard_model_path, device)
        imm_model = load_model('imm', imm_model_path, device)
    
    # Evaluate models with 50 inference steps
    evaluate_models(standard_model, imm_model, test_loader, device, num_steps=50)

if __name__ == '__main__':
    main() 