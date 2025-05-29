import torch
import torch.nn as nn
from model_imm import GPT, GPTConfig
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F

def compute_model_sizes(model):
    """Compute the size of the memory module and the rest of the model in terms of number of parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    memory_params = sum(p.numel() for p in model.imm.parameters())
    rest_params = total_params - memory_params
    return total_params, memory_params, rest_params

def load_shakespeare_data(block_size, num_sequences):
    """Load real Shakespeare data from train.bin and val.bin, ensuring each sequence is exactly block_size."""
    train_path = os.path.join('data', 'shakespeare_char', 'train.bin')
    val_path = os.path.join('data', 'shakespeare_char', 'val.bin')
    train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
    sequences = []
    for data in [train_data, val_data]:
        for i in range(0, len(data) - block_size + 1, block_size):
            if len(sequences) >= num_sequences:
                break
            seq = torch.from_numpy(data[i:i + block_size].astype(np.int64)).unsqueeze(0)  # Shape: (1, block_size)
            if seq.shape[1] == block_size:
                sequences.append(seq)
    if len(sequences) < num_sequences:
        print(f"[Warning] Only {len(sequences)} sequences of length {block_size} could be loaded (requested {num_sequences}).")
    if sequences:
        print(f"[Info] First loaded sequence shape: {sequences[0].shape}")
    else:
        print(f"[Error] No sequences loaded for block_size={block_size}.")
    return sequences

class MemoryExperiments:
    def __init__(self, model):
        self.model = model
        self.memory_stats = {
            'slot_usage': [],
            'gate_values': [],
            'attention_scores': [],
            'memory_updates': []
        }
        
    def track_memory_updates(self, module, input, output):
        """Hook to track memory updates"""
        self.memory_stats['memory_updates'].append(output.detach())
        
    def track_gate_values(self, module, input, output):
        """Hook to track gate values"""
        self.memory_stats['gate_values'].append(output.detach())
        
    def track_attention_scores(self, module, input, output):
        """Hook to track attention scores"""
        self.memory_stats['attention_scores'].append(output.detach())

    def setup_tracking(self):
        """Setup hooks for tracking memory behavior"""
        self.model.imm.update_gate.register_forward_hook(self.track_gate_values)
        self.model.imm.attn_dropout.register_forward_hook(self.track_attention_scores)
        
    def test_memory_pattern_storage(self, input_sequences, num_sequences=10):
        """Test how well the memory stores and retrieves patterns"""
        self.model.eval()
        pattern_similarities = []
        for i in range(num_sequences):
            with torch.no_grad():
                logits, _, hidden_states = self.model(input_sequences[i], update_memory=True, return_hidden_states=True)
            memory = self.model.imm.memory
            memory_mask = self.model.imm.memory_mask
            pattern_sim = self._calculate_pattern_similarity(memory, hidden_states.mean(dim=1))
            pattern_similarities.append(pattern_sim)
            if (i+1) % max(10, num_sequences//10) == 0 or (i+1) == num_sequences:
                print(f"Pattern Storage: {i+1}/{num_sequences} ({(i+1)/num_sequences*100:.1f}%) done")
        return pattern_similarities

    def test_memory_loss_improvement(self, input_sequences, num_sequences=10):
        """Test the relative improvement in loss when memory is enabled."""
        self.model.eval()
        loss_improvements = []
        for i in range(num_sequences):
            with torch.no_grad():
                # Compute logits without memory update
                logits1, _ = self.model(input_sequences[i], update_memory=False)
                # Compute logits with memory update
                logits2, _ = self.model(input_sequences[i], update_memory=True)
                # For language modeling, compare logits[:, :-1, :] to input_sequences[:, 1:]
                print(f"[DEBUG] logits1 shape: {logits1.shape}")
                print(f"[DEBUG] logits2 shape: {logits2.shape}")
                print(f"[DEBUG] input_sequences[i] shape: {input_sequences[i].shape}")
                if logits1.shape[1] < 2:
                    print(f"[Warning] Sequence {i} too short for loss calculation, skipping.")
                    continue
                logits1 = logits1[:, :-1, :]
                logits2 = logits2[:, :-1, :]
                targets = input_sequences[i][:, 1:]
                print(f"[DEBUG] logits1 (sliced) shape: {logits1.shape}")
                print(f"[DEBUG] targets shape: {targets.shape}")
                if logits1.numel() == 0 or logits2.numel() == 0 or targets.numel() == 0:
                    print(f"[Warning] Sequence {i} resulted in empty logits/targets, skipping.")
                    continue
                loss1 = nn.CrossEntropyLoss()(logits1.reshape(-1, logits1.size(-1)), targets.reshape(-1))
                loss2 = nn.CrossEntropyLoss()(logits2.reshape(-1, logits2.size(-1)), targets.reshape(-1))
                # Calculate relative improvement: (loss1 - loss2) / loss1
                if loss1 > 0:
                    rel_improvement = (loss1 - loss2) / loss1
                else:
                    rel_improvement = 0.0
                loss_improvements.append(rel_improvement)
                if (i+1) % max(10, num_sequences//10) == 0 or (i+1) == num_sequences:
                    print(f"Loss Improvement: {i+1}/{num_sequences} ({(i+1)/num_sequences*100:.1f}%) done")
        return loss_improvements

    def test_memory_slot_usage(self, input_sequences, num_sequences=10):
        """Analyze memory slot usage patterns"""
        self.model.eval()
        slot_usage = []
        for i in range(num_sequences):
            with torch.no_grad():
                _ = self.model(input_sequences[i], update_memory=True)
            usage = self.model.imm.memory_mask.detach().cpu().clone()  # Track the full vector
            slot_usage.append(usage)
            if (i+1) % max(10, num_sequences//10) == 0 or (i+1) == num_sequences:
                print(f"Slot Usage: {i+1}/{num_sequences} ({(i+1)/num_sequences*100:.1f}%) done")
        return slot_usage

    def test_memory_update_frequency(self, input_sequences, num_sequences=10):
        """Test impact of different memory update frequencies"""
        self.model.eval()
        update_freq_results = []
        frequencies = [0.2, 0.4, 0.6, 0.8, 1.0]
        for freq in frequencies:
            freq_results = []
            for i in range(num_sequences):
                with torch.no_grad():
                    should_update = torch.rand(1) < freq
                    _ = self.model(input_sequences[i], update_memory=should_update)
                memory_state = self.model.imm.memory.clone()
                freq_results.append(memory_state)
                if (i+1) % max(10, num_sequences//10) == 0 or (i+1) == num_sequences:
                    print(f"Update Freq {freq:.1f}: {i+1}/{num_sequences} ({(i+1)/num_sequences*100:.1f}%) done")
            update_freq_results.append(freq_results)
        return update_freq_results

    def _calculate_pattern_similarity(self, memory, hidden_state_mean):
        # Accepts memory as [num_slots, n_embd] or [batch, num_slots, n_embd]
        if memory.dim() == 2:
            memory = memory.unsqueeze(0)  # [1, num_slots, n_embd]
        batch_size, num_slots, n_embd = memory.shape
        # hidden_state_mean: [batch, n_embd] or [n_embd]
        if hidden_state_mean.dim() == 1:
            hidden_state_mean = hidden_state_mean.unsqueeze(0)
        
        # Normalize vectors for cosine similarity
        memory_norm = F.normalize(memory, dim=-1)
        hidden_norm = F.normalize(hidden_state_mean, dim=-1)
        
        # Compute cosine similarity between each memory slot and hidden state mean
        similarities = []
        for b in range(batch_size):
            mem_slots = memory_norm[b]  # [num_slots, n_embd]
            h = hidden_norm[b]  # [n_embd]
            
            # Compute cosine similarity (dot product of normalized vectors)
            sim = torch.matmul(mem_slots, h)  # [num_slots]
            
            # Handle any remaining NaN values
            sim = torch.nan_to_num(sim, nan=0.0)
            
            similarities.append(sim.mean().item())
        
        # Average across batch and handle any remaining NaN values
        result = float(sum(similarities) / len(similarities))
        return 0.0 if np.isnan(result) else result

    def _calculate_prediction_influence(self, logits1, logits2):
        """Calculate how much memory influenced predictions"""
        # Compare prediction distributions
        probs1 = torch.softmax(logits1, dim=-1)
        probs2 = torch.softmax(logits2, dim=-1)
        influence = torch.abs(probs1 - probs2).mean().item()
        return influence

    def plot_results(self, results, title, xlabel, ylabel, save_path=None):
        """Plot experiment results"""
        plt.figure(figsize=(10, 6))
        plt.plot(results)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.show()

def run_experiments():
    # Initialize model
    config = GPTConfig()
    model = GPT(config)
    
    # Compute and print model sizes
    total_params, memory_params, rest_params = compute_model_sizes(model)
    print(f"Total model parameters: {total_params:,}")
    print(f"Memory module parameters: {memory_params:,} ({memory_params/total_params*100:.2f}%)")
    print(f"Rest of the model parameters: {rest_params:,} ({rest_params/total_params*100:.2f}%)")
    
    # Create experiment instance
    experiments = MemoryExperiments(model)
    experiments.setup_tracking()
    
    # Load real Shakespeare data
    num_sequences = 200
    block_size = 256
    input_sequences = load_shakespeare_data(block_size, num_sequences)
    
    # Run experiments
    pattern_similarities = experiments.test_memory_pattern_storage(input_sequences, num_sequences=num_sequences)
    loss_imp = experiments.test_memory_loss_improvement(input_sequences, num_sequences=num_sequences)
    slot_usage = experiments.test_memory_slot_usage(input_sequences, num_sequences=num_sequences)
    update_freq_results = experiments.test_memory_update_frequency(input_sequences, num_sequences=num_sequences)
    
    # Plot and save results
    experiments.plot_results(pattern_similarities, 
                           "Pattern Similarity Over Time",
                           "Sequence Number",
                           "Similarity Score",
                           save_path="pattern_similarity.png")
    
    experiments.plot_results(loss_imp,
                           "Memory Loss Improvement",
                           "Sequence Number",
                           "Loss Improvement",
                           save_path="loss_improvement.png")
    
    # Plot and save slot usage
    slot_usage_tensor = torch.stack(slot_usage)
    plt.figure(figsize=(10, 6))
    plt.imshow(slot_usage_tensor.mean(dim=0).numpy())
    plt.title("Memory Slot Usage Heatmap")
    plt.xlabel("Slot Index")
    plt.ylabel("Sequence Number")
    plt.colorbar()
    plt.savefig("slot_usage_heatmap.png")
    plt.close()

def sweep_experiments():
    # Sweep settings
    slot_counts = [4, 8, 16, 32]
    seq_lens = [64, 128, 256]
    num_sequences = 20
    results = []

    print("\nStarting sweep over memory slots and sequence lengths...\n")
    for num_slots in slot_counts:
        for block_size in seq_lens:
            # Setup model config
            config = GPTConfig()
            config.block_size = block_size
            config.memory_slots = num_slots
            model = GPT(config)
            # Compute and print model sizes
            total_params, memory_params, rest_params = compute_model_sizes(model)
            # Prepare data
            input_sequences = load_shakespeare_data(block_size, num_sequences)
            # Run experiments
            experiments = MemoryExperiments(model)
            pattern_sim = experiments.test_memory_pattern_storage(input_sequences, num_sequences=num_sequences)
            slot_usage = experiments.test_memory_slot_usage(input_sequences, num_sequences=num_sequences)
            avg_pattern_sim = np.mean(pattern_sim)
            std_pattern_sim = np.std(pattern_sim)
            slot_usage_tensor = torch.stack(slot_usage)
            avg_slot_usage = slot_usage_tensor.detach().mean(dim=0).numpy()
            results.append({
                'slots': num_slots,
                'block_size': block_size,
                'pattern_sim': avg_pattern_sim,
                'pattern_sim_std': std_pattern_sim,
                'slot_usage': avg_slot_usage,
                'mem_pct': memory_params/total_params*100
            })
            print(f"Slots: {num_slots}, Block: {block_size} | PatternSim: {avg_pattern_sim:.4f} ± {std_pattern_sim:.4f} | Mem%: {memory_params/total_params*100:.2f}")
    # Print summary table
    print("\nSweep Summary:")
    print(f"{'Slots':>5} {'Block':>6} {'PatSim':>8} {'Mem%':>6}")
    for r in results:
        print(f"{r['slots']:>5} {r['block_size']:>6} {r['pattern_sim']:>8.4f} {r['mem_pct']:>6.2f}")

def sweep_memory_hyperparams():
    # Best configuration from previous sweep
    num_slots = 8
    block_size = 256
    num_sequences = 20
    results = []

    # Hyperparameters to sweep
    learning_rates = [0.01, 0.05, 0.1, 0.2]
    norm_strategies = ['l2', 'layer']
    temperatures = [0.1, 0.5, 1.0, 2.0]

    print("\nStarting sweep over memory hyperparameters (lr, norm, temp) for best config (8 slots, block size 256)...\n")
    for lr in learning_rates:
        for norm in norm_strategies:
            for temp in temperatures:
                # Setup model config
                config = GPTConfig()
                config.block_size = block_size
                config.memory_slots = num_slots
                model = GPT(config)
                # Override memory hyperparameters
                model.imm.lr = lr
                model.imm.norm_type = norm
                model.imm.temperature = torch.nn.Parameter(torch.tensor(temp))
                # Compute and print model sizes
                total_params, memory_params, rest_params = compute_model_sizes(model)
                # Prepare data
                input_sequences = load_shakespeare_data(block_size, num_sequences)
                # Run experiments
                experiments = MemoryExperiments(model)
                pattern_sim = experiments.test_memory_pattern_storage(input_sequences, num_sequences=num_sequences)
                avg_pattern_sim = np.mean(pattern_sim)
                std_pattern_sim = np.std(pattern_sim)
                results.append({
                    'lr': lr,
                    'norm': norm,
                    'temp': temp,
                    'pattern_sim': avg_pattern_sim,
                    'pattern_sim_std': std_pattern_sim,
                    'mem_pct': memory_params/total_params*100
                })
                print(f"lr: {lr}, norm: {norm}, temp: {temp} | PatternSim: {avg_pattern_sim:.4f} ± {std_pattern_sim:.4f} | Mem%: {memory_params/total_params*100:.2f}")
    # Print summary table
    print("\nSweep Summary:")
    print(f"{'lr':>6} {'norm':>6} {'temp':>6} {'PatSim':>8} {'Mem%':>6}")
    for r in results:
        print(f"{r['lr']:>6.2f} {r['norm']:>6} {r['temp']:>6.1f} {r['pattern_sim']:>8.4f} {r['mem_pct']:>6.2f}")

def validate_best_config():
    # Best configuration from hyperparameter sweep
    num_slots = 8
    block_size = 256
    num_sequences = 20
    num_experiments = 5
    results = []

    print("\nValidating best configuration (lr = 0.05, norm = layer, temp = 1.0)...\n")
    for exp in range(num_experiments):
        # Setup model config
        config = GPTConfig()
        config.block_size = block_size
        config.memory_slots = num_slots
        model = GPT(config)
        # Override memory hyperparameters
        model.imm.lr = 0.05
        model.imm.norm_type = 'layer'
        model.imm.temperature = torch.nn.Parameter(torch.tensor(1.0))
        # Prepare data
        input_sequences = load_shakespeare_data(block_size, num_sequences)
        # Run experiments
        experiments = MemoryExperiments(model)
        pattern_sim = experiments.test_memory_pattern_storage(input_sequences, num_sequences=num_sequences)
        avg_pattern_sim = np.mean(pattern_sim)
        std_pattern_sim = np.std(pattern_sim)
        results.append(avg_pattern_sim)
        print(f"Experiment {exp+1}/{num_experiments} | PatternSim: {avg_pattern_sim:.4f} ± {std_pattern_sim:.4f}")

    # Print final summary
    final_avg = np.mean(results)
    final_std = np.std(results)
    print(f"\nFinal Results (Best Config):")
    print(f"Average Pattern Similarity: {final_avg:.4f} ± {final_std:.4f}")

def fine_tune_best_config():
    # Best configuration from hyperparameter sweep
    num_slots = 8
    block_size = 256
    num_sequences = 20
    results = []

    # Fine-tune hyperparameters
    learning_rates = [0.03, 0.04, 0.05, 0.06, 0.07]
    temperatures = [0.8, 0.9, 1.0, 1.1, 1.2]

    print("\nFine-tuning best configuration (lr = 0.05, norm = layer, temp = 1.0)...\n")
    for lr in learning_rates:
        for temp in temperatures:
            # Setup model config
            config = GPTConfig()
            config.block_size = block_size
            config.memory_slots = num_slots
            model = GPT(config)
            # Override memory hyperparameters
            model.imm.lr = lr
            model.imm.norm_type = 'layer'
            model.imm.temperature = torch.nn.Parameter(torch.tensor(temp))
            # Prepare data
            input_sequences = load_shakespeare_data(block_size, num_sequences)
            # Run experiments
            experiments = MemoryExperiments(model)
            pattern_sim = experiments.test_memory_pattern_storage(input_sequences, num_sequences=num_sequences)
            avg_pattern_sim = np.mean(pattern_sim)
            std_pattern_sim = np.std(pattern_sim)
            results.append({
                'lr': lr,
                'temp': temp,
                'pattern_sim': avg_pattern_sim,
                'pattern_sim_std': std_pattern_sim
            })
            print(f"lr: {lr}, temp: {temp} | PatternSim: {avg_pattern_sim:.4f} ± {std_pattern_sim:.4f}")
    # Print summary table
    print("\nFine-Tuning Summary:")
    print(f"{'lr':>6} {'temp':>6} {'PatSim':>8} {'Std':>8}")
    for r in results:
        print(f"{r['lr']:>6.2f} {r['temp']:>6.1f} {r['pattern_sim']:>8.4f} {r['pattern_sim_std']:>8.4f}")

def test_memory_recall():
    # Best configuration from fine-tuning
    num_slots = 8
    block_size = 256
    num_sequences = 20

    # Modern facts to test recall
    modern_facts = [
        "The first iPhone was released in 2007.",
        "Python was created by Guido van Rossum.",
        "The World Wide Web was invented by Tim Berners-Lee.",
        "The first human landing on the Moon was in 1969.",
        "The first electric car was produced in the 19th century."
    ]

    print("\nTesting memory recall for modern facts (best config: lr = 0.04, temp = 0.8)...\n")
    
    # Setup model config
    config = GPTConfig()
    config.block_size = block_size
    config.memory_slots = num_slots
    model = GPT(config)
    
    # Override memory hyperparameters
    model.imm.lr = 0.04
    model.imm.norm_type = 'layer'
    model.imm.temperature = torch.nn.Parameter(torch.tensor(0.8))

    # Initialize experiments
    experiments = MemoryExperiments(model)

    # First pass: Present facts during inference
    print("\nFirst Pass: Presenting facts during inference...")
    for fact in modern_facts:
        # Create a sequence that includes the fact in the middle
        # Pad with random Shakespeare data before and after the fact
        fact_chars = [ord(c) for c in fact]
        fact_len = len(fact_chars)
        
        # Load some Shakespeare data for context
        shakespeare_data = load_shakespeare_data(block_size, 1)[0]
        shakespeare_chars = shakespeare_data[0].tolist()
        
        # Create sequence: [shakespeare] + [fact] + [shakespeare]
        pre_fact_len = (block_size - fact_len) // 2
        post_fact_len = block_size - fact_len - pre_fact_len
        
        sequence = shakespeare_chars[:pre_fact_len] + fact_chars + shakespeare_chars[pre_fact_len:pre_fact_len + post_fact_len]
        sequence_tensor = torch.tensor(sequence).unsqueeze(0)
        
        # Run inference with the sequence
        with torch.no_grad():
            logits, _, hidden_states = model(sequence_tensor, update_memory=True, return_hidden_states=True)
        
        # Calculate pattern similarity for this fact
        memory = model.imm.memory
        pattern_sim = experiments._calculate_pattern_similarity(memory, hidden_states.mean(dim=1))
        
        print(f"\nFact presented: {fact}")
        print(f"Pattern Similarity during presentation: {pattern_sim:.4f}")
        print("Memory state after presentation:")
        print(f"- Memory norm: {torch.norm(memory).item():.4f}")
        print(f"- Memory mask mean: {model.imm.memory_mask.mean().item():.4f}")
        print("---")

    # Second pass: Test recall with partial facts
    print("\nSecond Pass: Testing recall with partial facts...")
    for fact in modern_facts:
        # Create partial fact by removing some words
        words = fact.split()
        if len(words) > 2:
            partial_fact = " ".join(words[:2])  # Take first two words
        else:
            partial_fact = words[0]  # Take first word if only two words
        
        # Create sequence with partial fact
        partial_chars = [ord(c) for c in partial_fact]
        partial_len = len(partial_chars)
        
        # Load fresh Shakespeare data for context
        shakespeare_data = load_shakespeare_data(block_size, 1)[0]
        shakespeare_chars = shakespeare_data[0].tolist()
        
        # Create sequence: [shakespeare] + [partial_fact] + [shakespeare]
        pre_fact_len = (block_size - partial_len) // 2
        post_fact_len = block_size - partial_len - pre_fact_len
        
        sequence = shakespeare_chars[:pre_fact_len] + partial_chars + shakespeare_chars[pre_fact_len:pre_fact_len + post_fact_len]
        sequence_tensor = torch.tensor(sequence).unsqueeze(0)
        
        # Run inference to test recall
        with torch.no_grad():
            logits, _, hidden_states = model(sequence_tensor, update_memory=False, return_hidden_states=True)
        
        # Calculate pattern similarity for recall
        memory = model.imm.memory
        pattern_sim = experiments._calculate_pattern_similarity(memory, hidden_states.mean(dim=1))
        
        # Calculate attention scores
        attn_scores = model.imm.attn_scores if hasattr(model.imm, 'attn_scores') else None
        
        print(f"\nPartial fact: {partial_fact}")
        print(f"Original fact: {fact}")
        print(f"Pattern Similarity during recall: {pattern_sim:.4f}")
        print("Memory state during recall:")
        print(f"- Memory norm: {torch.norm(memory).item():.4f}")
        print(f"- Memory mask mean: {model.imm.memory_mask.mean().item():.4f}")
        if attn_scores is not None:
            print(f"- Attention scores mean: {attn_scores.mean().item():.4f}")
        print("---")

    # Final memory state analysis
    print("\nFinal Memory State Analysis:")
    memory = model.imm.memory
    memory_mask = model.imm.memory_mask
    print(f"Final memory norm: {torch.norm(memory).item():.4f}")
    print(f"Final memory mask mean: {memory_mask.mean().item():.4f}")
    print(f"Memory slot usage: {memory_mask.mean(dim=0).tolist()}")

if __name__ == "__main__":
    test_memory_recall() 