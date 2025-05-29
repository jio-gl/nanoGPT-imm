"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
3) Implicit Memory Module (IMM) with ImplicitMemory or LinformerMemory
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ImplicitMemory(nn.Module):
    """
    Implicit Memory Module (IMM) that writes a summary of the input hidden states,
    stores it in a memory bank, and later retrieves relevant memory via attention.
    """
    def __init__(self, config, num_slots=16):
        super().__init__()
        self.num_slots = num_slots
        self.n_embd = config.n_embd
        # Linear layers for write, query, and value operations
        self.write_layer = nn.Linear(config.n_embd, config.n_embd)
        self.query_layer = nn.Linear(config.n_embd, config.n_embd)
        self.value_layer = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(config.n_embd)
        # Memory bank is registered as a buffer; it will be re-initialized at each forward pass.
        self.register_buffer("memory", torch.zeros(num_slots, config.n_embd))

    def reset_memory(self, batch_size, device):
        # Initialize memory bank for each sample in the batch
        self.memory = torch.zeros(batch_size, self.num_slots, self.n_embd, device=device)

    def forward(self, x):
        # x: (batch, seq_len, n_embd)
        # Write: Compute summary from x and update memory.
        s = self.write_layer(x)  # (batch, seq_len, n_embd)
        # For simplicity, use the mean over sequence as the summary.
        s_avg = s.mean(dim=1, keepdim=True)  # (batch, 1, n_embd)
        # Instead of updating self.memory inplace, clone it.
        mem = self.memory.clone()  # (batch, num_slots, n_embd)
        mem[:, 0, :] = s_avg.squeeze(1)  # Update first memory slot with the summary.
        
        # Read: Compute query from x.
        q = self.query_layer(x)  # (batch, seq_len, n_embd)
        # Compute attention weights between q and memory.
        att = torch.matmul(q, mem.transpose(1, 2)) * self.scale  # (batch, seq_len, num_slots)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        # Compute value from memory.
        v = self.value_layer(mem)  # (batch, num_slots, n_embd)
        # Retrieve memory.
        r = torch.matmul(att, v)  # (batch, seq_len, n_embd)
        return r


class LinformerMemory(nn.Module):
    """
    Low-rank Implicit Memory Module (IMM) using Linformer-style attention for scalable memory retrieval.
    Reduces computational complexity from O(n_embd²) to O(n_embd * k), where k is the low-rank projection.
    """
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.num_slots = config.memory_slots
        self.memory = nn.Parameter(torch.zeros(self.num_slots, self.n_embd))
        self.memory_mask = nn.Parameter(torch.ones(self.num_slots))
        self.temperature = nn.Parameter(torch.ones(1))
        self.mha_proj = nn.Linear(self.n_embd, self.n_embd)
        self.context_proj = nn.Linear(self.num_slots, self.n_embd)
        self.reset_memory()

    def reset_memory(self):
        with torch.no_grad():
            # Initialize with smaller values for better stability
            self.memory.data = torch.randn(self.num_slots, self.n_embd) * 0.01
            self.memory_mask.data = torch.ones(self.num_slots)
            self.temperature.data = torch.ones(1)

    def update_memory(self, x):
        # x: [batch, seq_len, n_embd]
        # Normalize input for better stability
        x_norm = F.normalize(x, dim=-1)
        memory_norm = F.normalize(self.memory, dim=-1)
        
        # Compute attention scores with proper scaling
        attn_scores = torch.matmul(x_norm, memory_norm.t())  # [batch, seq_len, num_slots]
        attn_scores = attn_scores / (self.n_embd ** 0.5)  # Scale by sqrt(d_k)
        attn_scores = attn_scores / self.temperature  # Apply temperature scaling
        
        # Add small epsilon before softmax for numerical stability
        attn_scores = attn_scores + 1e-8
        attn_scores = F.softmax(attn_scores, dim=-1)  # [batch, seq_len, num_slots]
        
        # Compute memory update with proper scaling
        memory_update = torch.matmul(attn_scores.transpose(1, 2), x_norm)  # [batch, num_slots, n_embd]
        memory_update = memory_update.sum(dim=0)  # [num_slots, n_embd]
        
        # Normalize memory update
        memory_update = F.normalize(memory_update, dim=-1)
        
        # Adaptive learning rate based on update magnitude
        update_norm = torch.norm(memory_update, dim=-1, keepdim=True)
        lr = 0.01 / (1.0 + update_norm)  # Smaller updates for larger magnitudes
        
        # Update memory with adaptive learning rate
        self.memory.data = self.memory.data + lr * memory_update
        
        # Normalize memory after update
        self.memory.data = F.normalize(self.memory.data, dim=-1)
        
        # Ensure no nan values
        if torch.isnan(self.memory.data).any():
            print("[WARNING] NaN values detected in memory update. Resetting memory.")
            self.reset_memory()

    def forward(self, x, update_memory=True):
        # x: [batch, seq_len, n_embd]
        if update_memory:
            self.update_memory(x)
            
        # Normalize input for better stability
        x_norm = F.normalize(x, dim=-1)
        memory_norm = F.normalize(self.memory, dim=-1)
        
        # Compute memory context with normalized vectors
        context = torch.matmul(x_norm, memory_norm.t())  # [batch, seq_len, num_slots]
        context = self.context_proj(context)  # [batch, seq_len, n_embd]
        
        return context


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    memory_slots: int = 8  # Added memory_slots parameter with default value

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # Initialize the Implicit Memory Module (IMM)
        self.imm = LinformerMemory(config)  # Use LinformerMemory by default

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, update_memory=True, logits=None, return_hidden_states=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Process through Transformer blocks and integrate IMM output after each block
        for i, block in enumerate(self.transformer.h):
            x = block(x)
            # Get IMM output and integrate it
            if targets is not None and update_memory:
                # Get logits for memory update if not provided
                if logits is None:
                    block_logits = self.lm_head(x)
                else:
                    block_logits = logits
                # Update memory more frequently in later layers
                should_update = (i >= len(self.transformer.h) // 2) if update_memory else False
                r = self.imm(x, update_memory=should_update)
            else:
                r = self.imm(x, update_memory=update_memory)
            x = x + r  # Simple residual connection
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        if return_hidden_states:
            return logits, loss, x
        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
        sd_keys = [k for k in sd_keys if not k.startswith('imm.')] # exclude IMM-specific parameters

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        # Initialize IMM-specific parameters
        model.imm.reset_memory()  # Initialize with batch size 1

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
