import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
# Attention - multi head attention
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        # Flash attention implements the transposed calculation of the dot product of q and k, mask fill, softmax, dot product with v really fast 
        # Also a kernel fusion 
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        # Use gelu because of dead gradient problem for relu - at 0 the gradient is not exactly 0 
        # Later on in Llama they use swiGLU
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):  # transformer block
    # Residual pathways have normalization inside them in original paper - 
    # but we dont want normalization - use addition as a branch - gradients flow from the top 

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))  # feed forward network
        return x
    
    # Attention is a weighted sum or reduce function, mlp is a map
    # Transformer is a repeating pattern of map-reduce


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # position embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # transformer blocks
            ln_f = nn.LayerNorm(config.n_embd), # final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)
    # GPT 2 paper - weight initialization
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Std = sqrt(1/n) where n is model size (d_model_)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # Flatten 3-d to 2d tensor for crossentropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # Load pretrained weights (copy weights from pretrained model on Huggingface)
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

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

        return model

    # Common to not decay 1d tensors - biases, layernorms, etc.
    # Weight decay embeddings and tensors in matrix multiplications
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # Temp
        master_process = True
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
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
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")  # faster - kernel fusion into single kernel instead of for loop
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:  # Goes through the file in chunks
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # # get the shard filenames
        # data_root = "edu_fineweb10B"
        # shards = os.listdir(data_root)
        # shards = [s for s in shards if split in s]
        # shards = sorted(shards)
        # shards = [os.path.join(data_root, s) for s in shards]
        # self.shards = shards
        # assert len(shards) > 0, f"no shards found for split {split}"
        # if master_process:
        #     print(f"found {len(shards)} shards for split {split}")
        # self.reset()

        # Use the tiny shakespeare dataset
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (self.B * self.T)} batches")
        self.current_position = self.B * self.T * self.process_rank

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        # If there aren't enough tokens left for a full batch, reset to the beginning.
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = B * T * self.process_rank  # or simply 0

        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)   # targets
        # Advance the position for the next batch.
        self.current_position += B * T * self.num_processes
        return x, y


##################################################################
# Trying to run in DPP mode
# # simple launch:
    # # python train_gpt2.py
    # # DDP launch for e.g. 8 GPUs:
    # # torchrun --standalone --nproc_per_node=2 naive_train_gpt2.py
def main():
    # run the training loop
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist

    # set up DDP (distributed data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    # added after video, pytorch can be serious about it's device vs. device_type distinction
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    #####################################################################################

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    enc = tiktoken.get_encoding("gpt2")

    ### Training from randomly initialized weights
    from train_gpt2 import DataLoaderLite
    import time
    import math

    # Gradient accumulation
    total_batch_size = 524288  # 2^19 ~0.5M tokens (0.5 mill / T = B represents the Batch size as the 0.5 million batch size is in tokens)
    # B = 16  # Micro batch size
    # I am going to use 4 for the micro batch size as my gpu memory lower and didnt calculate the grad acum steps for B = 4 :(
    B = 4
    T = 1024  # sequence length
    assert total_batch_size % (B * T * ddp_world_size) == 0  # Check if total batch size is divisible by micro batch size and sequence length
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)  # 32 // When using DDP the grad accum steps also takes into account the number of processes
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"-> calculated grad accum steps: {grad_accum_steps}")  # (2^19 / (16 * 1024)) = grad acum


    train_loader = DataLoaderLite(B=2, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')  # 8 GB GPU limit at 4 batch size and 256 sequence length

    # Enable TF32 for faster matrix multiplications from Nvidia GPU
    torch.set_float32_matmul_precision('high')  # Multiplication itself is better but shipping FP32 around is not

    model = GPT(GPTConfig(vocab_size=50304))  # Use a power of 2 for optimization of kernel
    # New tokens are never used? Model has to learn that these weights need to be driven to 0 or negative infinity probabilities for their logits
    # Just adding more memory
    model.to('cuda')
    # x = x.to('cuda')
    # y = y.to('cuda')
    # logits, loss = model(x, y)

    model =  torch.compile(model) # More compilation overhead but faster training  -> reduce python overhead and gpu read/writes

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])  # In the forward pass - identitical, backward pass - each gpu has gradients - > allreduce = average the gradients and deposit them on all the GPUs on every rank

    # Cosine decay lr
    max_lr = 6e-4
    min_lr = max_lr * 0.1 # Min learning rate is 10% of max learning rate
    warmup_steps = 10
    max_steps = 50
    def get_lr(it):  # Learning rate scheduler - basically implementing the cosine decay curvve
        # linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps  # 0th iteration we dont use 0
        # If it > lr_decay_iters, then we're past the decay phase
        if it > max_steps:
            return min_lr
        # In between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges between 1 and 0
        return min_lr + coeff * (max_lr - min_lr)

    # print(logits.shape) # (B, T, vocab_size)
    # print(loss) # Loss is at 10.9 -> technically becasue we want to randomly initialize the weights
    # we dont want favouring a specific toke nearly on and 1/50257 - > cross entropy loss
    # -> negative log likelihood loss -> -ln(1/50257) whih is pretty much the loss

    # For loop for the optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    
    if ddp:
        optimizer = model.module.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type='cuda')
    else:
        optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type='cuda')

    # GPT-3 uses a cosine decay lr - down to 10% of its value over 260 billion tokens and continues at 10% of original lr; also a lr warmup over the first 375 million tokens
    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()  # zero the gradients
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x = x.to('cuda')
            y = y.to('cuda')
            # Also using mixed precision autocast from torch - we can do a tier lower to bf16 
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):  # Here we use bfloat16 instead of float16 because of gradient scaling issues?
                # Activation tensor is now bfloat16, logits; but parameters are still float16 - mix some in lower FP some stays in higher FP
                logits, loss = model(x, y)
            # calculating the loss is wrong - we lose the normalizer for minibatches
            loss = loss / grad_accum_steps  # add back the normalizer
            loss_accum += loss.detach()

            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)  # Only sync gradients on the last micro step - we dont need to sync gradients on every micro step

            loss.backward()  # backpropagate the loss; += the gradients (accumulation)  -> have the gradients of the param tensors
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)  # Allreduce - average the gradients and deposit them on all the GPUs on every rank
        # Clip them to have a max norm of 1.0
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip the gradients to prevent model gettign shocked of too big gradients
        
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.step()  # update the weights
        torch.cuda.synchronize()  # The commands above are orders instructions from CPU to GPU - need to synchronize to make sure the commands are executed before calculating the time
        t1 = time.time()
        dt = (t1 - t0) 
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_second = tokens_processed / dt
        if master_process:
            print(f"step {step} loss: {loss_accum.item():.6f}, lr: {lr:.4e}, norm: {norm:.4f}, dt: {dt:.2f}s, tok/sec: {tokens_per_second:.2f}")  # Norm of the gradient - climbing is bad, spikes are instability


        # Kernel fusion - reduce roundtrips from vram to GPU by torch compile doing all the computations at once
        # Gradient accumulation - instead of B=4, we do B=1 and accumulate the gradients 4 times - simulate larger batch sizes
        # DistributedDataParallel - parallel training across multiple GPUs - multiple processes and each process has a gpu - all processing different parts of the data and once they all calculate the gradients - average the gradients

    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    main()