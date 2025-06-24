from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import inspect
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # Because CausalSelfAttention object occurs in Block objects, which have residual pathways
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to the batch
        # nh is "number of heads", hs is "head size", and C = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64 -> C = nh * hs = 768
        qkv = self.c_attn(x) # (B, T, C) * (B, C, 3*C) = (B, T, 3*C)
        q,k,v = qkv.split(self.n_embd, dim=2) # (B, T, C)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # # compute attention scores
        # att = (q@k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # apply the causal mask
        # att = F.softmax(att, dim=-1) # (B, nh, T, T)
        # # perform the attention and combine heads
        # y = att @ v # (B, nh, T, T) * (B, nh, T, hs) = (B, nh, T, hs)
        # flash attention: does not materialize attention matrices (do not wait for full matrices), instead of loading a small part and doing all the computation (including softmax by using online softmax computation)
        y = F.scaled_dot_product_attention(q,k,v, is_causal=True) # use flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y) # (B, T, C)
        return y

        
        
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU() # Avoid dead RELU
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # Because MLP objects occurs in Block objects, which have residual pathways

    def forward(self, x):
        x = self.c_fc(x) # fc: fully-connected
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # residual pathway
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPR merges + 256 bytes tokens + 1 <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme (GPT-2)
        self.transformer.wte.weight = self.lm_head.weight

        # init params (compensate for the embedding dimension, = 1/sqrt(d_models))
        self.apply(self._init_weights)
    
    # compensate for the embedding dimension, = 1/sqrt(d_models)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # 2: two residual pathways in each block, n_layer block, Var(X) = 1 -> Var(sum(X)) = n -> std(sum(x)) = sqrt(n) -> initialize 1/sqrt(n) will get the std of output is 1
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    
    def forward(self, idx, targets=None): # Output is logits
        # idx is of shape (B,T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # shape (B, T, n_embd)
        x = pos_emb + tok_emb # Broadcast B dimesion to pos
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x) #(B,T,n_embd)
        logits = self.lm_head(x) # shape (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Load a pretained GPT-2 model from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large','gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretained gpt: %s" % model_type)

        # n_layer, n_head, n_embd are determined by the model type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initiliazed minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask/buffer, not params

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # ignore these, just a mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically, the openai checkpoints use a "Conv1D" module, but we want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(t) for t in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other params
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn:p for pn, p in self.named_parameters()}
        param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2] #n: name
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} paramerters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # create AdamW optimizer and use the fused version if it is available
        # fused the kernel (gather more operatios on one kernel) for better optimization.
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

import tiktoken
class DataLoader:
    def __init__(self, B, T, process_rank, num_processes): #num_processes: num of GPUs
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        #print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        # state
        self.current_position = self.B * self.T * self.process_rank # for each process that does not overlap each other
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x = buf[:-1].view(B,T) # inputs
        y = buf[1:].view(B,T)  # targets
        # advance the position in the tensor
        self.current_position += B*T*self.num_processes # move to the next training loop because each process handles each batch size: B*T, total batch size: B*T*num_processes
        # if loadind the next batch out of bound, reset
        if self.current_position + B*T*self.num_processes + 1 > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x,y

# running the training loops with multiple GPUs
from torch.distributed import init_process_group, destroy_process_group

# set up DDP (distributed data parallel)
# torchrun command sets the env variable RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run, os.environ is a dict
if ddp: # each process runs for each GPU
    # use of DDP atm (at the moment) demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl') # each process runs this script independently
    ddp_rank = int(os.environ['RANK']) # global rank
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # local index of GPU on node
    ddp_world_size = int(os.environ['WORLD_SIZE']) # total number of GPUs
    # set the device
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank==0 # this process wii do logging, checkpoint of cuda:0 etc
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device='cpu'
    if torch.cuda.is_available():
        device='cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # for macbook
        device = 'mps'
    print(f"using device: {device}")

# DDP launch for e.g. 8 GPUs: torchrun --standalone --nproc_per_node=8 GPT_2.py







# gradient accumulation because of large batch size
total_batch_size = 524288 # 2**19
B = 16
T = 1024
assert total_batch_size % (B*T*ddp_world_size) == 0, "make sure total_batch_size is divisible by B*T*ddp_world_size"
grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


# if this batch makes the overload memory of GPU, decrease the size of the batch (.e.g. 4)``
train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size) # the hyperparameters is evenly divided by 4 are the best because of its matrix multiplication structure.

torch.set_float32_matmul_precision("high") # convert from FP32 to TF32 (on A100)
num_return_sequences = 5
max_length = 50

# instantiate a model
#model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig(vocab_size=50304)) # exploit 4x4 matrix mutiplication structure
model.to(device)
model = torch.compile(model) # compile the file into a compiled file that provide overall image, not execute piece by piece (Python)
# -> load the values into kernels and do operations as much as possible, avoid round-trips on slow streams between HBM and kernels.
if ddp:
    model = DDP(model, device_ids = [ddp_local_rank]) # average the gradients in backward pass and cast back to GPUs (overdue)

# GPT-3 hyperparameters
max_lr = 6e-4 
min_lr = 0.1 * max_lr
warmup_steps = 10
max_steps = 50

# get the learning rate for each training step
def get_lr(it):
    # linear warmup for warmup iter steps
    if it < warmup_steps:
        return max_lr * (it+1) /warmup_steps
    # if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1, goes to 0
    return min_lr + (max_lr - min_lr) * coeff


# optimize
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device) #hyperparams of GPT-3



for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # convert from TF32 to BF16
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x,y)
        loss /= grad_accum_steps # because we take the mean of NLL or MSE not sum
        loss_accum += loss.detach() # for loss inspecting purpose
        # to check the datatype of loss
        # import code; code.interact(local=locals())
        # sync only the last step of inner loop to avoid wasting synchronization computation
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward() # loss.backward(): += the gradient like in micrograd
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    # clip the grad norm because of bad batches -> high loss -> high grad -> shock the model.
    # if norm > 1, scale down the norm to 1.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups: #only 1 param_groups dict by default.
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1-t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed // dt
    if master_process: # print only the result of cuda:0
        print(f"step {step:4d} | loss: {loss_accum.item():.4f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec}")
    # 4e: scientific notation (e.g. 1e-9)
# A100: 4x4 matrix multiplication, TF32: truncated the mantissa (precision) part of numbers (crop out 13 bits from FP32).
# BF16: truncated the mantissa part three bits from FP32, FP16: truncated the range -> need to gradient scaling
# FP: floating-point, BF: brain floating point.
# Flash attention: compensate for torch.compile.







if ddp:
    destroy_process_group()

import sys; sys.exit(0)

model.eval()
model.to('cuda')

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to("cuda")

# generate! right now x is (B,T), B=5, T=8
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad(): # Although not running loss.backward(), torch.no_grad() is still useful.
        # Not build computation graph and store intermediate gradient values
        logits = model(x)
        # take logits at last position
        logits = logits[:,-1,:] # (B, vocab_size)
        # get the probabilites from logits
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) 
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1) # (B,1), the input tensor does not need to be a probability distribution (it will automatically normalize the input tensor), 
        # it needs not to contain any negative elements
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # select the corresponding elements along the last dimension of input tensor
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)





