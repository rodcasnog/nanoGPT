"""
A simple character-level language model based on the GPT2 architecture.
Trains on text data and generates new text sequences.
"""

import collections
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys

# Model Hyperparameters
BATCH_SIZE = 256        # Number of sequences processed in parallel
BLOCK_SIZE = 128        # Maximum context length for predictions
N_EMBD = 128            # Embedding dimension
N_HEADS = 8             # Number of attention heads
N_LAYERS = 8            # Number of transformer blocks
DROPOUT = 0.2           # Dropout rate

# Training Hyperparameters
MAX_ITERS = 10000       # Total training iterations
EVAL_INTERVAL = 1000    # How often to evaluate model performance
EVAL_ITERS = 256        # Number of batches to average for evaluation loss
LEARNING_RATE = 1e-4    # Optimizer learning rate

# Data & Setup
DATA_DIR = Path('data')
INPUT_FILES = ['DonQuixote.txt', 'ExemplaryNovels.txt']  # List of input text files
SEED = 0

# Device Configuration
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
print(f"Using device: {DEVICE}")

torch.manual_seed(SEED)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Data Loading and Preparation
def load_data(data_dir: Path, filenames: list[str]) -> str:
    if not filenames:
        raise ValueError("No input filenames provided.")

    texts = []
    for filename in filenames:
        file_path = data_dir / filename
        logging.info(f"Loading data from: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
            logging.info(f"{file_path} loaded successfully: {len(texts[-1])} characters.")
        except FileNotFoundError:
            logging.error(f"Error: Data file not found at {file_path}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"An error occurred while reading the file: {e}")
            sys.exit(1)
    
    return ''.join(texts)

# Load data
corpus_text = load_data(DATA_DIR, INPUT_FILES)

# Byte Pair Encoding (BPE) for vocabulary reduction
def byte_pair_encoding(text: str, num_merges: int = 100) -> list[str]:
    logging.info(f"Vocab size before BPE: {len(set(text))}, text length before BPE: {len(text)}")
    text = list(text)
    for i in range(num_merges):
        pairs = collections.defaultdict(int)
        for i in range(len(text) - 1):
            pairs[(text[i], text[i+1])] += 1
        if not pairs:
            logging.info("No more pairs to merge.")
            break
        max_pair = max(pairs, key=pairs.get)
        
        i = j = 0
        new_token = ''.join(max_pair)
        while i < len(text):
            if i + 1 < len(text) and (text[i], text[i+1]) == max_pair:
                text[j] = new_token
                i += 2
            else:
                text[j] = text[i]
                i += 1
            j += 1
        text = text[:j]
        if i % 100 == 0:
            logging.info(f"New vocab size: {len(set(text))}, new text length: {len(text)}")
    
    return text

BPE_text = byte_pair_encoding(corpus_text)
vocab = sorted(set(BPE_text))
VOCAB_SIZE = len(vocab)
logging.info(f"Vocabulary size after BPE: {VOCAB_SIZE}")

# Token mapping
token_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_token = {i: ch for i, ch in enumerate(vocab)}
encode = lambda s: [token_to_idx[c] for c in s]
decode = lambda l: ''.join([idx_to_token[i] for i in l])

# Split data into train and validation
data = torch.tensor(encode(BPE_text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
logging.info(f"Training data size: {len(train_data)}, Validation data size: {len(val_data)}")

# Data loading
def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of data (inputs x and targets y).

    Args:
        split (str): 'train' or 'val' to select the data split.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing input and target tensors.
    """
    data_source = train_data if split == 'train' else val_data
    # Generate random starting indices for sequences in the batch
    ix = torch.randint(len(data_source) - BLOCK_SIZE, (BATCH_SIZE,))
    # Stack the sequences into batch tensors
    x = torch.stack([data_source[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data_source[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    # Move tensors to the configured device
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

# Model Definition

class Head(nn.Module):
    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril is not a parameter, register it as a buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, head_size).
        """
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # Compute attention
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        # Mask out future positions
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # Perform the weighted aggregation of values
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, head_size: int, n_embd: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, BLOCK_SIZE, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (B, T, C).

        Returns:
            torch.Tensor: Output tensor (B, T, C).
        """
        # Concatenate outputs from all heads along the embedding dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, n_heads * head_size)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Inner layer is 4x embedding dim
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Projection back to embedding dim
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_heads
        if n_embd % n_heads != 0:
             logging.warning(f"Warning: n_embd ({n_embd}) should be divisible by n_heads ({n_heads}).")
        self.sa = MultiHeadAttention(n_heads, head_size, n_embd, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (B, T, C).

        Returns:
            torch.Tensor: Output tensor (B, T, C).
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, block_size: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads, block_size, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)
        logging.info("Language Model initialized.")
        logging.info(f"Number of parameters: {sum(p.numel() for p in self.parameters())/1e6:.2f} M")


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx (torch.Tensor): Input sequence indices (B, T).
            targets (torch.Tensor, optional): Target sequence indices (B, T). If provided, computes loss.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: Logits (B, T, vocab_size) and loss (if targets provided).
        """
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, C=n_embd)
        pos = torch.arange(T, device=DEVICE)
        pos_emb = self.position_embedding_table(pos) # (T, C=n_embd)
        x = tok_emb + pos_emb # (B, T, C) - broadcasting pos_emb: (1, T, C)

        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C_vocab = logits.shape
            logits_flat = logits.view(B * T, C_vocab)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Args:
            idx (torch.Tensor): Starting context sequence indices (B, T_start), B=1 typically.
            max_new_tokens (int): Maximum number of tokens to generate.

        Returns:
            torch.Tensor: The generated sequence including the context (B, T_start + max_new_tokens).
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits_last_step = logits[:, -1, :] # (B, vocab_size)
            probs = F.softmax(logits_last_step, dim=-1) # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        self.train()
        return idx

# Loss Estimation
@torch.no_grad()
def estimate_loss(model: LanguageModel, eval_iters: int) -> dict[str, float]:
    """
    Args:
        model (LanguageModel): The model to evaluate.
        eval_iters (int): The number of iterations (batches) to average over.

    Returns:
        dict[str, float]: A dictionary containing 'train' and 'val' average losses.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=DEVICE) 
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

if __name__ == "__main__":

    model = LanguageModel(
        vocab_size=VOCAB_SIZE,
        n_embd=N_EMBD,
        block_size=BLOCK_SIZE,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        dropout=DROPOUT
    )
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITERS, eta_min=LEARNING_RATE/10)

    logging.info("Starting training...")
    model.train()
    for iter_num in range(MAX_ITERS):
        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            losses = estimate_loss(model, EVAL_ITERS)
            logging.info(f"Iteration {iter_num:>{len(str(MAX_ITERS))}} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

        xb, yb = get_batch('train')

        # Forward pass
        logits, loss = model(xb, yb)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Update weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()


    logging.info("Training finished.")
    save_path = "language_model.pth"
    torch.save(model.state_dict(), save_path)
    logging.info(f"Model state dictionary saved to {save_path}")

    # Text Generation
    logging.info("Generating text...")
    start_context = torch.tensor([[token_to_idx['\n']]], dtype=torch.long, device=DEVICE) # Shape (1, 1)
    generated_indices = model.generate(start_context, max_new_tokens=5000)
    generated_text = decode(generated_indices[0].tolist())

    print("\n--- Generated Text ---")
    print(generated_text)
    print("----------------------\n")
