# xzy_mini.py
#
# A minimal GPT-style language model trained on motivational phrases.
# Built from scratch using PyTorch to demonstrate the core Transformer stack.
#
# Author: Marcos Júnior Lemes Ferreira
# Repository: https://github.com/marcos-j-ferreira/model-xzy-generative
# Hugging Face: https://huggingface.co/spaces/marcos-j-ferreira/xzy_mini

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────

DATASET = [
    "i am not done",
    "it is not over",
    "i will not quit",
    "i keep going",
    "i keep trying",
    "i move forward",
    "i stay strong",
    "i believe in myself",
    "i will win",
    "i will succeed",
    "i keep fighting",
    "i do not stop",
    "i go until the end",
    "i never give up",
    "this is not the end",
    "i am still here",
    "i am getting stronger",
    "i am improving",
    "i am learning",
    "i stay focused",
    "i keep pushing",
    "i will make it",
    "i am not finished",
    "i rise again",
    "i try again",
    "i keep moving",
    "i stay determined",
    "i do not lose",
    "i keep my goal",
    "i fight for my dream",
]


# ─────────────────────────────────────────────
#  Vocabulary
# ─────────────────────────────────────────────

all_words   = " ".join(DATASET)
vocab       = sorted(set(all_words.split()))
word2idx    = {word: idx for idx, word in enumerate(vocab)}
idx2word    = {idx: word for idx, word in enumerate(vocab)}
VOCAB_SIZE  = len(vocab)


def tokenize(text: str) -> list[int]:
    """Converts a string of known words into a list of token indices."""
    return [word2idx[word] for word in text.strip().split() if word in word2idx]


tokenized_dataset = [tokenize(sentence) for sentence in DATASET]

print("-" * 50)
print("Vocabulary summary")
print(f"  Size   : {VOCAB_SIZE} words")
#print(f"  Words  : {vocab}")
print("-" * 50)


# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────

class MiniGPT(nn.Module):
    """
    Minimal GPT-style model using PyTorch's built-in TransformerEncoder.

    Architecture:
        Token Embedding  →  Positional Embedding  →  Transformer Stack  →  Linear Head
    """

    def __init__(
        self,
        vocab_size:    int,
        embedding_dim: int,
        num_heads:     int,
        num_layers:    int,
        max_seq_len:   int,
        ffn_dim:       int,
    ):
        super().__init__()

        # Maps each token index to a dense vector
        self.token_emb = nn.Embedding(vocab_size, embedding_dim)

        # Learnable positional encoding (one vector per position)
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim)

        # Stack of Transformer encoder layers with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Projects the hidden state to vocabulary logits
        self.linear_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: LongTensor of shape (batch, seq_len)
        Returns:
            logits: FloatTensor of shape (batch, seq_len, vocab_size)
        """
        seq_len = tokens.shape[1]

        token_emb = self.token_emb(tokens)
        positions = torch.arange(seq_len, device=tokens.device)
        pos_emb   = self.pos_emb(positions).unsqueeze(0)   # (1, seq_len, dim)
        x         = token_emb + pos_emb

        # Causal mask: each position can only attend to past positions
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tokens.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)

        return self.linear_head(x)


# ─────────────────────────────────────────────
#  Hyperparameters
# ─────────────────────────────────────────────

EMBEDDING_DIM = 32
NUM_HEADS     = 2
NUM_LAYERS    = 2
MAX_SEQ_LEN   = 10
FFN_DIM       = 64       # rule of thumb: 2–4× embedding_dim

model     = MiniGPT(VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN, FFN_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")


# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────

EPOCHS = 500

print("\nTraining...\n")

for epoch in range(EPOCHS):
    total_loss = 0.0
    model.train()

    for seq in tokenized_dataset:
        if len(seq) < 2:
            continue

        tokens        = torch.tensor(seq).unsqueeze(0)   # (1, seq_len)
        input_tokens  = tokens[:, :-1]                   # all but last
        target_tokens = tokens[:, 1:]                    # all but first

        logits = model(input_tokens)                     # (1, seq_len-1, vocab_size)

        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE),
            target_tokens.view(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(tokenized_dataset)

    if epoch % 50 == 0:
        print(f"  Epoch {epoch:>4} | Loss: {avg_loss:.4f}")

print("\nTraining complete.\n")


# ─────────────────────────────────────────────
#  Text generation
# ─────────────────────────────────────────────

def generate(model: nn.Module, prompt: str, max_new_tokens: int = 25) -> str:
    """
    Generates a sequence of tokens given a prompt string.

    Args:
        model          : trained MiniGPT instance
        prompt         : starting words (must exist in vocabulary)
        max_new_tokens : how many tokens to generate after the prompt

    Returns:
        Full generated sentence as a string.
    """
    model.eval()
    tokens = torch.tensor([tokenize(prompt)])

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits     = model(tokens[:, -MAX_SEQ_LEN:])  # respect max context window
            last_logits = logits[:, -1, :]
            probs      = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens     = torch.cat([tokens, next_token], dim=1)

    return " ".join(idx2word[i] for i in tokens[0].tolist())


# ─────────────────────────────────────────────
#  Interactive demo  (Hugging Face / terminal)
# ─────────────────────────────────────────────


def stream_output(text: str, delay: float = 0.08) -> None:
    """Prints words one by one to mimic streaming output."""
    words = text.split()
    for i, word in enumerate(words):
        print(word, end=" ", flush=True)
        if (i + 1) % 10 == 0:
            print()
        time.sleep(delay)
    print("\n")


print("=" * 50)
print("MiniGPT — Motivational Text Generator")
print("=" * 50)
print(f"Available starting words: {vocab}\n")

while True:
    user_input = input("Enter a word to start (or 'quit' to exit): ").strip().lower()

    if user_input in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    if user_input not in word2idx:
        print(f"  ⚠  '{user_input}' is not in the vocabulary. Try one of: {vocab}\n")
        continue

    print("\nGenerating", end="", flush=True)
    for _ in range(3):
        time.sleep(0.3)
        print(".", end="", flush=True)
    print("\n")

    output = generate(model, user_input, max_new_tokens=25)
    stream_output(output)
