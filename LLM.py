import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
BLOCK_SIZE = 64
EPOCHS = 10
LR = 3e-4
EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4

# -----------------------------
# LOAD TEXT
# -----------------------------
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# -----------------------------
# CHAR TOKENIZER (OFFLINE)
# -----------------------------
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# -----------------------------
# DATASET
# -----------------------------
class TextDataset(Dataset):
    def __len__(self):
        return len(data) - BLOCK_SIZE

    def __getitem__(self, idx):
        x = data[idx:idx + BLOCK_SIZE]
        y = data[idx + 1:idx + BLOCK_SIZE + 1]
        return x, y

loader = DataLoader(TextDataset(), batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# MODEL
# -----------------------------
class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(EMBED_DIM, EMBED_DIM * 3)
        self.fc = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.head_dim = EMBED_DIM // NUM_HEADS

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, NUM_HEADS, self.head_dim).transpose(1, 2)
        k = k.view(B, T, NUM_HEADS, self.head_dim).transpose(1, 2)
        v = v.view(B, T, NUM_HEADS, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(T, T)).to(x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.fc(out)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = SelfAttention()
        self.ff = nn.Sequential(
            nn.Linear(EMBED_DIM, 4 * EMBED_DIM),
            nn.GELU(),
            nn.Linear(4 * EMBED_DIM, EMBED_DIM)
        )
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.ln2 = nn.LayerNorm(EMBED_DIM)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)

        self.blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(NUM_LAYERS)]
        )

        self.ln = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)

        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln(x)
        return self.head(x)

# -----------------------------
# TRAIN
# -----------------------------
model = MiniGPT().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

print("Training started...")
for epoch in range(EPOCHS):
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

# -----------------------------
# GENERATE TEXT
# -----------------------------
def generate(start, length=300):
    model.eval()
    tokens = torch.tensor(encode(start), dtype=torch.long).unsqueeze(0).to(DEVICE)

    for _ in range(length):
        logits = model(tokens[:, -BLOCK_SIZE:])
        next_token = torch.argmax(logits[:, -1], dim=-1)
        tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)

    return decode(tokens[0].tolist())

print("\nGenerated text:\n")
print(generate("Machine learning "))
