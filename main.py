"""
RNN Text Generator
Character-level LSTM language model for text generation with temperature sampling.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Training corpus — classic text
CORPUS = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub,
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life.

It was the best of times, it was the worst of times, it was the age of wisdom,
it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity,
it was the season of Light, it was the season of Darkness, it was the spring of hope,
it was the winter of despair, we had everything before us, we had nothing before us.

The quick brown fox jumps over the lazy dog.
A stitch in time saves nine. Actions speak louder than words.
All that glitters is not gold. Beauty is in the eye of the beholder.
Every cloud has a silver lining. Fortune favors the bold.
Give credit where credit is due. Haste makes waste.
If you want something done right, do it yourself.
Judge not, that ye be not judged. Knowledge is power.
Laughter is the best medicine. Make hay while the sun shines.
Necessity is the mother of invention. Once bitten, twice shy.
Practice makes perfect. Quality over quantity.
Rome was not built in a day. Slow and steady wins the race.
The early bird catches the worm. United we stand, divided we fall.
Virtue is its own reward. Waste not, want not. You reap what you sow.
""" * 5  # Repeat for more training data


class CharRNN(nn.Module):
    """Character-level LSTM language model."""
    def __init__(self, vocab_size, embed_dim=64, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embed      = nn.Embedding(vocab_size, embed_dim)
        self.lstm       = nn.LSTM(embed_dim, hidden_size, num_layers,
                                   batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout    = nn.Dropout(dropout)
        self.fc         = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))


def build_vocab(text):
    chars = sorted(set(text))
    ch2i  = {c: i for i, c in enumerate(chars)}
    i2ch  = {i: c for c, i in ch2i.items()}
    return chars, ch2i, i2ch


def make_sequences(encoded, seq_len=100, step=1):
    X, y = [], []
    for i in range(0, len(encoded) - seq_len, step):
        X.append(encoded[i:i + seq_len])
        y.append(encoded[i + 1:i + seq_len + 1])
    return np.array(X), np.array(y)


def temperature_sample(logits, temperature=1.0, top_k=0):
    """Sample next character with temperature scaling and optional top-k."""
    logits = logits / max(temperature, 1e-9)
    if top_k > 0:
        top_k_vals, _ = torch.topk(logits, top_k)
        min_val = top_k_vals[-1]
        logits = logits.masked_fill(logits < min_val, float('-inf'))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def generate_text(model, ch2i, i2ch, seed_text, length=300, temperature=0.7, device='cpu'):
    """Generate text from a seed string."""
    model.eval()
    chars = [ch2i.get(c, 0) for c in seed_text]
    input_ids = torch.LongTensor([chars]).to(device)
    hidden = model.init_hidden(1, device)

    generated = seed_text
    with torch.no_grad():
        # Warm up
        for _ in range(len(chars) - 1):
            _, hidden = model(input_ids[:, :1], hidden)
            input_ids = input_ids[:, 1:]

        inp = input_ids
        for _ in range(length):
            logits, hidden = model(inp, hidden)
            next_char_id   = temperature_sample(logits[0, -1], temperature)
            generated     += i2ch[next_char_id]
            inp = torch.LongTensor([[next_char_id]]).to(device)

    return generated


def train(model, loader, optimizer, criterion, device, clip=1.0):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        hidden = model.init_hidden(X_batch.size(0), device)
        hidden = (hidden[0].detach(), hidden[1].detach())
        optimizer.zero_grad()
        logits, _ = model(X_batch, hidden)
        loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def plot_training(losses, perplexities, save_path='training.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses, 'b-', lw=1.5)
    ax1.set_title('Training Loss'); ax1.set_xlabel('Epoch'); ax1.grid(True, alpha=0.3)
    ax2.plot(perplexities, 'r-', lw=1.5)
    ax2.set_title('Perplexity'); ax2.set_xlabel('Epoch'); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("RNN CHARACTER-LEVEL TEXT GENERATOR")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Prepare data
    chars, ch2i, i2ch = build_vocab(CORPUS)
    vocab_size = len(chars)
    encoded    = np.array([ch2i[c] for c in CORPUS])
    print(f"Corpus: {len(CORPUS):,} chars, vocab size: {vocab_size}")

    SEQ_LEN = 100
    X, y = make_sequences(encoded, SEQ_LEN, step=3)
    print(f"Sequences: {len(X)}")

    X_t = torch.LongTensor(X)
    y_t = torch.LongTensor(y)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

    model     = CharRNN(vocab_size, embed_dim=64, hidden_size=256, num_layers=2).to(device)
    params    = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    EPOCHS = 30
    losses, perplexities = [], []

    print(f"\n--- Training for {EPOCHS} epochs ---")
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, loader, optimizer, criterion, device)
        perp = np.exp(loss)
        losses.append(loss)
        perplexities.append(perp)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}: Loss={loss:.4f}, Perplexity={perp:.2f}")
            # Sample text
            seed = "To be, or not"
            sample = generate_text(model, ch2i, i2ch, seed, length=200, temperature=0.7, device=device)
            print(f"  Sample: '{sample[:100]}...'")

    print("\n--- Generated Text Samples ---")
    for temp in [0.3, 0.7, 1.0, 1.5]:
        text = generate_text(model, ch2i, i2ch, "The ", length=150, temperature=temp, device=device)
        print(f"\nTemperature={temp}:")
        print(f"  {text[:150]}")

    plot_training(losses, perplexities)

    # Save
    torch.save({'model': model.state_dict(), 'ch2i': ch2i, 'i2ch': i2ch,
                'vocab_size': vocab_size, 'seq_len': SEQ_LEN}, 'char_rnn.pth')
    print("\nModel saved to char_rnn.pth")
    print("Plot saved to training.png")
    print("\n✓ RNN Text Generator complete!")


if __name__ == '__main__':
    main()
