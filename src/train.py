"""
BiLSTM Char-to-Word Model — Training Script
============================================
Trains a character-level BiLSTM classifier to predict a target word
from a chord string encoded as individual characters (e.g. "aclsifiac").

Usage:
    pip install torch pandas scikit-learn
    python train.py --data chord_dataset.csv

Outputs:
    model.pt       — trained model weights + vocab metadata
"""

import argparse
import random

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from wordfreq import word_frequency

# ---------------------------------------------------------------------------
# Config defaults (override via CLI args)
# ---------------------------------------------------------------------------
HIDDEN_DIM   = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
BATCH_SIZE   = 256
EPOCHS       = 20
LR           = 1e-3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
PAD_TOKEN    = "<PAD>"
UNK_TOKEN    = "<UNK>"

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class CharVocab:
    """Maps individual characters ↔ integer indices."""

    def __init__(self):
        self.token2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2token = [PAD_TOKEN, UNK_TOKEN]

    def build(self, sequences, min_freq=5):
        from collections import Counter
        counts = Counter(c for seq in sequences for c in seq)
        for char, freq in counts.items():
            if freq >= min_freq and char not in self.token2idx:
                self.token2idx[char] = len(self.idx2token)
                self.idx2token.append(char)

    def encode(self, chars):
        return [self.token2idx.get(c, 1) for c in chars]  # 1 = UNK

    def __len__(self):
        return len(self.idx2token)


class WordVocab:
    """Maps target words ↔ class indices."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def build(self, words):
        for w in sorted(set(words)):
            if w not in self.word2idx:
                self.word2idx[w] = len(self.idx2word)
                self.idx2word.append(w)

    def __len__(self):
        return len(self.idx2word)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChordDataset(Dataset):
    def __init__(self, rows, char_vocab, word_vocab):
        self.samples = []
        for chords_str, word in rows:
            chars = list(chords_str)
            x = char_vocab.encode(chars)
            y = word_vocab.word2idx[word]
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Pad sequences to the same length within a batch."""
    xs, ys = zip(*batch)
    lengths = [len(x) for x in xs]
    max_len = max(lengths)
    padded = torch.zeros(len(xs), max_len, dtype=torch.long)
    for i, x in enumerate(xs):
        padded[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return padded, torch.tensor(lengths), torch.tensor(ys, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ChordBiLSTM(nn.Module):
    def __init__(self, vocab_size, num_classes, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # BiLSTM outputs hidden_dim * 2 (forward + backward)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))

        # Pack for efficiency (skip padding in LSTM)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)

        # hidden shape: (num_layers * 2, batch, hidden_dim)
        # Take the last layer's forward and backward hidden states
        fwd = hidden[-2]  # last layer, forward
        bwd = hidden[-1]  # last layer, backward
        pooled = torch.cat([fwd, bwd], dim=1)  # (batch, hidden_dim * 2)

        return self.classifier(self.dropout(pooled))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths, y.to(device)
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths, y.to(device)
        logits = model(x, lengths)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',        default='data/chord_dataset.csv')
    parser.add_argument('--output',      default='src/model.pt')
    parser.add_argument('--epochs',      type=int,   default=EPOCHS)
    parser.add_argument('--batch_size',  type=int,   default=BATCH_SIZE)
    parser.add_argument('--hidden_dim',  type=int,   default=HIDDEN_DIM)
    parser.add_argument('--num_layers',  type=int,   default=NUM_LAYERS)
    parser.add_argument('--dropout',     type=float, default=DROPOUT)
    parser.add_argument('--lr',          type=float, default=LR)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    # Load data
    df = pd.read_csv(args.data, dtype=str, keep_default_na=False)
    rows = list(zip(df['chords'], df['target_word']))
    print(f"Loaded {len(rows):,} examples")

    # Build vocabularies from full dataset
    char_vocab = CharVocab()
    word_vocab = WordVocab()
    char_vocab.build([list(r[0]) for r in rows])
    word_vocab.build([r[1] for r in rows])
    print(f"Char vocab size  : {len(char_vocab)}")
    print(f"Word vocab size  : {len(word_vocab)}")

    # Group rows by word, then for each word put 1 variant in val, rest in train
    from collections import defaultdict
    word_to_rows = defaultdict(list)
    for r in rows:
        word_to_rows[r[1]].append(r)

    train_rows, val_rows = [], []
    for word, word_rows in word_to_rows.items():
        random.shuffle(word_rows)
        split = max(1, len(word_rows) // 10)  # 10% of each word's variants go to val
        val_rows.extend(word_rows[:split])
        train_rows.extend(word_rows[split:])

    train_ds = ChordDataset(train_rows, char_vocab, word_vocab)
    val_ds   = ChordDataset(val_rows,   char_vocab, word_vocab)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = ChordBiLSTM(
        vocab_size=len(char_vocab),
        num_classes=len(word_vocab),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters : {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Training
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        vl_loss, vl_acc = eval_epoch(model,  val_loader,   criterion, DEVICE)
        scheduler.step(vl_loss)

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"train loss {tr_loss:.4f}  acc {tr_acc:.4f}  |  "
              f"val loss {vl_loss:.4f}  acc {vl_acc:.4f}")

        # Save best checkpoint
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                'model_state': model.state_dict(),
                'char_vocab_token2idx': char_vocab.token2idx,
                'char_vocab_idx2token': char_vocab.idx2token,
                'word_vocab_word2idx':  word_vocab.word2idx,
                'word_vocab_idx2word':  word_vocab.idx2word,
                'word_frequencies': {w: word_frequency(w, 'en') for w in word_vocab.idx2word},
                'config': {
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'dropout':    args.dropout,
                },
            }, args.output)
            print(f"Saved best model (val acc {best_val_acc:.4f})")

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {args.output}")


if __name__ == '__main__':
    main()