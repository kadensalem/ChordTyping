"""
FastAPI Chord-to-Word Prediction Server
========================================
Features:
  - BiLSTM model inference
  - Word frequency bias (wordfreq)

Usage:
    pip install fastapi uvicorn torch wordfreq nltk
    uvicorn server:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import defaultdict
import math
import torch
import torch.nn as nn
import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH   = "model.pt"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
PAD_TOKEN    = "<PAD>"
UNK_TOKEN    = "<UNK>"
TOP_CANDIDATES = 50
FREQ_WEIGHT    = 0.5   # how strongly to bias toward common words (0 = off)

# ---------------------------------------------------------------------------
# Vocab + Model classes (must match train.py)
# ---------------------------------------------------------------------------

class ChordVocab:
    def __init__(self):
        self.token2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2token = [PAD_TOKEN, UNK_TOKEN]

    def encode(self, chords):
        return [self.token2idx.get(c, 1) for c in chords]

    def __len__(self):
        return len(self.idx2token)


class WordVocab:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def __len__(self):
        return len(self.idx2word)


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
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        fwd = hidden[-2]
        bwd = hidden[-1]
        pooled = torch.cat([fwd, bwd], dim=1)
        return self.classifier(self.dropout(pooled))


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    cfg = checkpoint['config']

    chord_vocab = ChordVocab()
    word_vocab = WordVocab()

    if 'chord_vocab' in checkpoint:
        chord_vocab = checkpoint['chord_vocab']
        word_vocab  = checkpoint['word_vocab']
    else:
        chord_vocab.token2idx = checkpoint['chord_vocab_token2idx']
        chord_vocab.idx2token = checkpoint['chord_vocab_idx2token']
        word_vocab.word2idx   = checkpoint['word_vocab_word2idx']
        word_vocab.idx2word   = checkpoint['word_vocab_idx2word']

    model = ChordBiLSTM(
        vocab_size=len(chord_vocab),
        num_classes=len(word_vocab),
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, chord_vocab, word_vocab


# ---------------------------------------------------------------------------
# Build word frequency table
# ---------------------------------------------------------------------------

def build_freq_table(vocab_words: list[str]) -> dict[str, float]:
    from wordfreq import word_frequency
    print("Building frequency table...")
    table = {w: math.log(word_frequency(w, 'en') + 1e-10) for w in vocab_words}
    print(f"  Done. {len(table)} words.")
    return table

# ---------------------------------------------------------------------------
# Startup: load everything
# ---------------------------------------------------------------------------

print(f"Loading model from {MODEL_PATH}...")
model, chord_vocab, word_vocab = load_model()
print(f"  {len(word_vocab)} words, {len(chord_vocab)} chord tokens.")

freq_table = build_freq_table(word_vocab.idx2word)

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    chords: str
    previous_word: str | None = None
    topk: int = 5

@app.post("/predict")
@torch.no_grad()
def predict(req: PredictRequest):
    # --- Model inference ---
    chars   = list(req.chords.strip())
    indices = chord_vocab.encode(chars)
    x       = torch.tensor([indices], dtype=torch.long).to(DEVICE)
    lengths = torch.tensor([len(indices)])

    logits = model(x, lengths)
    probs  = torch.softmax(logits, dim=1)[0]

    # Pull top candidates from model to rerank (more headroom for reranking)
    k = min(TOP_CANDIDATES, len(word_vocab))
    top_probs, top_indices = probs.topk(k)

    candidates = [
        (word_vocab.idx2word[idx.item()], prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]

    # --- Reranking ---
    prev = req.previous_word.lower() if req.previous_word else None

    def score(word: str, model_prob: float) -> float:
        s = math.log(model_prob + 1e-10)
        # Frequency bias
        s += FREQ_WEIGHT * freq_table.get(word, math.log(1e-10))
        return s

    reranked = sorted(candidates, key=lambda x: score(x[0], x[1]), reverse=True)

    return {
        "predictions": [
            {"word": word, "probability": round(prob, 4)}
            for word, prob in reranked[:req.topk]
        ]
    }

@app.get("/health")
def health():
    return {"status": "ok"}