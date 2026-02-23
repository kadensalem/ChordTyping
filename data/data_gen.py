import re
from wordfreq import top_n_list
import random
import string
import pyphen

# ── tuneable hyperparams ───────────────────────────────────────────────────
DELETE_PROB  = 0.15   # chance each letter is dropped
ADD_PROB     = 0.15   # chance a noise letter is inserted after each letter
AUGMENT_N    = 5      # alternate versions per word per method
VOCAB_SIZE   = 20000  # number of original words to include in vocab
# ───────────────────────────────────────────────────────────────────────────

# ── Vocab generation ───────────────────────────────────────────────────────
def build_vocab(n_target=20000, lang="en"):
    """Build a vocabulary of the top n_target words in the specified language."""
    raw = top_n_list(lang, n_target * 3)
    vocab = []
    seen = set()

    for w in raw:
        w = w.lower().strip()

        # Filter out words with special characters
        if not re.fullmatch(r"[a-z]+(?:['-][a-z]+)*", w):
            continue

        if w in seen:
            continue

        seen.add(w)
        vocab.append(w)

        if len(vocab) >= n_target:
            break

    return vocab

# ── Word augmentation ───────────────────────────────────────────────────────

dic = pyphen.Pyphen(lang="en")

def alter_chunk(chunk: str) -> str:
    """Scramble, delete, and add noise to a single chunk of letters."""
    letters = list(chunk)

    # 1. scramble
    random.shuffle(letters)

    # 2. delete letters with low probability
    letters = [ch for ch in letters if random.random() > DELETE_PROB]

    # 3. insert noise letters with low probability
    result = []
    for ch in letters:
        result.append(ch)
        if random.random() < ADD_PROB:
            result.append(random.choice(string.ascii_lowercase))

    return "".join(result)

def augment_whole(word: str) -> list[str]:
    """Augment whole word by scrambling, deleting, and adding noise."""
    return [alter_chunk(word.lower()) for _ in range(AUGMENT_N)]


def augment_syllable(word: str) -> list[str]:
    """Augment word by splitting into syllables and altering each chunk."""
    syllables = dic.inserted(word.lower(), hyphen="|").split("|")
    results = []
    for _ in range(AUGMENT_N):
        altered = "".join(alter_chunk(s) for s in syllables)
        results.append(altered)
    return results

# ── Main ──────────────────────────────────────────────────────────────

vocab = build_vocab(VOCAB_SIZE, "en")
with open("data/data.txt", "w", encoding="utf-8") as f:
    for w in vocab:
        augments = list(dict.fromkeys(augment_whole(w) + augment_syllable(w))) # remove duplicates
        augments = [a.replace("'", "") for a in augments] # remove apostrophes
        f.write(w + " " + " ".join(augments) + "\n")