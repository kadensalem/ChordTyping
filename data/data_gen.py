"""
Chord-to-Word Dataset Generator
================================
Generates a synthetic CSV dataset for training a chord-based word prediction model,
inspired by stenography. Each row maps a chord sequence to a target word.

Chord encoding:
  - Split word into syllables
  - For each syllable: remove duplicate letters, sort alphabetically
  - Join chords with '-'

Usage:
  pip install nltk pyphen
  python generate_chord_dataset.py

Output:
  chord_dataset.csv  — columns: chords, target_word
"""

import csv
import random
import re
import string
from collections import defaultdict
from itertools import combinations

import nltk
import pyphen

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NOISE_RATE = 0.40           # fraction of examples that are noisy
KEY_SUB_PROB = 0.12         # per-chord probability of a key substitution
MISSING_LETTER_PROB = 0.10  # per-chord probability of dropping a letter
EXTRA_LETTER_PROB = 0.08    # per-chord probability of doubling a letter
BOUNDARY_SHIFT_PROB = 0.25  # per-word probability of shifting a syllable boundary
CLEAN_VARIANTS_PER_WORD = 3 # how many clean syllabification variants to keep
OUTPUT_FILE = "data/chord_dataset.csv"
MIN_WORD_LENGTH = 1
MAX_WORD_LENGTH = 20
TARGET_VOCAB_SIZE = 20000

# Keyboard adjacency map (QWERTY)
KEYBOARD_ADJACENCY = {
    'a': 'sqwz', 'b': 'vghn', 'c': 'xdfv', 'd': 'erfcs', 'e': 'wrsdf',
    'f': 'rtgdc', 'g': 'tyhfe', 'h': 'yugnj', 'i': 'uojkl', 'j': 'uihnkm',
    'k': 'iolmj', 'l': 'opk', 'm': 'nkj', 'n': 'bhjm', 'o': 'iplk',
    'p': 'ol', 'q': 'wa', 'r': 'etdf', 's': 'wedza', 't': 'ryfg',
    'u': 'yhij', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tunh',
    'z': 'asx',
}

# ---------------------------------------------------------------------------
# Syllabification helpers
# ---------------------------------------------------------------------------

dic = pyphen.Pyphen(lang='en_US')


def syllabify(word: str) -> list[str]:
    """Return the pyphen canonical syllabification of a word."""
    hyphenated = dic.inserted(word.lower())
    return hyphenated.split('-') if '-' in hyphenated else [word.lower()]


def boundary_shift_variants(syllables: list[str]) -> list[list[str]]:
    """
    Generate variants by shifting one consonant across a boundary.
    E.g. ['clas', 'si'] -> ['cla', 'ssi'] or ['class', 'i']
    """
    variants = []
    for i in range(len(syllables) - 1):
        left, right = syllables[i], syllables[i + 1]
        # shift last char of left -> prepend to right
        if len(left) > 1:
            new = syllables[:i] + [left[:-1], left[-1] + right] + syllables[i+2:]
            variants.append(new)
        # shift first char of right -> append to left
        if len(right) > 1:
            new = syllables[:i] + [left + right[0], right[1:]] + syllables[i+2:]
            variants.append(new)
    return variants


def all_syllabification_variants(word: str) -> list[list[str]]:
    """Return canonical + boundary-shifted syllabifications."""
    base = syllabify(word)
    variants = [base]
    for shifted in boundary_shift_variants(base):
        if shifted not in variants:
            variants.append(shifted)
    return variants


def letter_by_letter(word: str) -> list[str]:
    """Return each character as its own 'syllable'."""
    return list(word.lower())


# ---------------------------------------------------------------------------
# Chord encoding
# ---------------------------------------------------------------------------

def syllable_to_chord(syllable: str) -> str:
    """Remove duplicate letters and sort alphabetically."""
    return ''.join(sorted(set(syllable.lower())))


def syllables_to_chords(syllables: list[str]) -> str:
    """Convert a syllable list to a dash-joined chord string."""
    return '-'.join(syllable_to_chord(s) for s in syllables)


# ---------------------------------------------------------------------------
# Noise functions (applied BEFORE chord encoding)
# ---------------------------------------------------------------------------

def apply_key_substitution(syllable: str) -> str:
    chars = list(syllable)
    for i, ch in enumerate(chars):
        if ch in KEYBOARD_ADJACENCY and random.random() < KEY_SUB_PROB:
            chars[i] = random.choice(KEYBOARD_ADJACENCY[ch])
    return ''.join(chars)


def apply_missing_letter(syllable: str) -> str:
    if len(syllable) <= 1:
        return syllable
    if random.random() < MISSING_LETTER_PROB:
        idx = random.randrange(len(syllable))
        return syllable[:idx] + syllable[idx+1:]
    return syllable


def apply_extra_letter(syllable: str) -> str:
    if random.random() < EXTRA_LETTER_PROB:
        idx = random.randrange(len(syllable))
        return syllable[:idx] + syllable[idx] + syllable[idx:]
    return syllable


def apply_noise_to_syllables(syllables: list[str]) -> list[str]:
    """Apply all noise types to a syllable list (pre-encoding)."""
    noisy = []
    for s in syllables:
        s = apply_key_substitution(s)
        s = apply_missing_letter(s)
        s = apply_extra_letter(s)
        noisy.append(s)
    return noisy


# ---------------------------------------------------------------------------
# Variant generation per word
# ---------------------------------------------------------------------------

def generate_variants(word: str) -> list[tuple[str, str]]:
    """
    Returns a list of (chords, target_word) tuples for a word,
    including clean and noisy variants.
    """
    examples = []
    seen_chords = set()

    def add(chords: str, label: str):
        if chords not in seen_chords:
            seen_chords.add(chords)
            examples.append((chords, label))

    # 1. Letter-by-letter (always clean, always included)
    add(syllables_to_chords(letter_by_letter(word)), word)

    # 2. Clean syllabification variants (up to CLEAN_VARIANTS_PER_WORD)
    syl_variants = all_syllabification_variants(word)
    for sylls in syl_variants[:CLEAN_VARIANTS_PER_WORD]:
        add(syllables_to_chords(sylls), word)

    # 3. Noisy versions of each clean variant
    if(len(word) < 3):
        return examples  # skip noise for very short words
    for sylls in syl_variants[:CLEAN_VARIANTS_PER_WORD]:
        for _ in range(2):  # 2 noisy draws per variant
            if random.random() < NOISE_RATE:
                noisy_sylls = apply_noise_to_syllables(sylls)
                add(syllables_to_chords(noisy_sylls), word)

    return examples


# ---------------------------------------------------------------------------
# Vocabulary loading
# ---------------------------------------------------------------------------

def load_vocabulary() -> list[str]:
    from wordfreq import top_n_list
    print(f"Loading top {TARGET_VOCAB_SIZE} English words from wordfreq...")
    words = top_n_list('en', TARGET_VOCAB_SIZE * 2)  # oversample to account for filtering
    filtered = [
        w for w in words
        if w.isalpha()
        and w.isascii()
        and MIN_WORD_LENGTH <= len(w) <= MAX_WORD_LENGTH
    ]
    result = filtered[:TARGET_VOCAB_SIZE]
    print(f"Loaded {len(result)} vocabulary words.")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(42)
    vocab = load_vocabulary()

    rows = []
    skipped = 0

    for word in vocab:
        try:
            variants = generate_variants(word)
            rows.extend(variants)
        except Exception:
            skipped += 1

    # Shuffle so the CSV isn't grouped by word
    random.shuffle(rows)

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['chords', 'target_word'])
        writer.writerows(rows)

    # Summary
    total = len(rows)
    unique_words = len(set(r[1] for r in rows))
    avg_per_word = total / unique_words if unique_words else 0

    print(f"\nDataset written to: {OUTPUT_FILE}")
    print(f"  Total examples  : {total:,}")
    print(f"  Unique words    : {unique_words:,}")
    print(f"  Avg variants/wd : {avg_per_word:.1f}")
    print(f"  Skipped words   : {skipped}")

    # Preview
    print("\nSample rows:")
    for chords, word in random.sample(rows, min(10, len(rows))):
        print(f"  {chords:<40} -> {word}")


if __name__ == '__main__':
    main()