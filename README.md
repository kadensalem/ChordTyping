# Chord-to-Word Dataset Generator

This script generates a synthetic training dataset for a chord-based word prediction model — similar in spirit to stenography. The model's goal is to learn that many different chord sequences can resolve to the same target word.

---

## What Is a Chord Sequence?

A chord sequence is a structured, compressed representation of a word. To produce one:

1. **Split** the word into syllables: `classification` → `class-i-fi-ca-tion`
2. **Normalize** each syllable by removing duplicate letters and sorting alphabetically: `class` → `acls`, `tion` → `inot`
3. **Join** with dashes: `acls-i-fi-ac-inot`

This encoding is intentionally lossy — `at` and `ta` produce the same chord `at`. The model must learn to resolve these ambiguities using the context of the full chord sequence.

---

## Vocabulary Source

The script uses [`wordfreq`](https://github.com/rspeer/wordfreq) to load the most common English words ranked by real-world usage frequency. This is intentional: frequency-ranked vocabulary means the model trains on words it will encounter most in practice, rather than a flat dictionary dump that includes many obscure or archaic words. The vocabulary is filtered to alphabetic ASCII words within the configured length bounds, then truncated to the top `TARGET_VOCAB_SIZE` entries.

---

## Variant Generation

For each word, the script generates multiple chord sequences to represent the realistic diversity of how a user might type that word. There are three sources of variation.

### 1. Canonical Syllabification

The script uses `pyphen` to produce the dictionary-standard syllabification of each word. This is the baseline encoding — the "correct" way to chord the word.

### 2. Boundary-Shifted Variants

Users don't always split syllables at dictionary boundaries. Someone might chord `classification` as `cla-ssif-ic-ation` rather than `clas-si-fi-ca-tion`. The script generates these alternatives by shifting one consonant left or right across each syllable boundary. For example:

- Canonical: `['clas', 'si', 'fi', 'ca', 'tion']`
- Shifted: `['cla', 'ssi', 'fi', 'ca', 'tion']` or `['class', 'i', 'fi', 'ca', 'tion']`

Up to `CLEAN_VARIANTS_PER_WORD` clean variants are kept per word, to avoid combinatorial explosion while still giving the model exposure to plausible idiosyncratic splitting habits.

### 3. Letter-by-Letter Spelling

Every word always includes a fully spelled-out variant where each letter is its own chord: `c-l-a-s-s-i-f-i-c-a-t-i-o-n`. This is a deliberate fallback behavior — users reach for it when a word is hard to chunk or unfamiliar. Because it's intentional and deterministic, it is always included clean (never noisified).

---

## Noise Injection

A fraction of training examples (controlled by `NOISE_RATE`) are noisy. Noise is injected **before** the chord normalization step — that is, noise is applied to the raw syllable strings, and then the corrupted syllables are encoded into chords. This correctly models what actually happens: the user mistypes, and the chord normalization runs on whatever they produced.

For each clean variant, `NOISY_DRAWS_PER_VARIANT` noisy versions are drawn, each independently gated by `NOISE_RATE`. Four noise types are applied:

### Key Substitution
With probability `KEY_SUB_PROB` per chord, one letter in the syllable is replaced with an adjacent key on a QWERTY keyboard. For example, `a` might become `q`, `s`, `w`, or `z`. This models the most common real-world typing error.

### Missing Letter
With probability `MISSING_LETTER_PROB` per chord, one letter is dropped from the syllable. This models the case where a user forgets a letter in a chunk, e.g. typing `clas` as `cls`.

### Extra/Doubled Letter
With probability `EXTRA_LETTER_PROB` per chord, one letter is duplicated. This models an accidental double-press or a user over-emphasizing a sound. Note that because chord normalization deduplicates letters, a doubled letter often has no effect on the final chord — this noise only matters when the doubled letter was not already present, making it a relatively subtle perturbation.

### Random Oversplitting
With probability `SYLLABLE_SPLIT_PROB` per variant, one syllable is randomly split into two at an interior position. This models the case where a user intends one chunk but breaks it into two, independently of the boundary-shift variants.

### Note on Small Words
Small words (fewer than 3 letters) are skipped for noise injection to help strengthen their signal in training.

---

## Dataset Composition

For each word, the script generates:

- 1 letter-by-letter example (always clean)
- Up to `CLEAN_VARIANTS_PER_WORD` clean syllabification variants (canonical + boundary shifts)
- Up to `NOISY_DRAWS_PER_VARIANT` noisy draws per clean variant, each gated by `NOISE_RATE`

Duplicate chord sequences for the same word are deduplicated — if two variants happen to produce the same chord string, only one is kept. The final CSV is shuffled so examples are not grouped by word.

---

## Output Format

The output is a CSV file with two columns:

| Column | Description |
|---|---|
| `chords` | Dash-separated chord sequence, e.g. `acls-i-fi-ac-inot` |
| `target_word` | The word the chord sequence should resolve to, e.g. `classification` |

---

## Configuration

All key parameters are defined at the top of the script:

| Parameter | Description |
|---|---|
| `TARGET_VOCAB_SIZE` | Number of words to include |
| `MIN_WORD_LENGTH` | Minimum word length (in characters) to include |
| `MAX_WORD_LENGTH` | Maximum word length (in characters) to include |
| `NOISE_RATE` | Fraction of noisy draw attempts that are actually kept |
| `KEY_SUB_PROB` | Per-chord probability of replacing a letter with an adjacent QWERTY key |
| `MISSING_LETTER_PROB` | Per-chord probability of dropping a letter from a syllable |
| `EXTRA_LETTER_PROB` | Per-chord probability of duplicating a letter in a syllable |
| `SYLLABLE_SPLIT_PROB` | Per-variant probability of randomly splitting a syllable into two before noise |
| `CLEAN_VARIANTS_PER_WORD` | Max number of clean syllabification variants to generate per word |
| `NOISY_DRAWS_PER_VARIANT` | Number of noisy versions to attempt drawing per clean variant |
| `OUTPUT_FILE` | Output file path |

---

## Training

The dataset is consumed by a character-level BiLSTM classifier (`train.py`) that predicts a target word from a chord string.

### Model Architecture

Each chord string is fed into the model as a sequence of individual characters. Characters are embedded and passed through a bidirectional LSTM — the final forward and backward hidden states are concatenated and passed to a linear classifier over the full word vocabulary.

### Train / Validation Split

Rather than a random split, examples are grouped by word and split per-word — a fixed fraction of each word's variants go to validation, the rest to training. This ensures every word in the vocabulary is represented in both splits, and that validation measures generalization across chord variants rather than unseen words.

### Checkpointing

Only the best model by validation accuracy is saved. The checkpoint includes the model weights, the character and word vocabularies, per-word frequency data from `wordfreq`, and the model config — everything needed to run inference without keeping the original CSV around.

### Key Training Details

- **Optimizer**: Adam with a `ReduceLROnPlateau` scheduler that halves the learning rate after 3 epochs of no validation loss improvement
- **Loss**: Cross-entropy with label smoothing, which helps given that many chord sequences are ambiguous by design
- **Gradient clipping**: Applied at norm 1.0 to stabilize training

---

## Dependencies

For data generation:
```
pip install wordfreq pyphen
```

For training:
```
pip install torch pandas
```