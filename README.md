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

The script uses [`wordfreq`](https://github.com/rspeer/wordfreq) to load the most common English words ranked by real-world usage frequency. This is intentional: frequency-ranked vocabulary means the model trains on words it will encounter most in practice, rather than a flat dictionary dump that includes many obscure or archaic words. The vocabulary is filtered to alphabetic ASCII words between 3 and 20 characters, then truncated to the top `TARGET_VOCAB_SIZE` entries (default: 5,000).

---

## Variant Generation

For each word, the script generates multiple chord sequences to represent the realistic diversity of how a user might type that word. There are three sources of variation.

### 1. Canonical Syllabification

The script uses `pyphen` to produce the dictionary-standard syllabification of each word. This is the baseline encoding — the "correct" way to chord the word.

### 2. Boundary-Shifted Variants

Users don't always split syllables at dictionary boundaries. Someone might chord `classification` as `cla-ssif-ic-ation` rather than `clas-si-fi-ca-tion`. The script generates these alternatives by shifting one consonant left or right across each syllable boundary. For example:

- Canonical: `['clas', 'si', 'fi', 'ca', 'tion']`
- Shifted: `['cla', 'ssi', 'fi', 'ca', 'tion']` or `['class', 'i', 'fi', 'ca', 'tion']`

Up to `CLEAN_VARIANTS_PER_WORD` (default: 3) clean variants are kept per word, to avoid combinatorial explosion while still giving the model exposure to plausible idiosyncratic splitting habits.

### 3. Letter-by-Letter Spelling

Every word always includes a fully spelled-out variant where each letter is its own chord: `c-l-a-s-s-i-f-i-c-a-t-i-o-n`. This is a deliberate fallback behavior — users reach for it when a word is hard to chunk or unfamiliar. Because it's intentional and deterministic, it is always included clean (never noisified).

---

## Noise Injection

Roughly `NOISE_RATE` (default: 40%) of training examples are noisy. Noise is injected **before** the chord normalization step — that is, noise is applied to the raw syllable strings, and then the corrupted syllables are encoded into chords. This correctly models what actually happens: the user mistypes, and the chord normalization runs on whatever they produced.

Four noise types are applied:

### Key Substitution
With probability `KEY_SUB_PROB` (default: 12%) per chord, one letter in the syllable is replaced with an adjacent key on a QWERTY keyboard. For example, `a` might become `q`, `s`, `w`, or `z`. This models the most common real-world typing error.

### Missing Letter
With probability `MISSING_LETTER_PROB` (default: 10%) per chord, one letter is dropped from the syllable. This models the case where a user forgets a letter in a chunk, e.g. typing `clas` as `cls`.

### Extra/Doubled Letter
With probability `EXTRA_LETTER_PROB` (default: 8%) per chord, one letter is duplicated. This models an accidental double-press or a user over-emphasizing a sound. Note that because chord normalization deduplicates letters, a doubled letter often has no effect on the final chord — this noise only matters when the doubled letter was not already present, making it a relatively subtle perturbation.

### Syllable Boundary Shift (as noise)
In addition to using boundary shifts as clean variants, the shift operation can also act as noise — modeling the case where a user intends one split but executes another. This is already embedded in the variant generation: noisy draws are taken from all clean variants, including shifted ones.

---

## Dataset Composition

For each word, the script generates:

- 1 letter-by-letter example (always clean)
- Up to 3 clean syllabification variants (canonical + boundary shifts)
- Up to 6 additional noisy draws (2 per clean variant, each independently gated by `NOISE_RATE`)

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

| Parameter | Default | Description |
|---|---|---|
| `TARGET_VOCAB_SIZE` | 5000 | Number of words to include |
| `NOISE_RATE` | 0.40 | Fraction of examples that receive noise |
| `KEY_SUB_PROB` | 0.12 | Per-chord probability of a key substitution |
| `MISSING_LETTER_PROB` | 0.10 | Per-chord probability of dropping a letter |
| `EXTRA_LETTER_PROB` | 0.08 | Per-chord probability of doubling a letter |
| `CLEAN_VARIANTS_PER_WORD` | 3 | Max clean syllabification variants per word |
| `OUTPUT_FILE` | `chord_dataset.csv` | Output filename |

---

## Dependencies

```
pip install wordfreq pyphen
```
