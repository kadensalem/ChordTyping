import re
from wordfreq import top_n_list
import random
import pyphen

QWERTY_NEIGHBORS = {
    'q': 'wa',       'w': 'qeasd',    'e': 'wrsdf',    'r': 'etdfg',
    't': 'ryfgh',    'y': 'tughj',    'u': 'yihjk',    'i': 'uojkl',
    'o': 'ipkl',     'p': 'ol',
    'a': 'qwsz',     's': 'awedxz',   'd': 'serfcx',   'f': 'drtgvc',
    'g': 'ftyhbv',   'h': 'gyujnb',   'j': 'huikmn',   'k': 'jiolm',
    'l': 'kop',
    'z': 'asx',      'x': 'zsdc',     'c': 'xdfv',     'v': 'cfgb',
    'b': 'vghn',     'n': 'bhjm',     'm': 'njk',
}

VOWELS = set('aeiou')

# ── Legitimate single-letter words ────────────────────────────────────────────
SINGLE_LETTER_WORDS = {'a', 'i'}


# ── Vocab generation ──────────────────────────────────────────────────────────
def build_vocab(n_target=20000, lang="en"):
    """Build a vocabulary of the top n_target words in the specified language."""
    raw = top_n_list(lang, n_target * 3)
    vocab = []
    seen = set()

    for w in raw:
        w = w.lower().strip()
        if not re.fullmatch(r"[a-z]+(?:['-][a-z]+)*", w):
            continue
        if w in seen:
            continue
        seen.add(w)
        vocab.append(w)
        if len(vocab) >= n_target:
            break

    return vocab


# ── Helpers ───────────────────────────────────────────────────────────────────
def chord_char_count(chord: str) -> int:
    """Count characters in a chord, excluding hyphens."""
    return len(chord.replace("-", ""))


def is_valid_chord(chord: str, word: str) -> bool:
    """
    Reject chords with fewer than 2 non-hyphen characters,
    unless the word itself is a legitimate single-letter word.
    """
    if word in SINGLE_LETTER_WORDS:
        return True
    return chord_char_count(chord) >= 2


def syllable_chord(syllable: str) -> str:
    """Return sorted unique characters of a syllable."""
    return "".join(sorted(set(syllable)))


# ── Structural variants (always generated) ────────────────────────────────────
def gen_golden_chord(syllables: list) -> str:
    """Sorted unique chars per syllable, joined by hyphens."""
    return "-".join(syllable_chord(s) for s in syllables)


def gen_consonant_chord(syllables: list) -> str:
    """Sorted unique consonants per syllable, joined by hyphens.

    For a syllable that is a lone vowel within a single-syllable word,
    the vowel is preserved so the chord is not empty.
    """
    chords = []
    for syllable in syllables:
        # Single-syllable word whose only syllable is a lone vowel
        if len(syllable) == 1 and len(syllables) == 1 and syllable[0] in VOWELS:
            chords.append(syllable[0])
        else:
            consonants = "".join(sorted(set(c for c in syllable if c not in VOWELS)))
            # Vowel-only syllable: preserve its first vowel to avoid dropping the slot
            chords.append(consonants if consonants else syllable[0])
    return "-".join(chords)


# ── Error variants (generated with probability 0.3 each) ─────────────────────
def gen_delete_chord(syllables: list) -> str:
    """Remove one random character from one eligible syllable chord."""
    golden_chords = [syllable_chord(s) for s in syllables]

    # Only consider syllables with at least 3 unique chars
    eligible = [i for i, chord in enumerate(golden_chords) if len(chord) >= 3]
    if not eligible:
        return "-".join(golden_chords)

    i = random.choice(eligible)
    chord = golden_chords[i]
    j = random.randint(0, len(chord) - 1)
    golden_chords[i] = chord[:j] + chord[j + 1:]

    return "-".join(golden_chords)


def gen_mistype_chord(syllables: list) -> str:
    """Replace one character in one eligible syllable chord with a QWERTY neighbor."""
    golden_chords = [syllable_chord(s) for s in syllables]

    # Only consider syllables with at least 3 unique chars
    eligible = [i for i, chord in enumerate(golden_chords) if len(chord) >= 3]
    if not eligible:
        return "-".join(golden_chords)

    i = random.choice(eligible)
    chord = golden_chords[i]

    # Only consider characters that have QWERTY neighbors
    mistype_candidates = [j for j, ch in enumerate(chord) if ch in QWERTY_NEIGHBORS]
    if not mistype_candidates:
        return "-".join(golden_chords)

    j = random.choice(mistype_candidates)
    neighbor = random.choice(QWERTY_NEIGHBORS[chord[j]])
    new_chord = chord[:j] + neighbor + chord[j + 1:]
    golden_chords[i] = "".join(sorted(set(new_chord)))

    return "-".join(golden_chords)


# ── Per-word variant assembly ─────────────────────────────────────────────────
def generate_variants(word: str, syllables: list) -> list:
    """
    Return a deduplicated, validity-filtered list of chord variants for a word.

    Structural variants are always included (if valid).
    Error variants are each included independently with probability 0.3.
    """
    variants: set[str] = set()

    # ── Structural (always) ───────────────────────────────────────────────────
    golden = gen_golden_chord(syllables)
    if is_valid_chord(golden, word):
        variants.add(golden)

    consonant = gen_consonant_chord(syllables)
    if consonant and is_valid_chord(consonant, word):
        variants.add(consonant)

    # ── Error (probabilistic) ─────────────────────────────────────────────────
    if random.random() < 0.3:
        delete = gen_delete_chord(syllables)
        if is_valid_chord(delete, word):
            variants.add(delete)

    if random.random() < 0.3:
        mistype = gen_mistype_chord(syllables)
        if is_valid_chord(mistype, word):
            variants.add(mistype)

    # Return as a stable list (sorted for determinism within a run)
    return sorted(variants)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    vocab = build_vocab()
    dic = pyphen.Pyphen(lang="en")

    with open("data/data.txt", "w", encoding="utf-8") as f:
        for word in vocab:
            # Split into syllables, strip apostrophes, drop empties
            syllables = dic.inserted(word, hyphen="|").split("|")
            syllables = [s.replace("'", "") for s in syllables if s.replace("'", "")]

            if not syllables:
                continue

            variants = generate_variants(word, syllables)

            # Skip words that produced no valid variants at all
            if not variants:
                continue

            f.write(f"{word} {' '.join(variants)}\n")


if __name__ == "__main__":
    main()