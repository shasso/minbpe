# Understanding SentencePiece Tokenizer Output

This document specifically addresses questions about SentencePiece tokenizer output, particularly when the encoded pieces don't seem to match the input text.

## The Problem

You trained a SentencePiece tokenizer on your corpus and when you encode text like this:

```
ܐܲܒ݂ܪܵܗܵܡ ܡܘܼܠܸܕ ܠܹܗ ܠܐܝܼܣܚܵܩ
```

You get encoded pieces like:

```python
['▁ܐܵܗܵܐ', '▁ܝܠܹܗ', '▁ܟܬܵܒ݂ܵܐ', '▁ܕܝܼ', 'ܠܝܼ', 'ܕܘܼܬܵܐ', ...]
```

And you're confused because:
1. These encoded pieces are different text than your input
2. Some pieces seem to be full words
3. There are `▁` characters everywhere

## Understanding What Happened

### Key Insight #1: You're looking at the VOCABULARY tokens, not a representation of your input

When SentencePiece shows you encoded pieces like `['▁ܐܵܗܵܐ', '▁ܝܠܹܗ', ...]`, it's showing you **which tokens from the vocabulary were used**, not **how your input text was represented**.

Think of it like this:
- Your input text: "ܐܲܒ݂ܪܵܗܵܡ"
- SentencePiece vocabulary: A list of 8000 or 32000 learned tokens
- Encoding result: [token_15, token_234, token_567, ...] → shown as the token strings from vocabulary

### Key Insight #2: The vocabulary was learned from your training corpus

The token strings you see (`'▁ܐܵܗܵܐ'`, `'▁ܝܠܹܗ'`, etc.) are pieces of text that appeared frequently in your **training corpus**. They may or may not appear in the specific text you're encoding now.

### Key Insight #3: The ▁ character marks word boundaries

In SentencePiece:
- `▁` (U+2581, not an underscore!) marks the beginning of a word
- It replaces the space character
- This allows the tokenizer to know where words start

Example:
```
Input:     "hello world"
SentencePiece: ["▁hello", "▁world"]
                 ↑         ↑
                 word      word
                 start     start
```

## What You Should Check

### 1. Can you decode back to the original text?

This is the most important test:

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('your_model.model')

# Your input text
text = "ܐܲܒ݂ܪܵܗܵܡ ܡܘܼܠܸܕ ܠܹܗ"

# Encode
pieces = sp.encode_as_pieces(text)
print("Encoded pieces:", pieces)

# Decode
decoded = sp.decode_pieces(pieces)
print("Decoded text:", decoded)

# Check if it matches
assert text == decoded, "Encoding is lossy!"
```

If `text == decoded`, then your tokenizer is working correctly, even if the pieces look strange.

### 2. Understanding the encoded pieces

The pieces you see are **subword units** learned from your training data. They might be:
- Complete words that appeared frequently in training
- Parts of words (prefixes, suffixes, roots)
- Single characters
- Character combinations

For your Syriac text, if you trained on a different Syriac corpus, the learned tokens will reflect the word patterns in that corpus, not necessarily the text you're encoding now.

## Example Walkthrough

Let's trace through what's happening:

```python
# You have:
input_text = "ܐܲܒ݂ܪܵܗܵܡ"

# SentencePiece encodes it as:
pieces = ['▁ܐܲܒ݂', 'ܪܵܗܵܡ']  # hypothetical example

# What this means:
# Token 1: '▁ܐܲܒ݂' - A token that starts a word and contains these characters
#          This token was learned from your training data
# Token 2: 'ܪܵܗܵܡ' - A token (not word-initial) with these characters
#          This token was also learned from your training data

# When you decode:
decoded = sp.decode_pieces(pieces)
# Result: "ܐܲܒ݂ܪܵܗܵܡ" - same as input!
```

The tokens used to encode text depend on what was learned during training, not on the input text itself.

## Why Are the Pieces Different from My Input?

You encoded:
```
"ܐܲܒ݂ܪܵܗܵܡ ܡܘܼܠܸܕ ܠܹܗ ܠܐܝܼܣܚܵܩ"
```

But got pieces like:
```
['▁ܐܵܗܵܐ', '▁ܝܠܹܗ', '▁ܟܬܵܒ݂ܵܐ', ...]
```

**This is wrong.** The pieces should, when decoded, reconstruct your input text exactly.

Possible issues:
1. **You're looking at a different text's encoding**: Double-check that the pieces you're seeing actually correspond to the text you think you encoded
2. **You're looking at the vocabulary, not the encoding**: Make sure you're calling `encode_as_pieces()` not just inspecting the model's vocabulary
3. **There's a mismatch in your code**: Verify your encoding/decoding pipeline

## Debugging Your Tokenizer

### Step 1: Verify encode/decode works

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('your_model.model')

test_text = "ܐܲܒ݂ܪܵܗܵܡ"
pieces = sp.encode_as_pieces(test_text)
decoded = sp.decode_pieces(pieces)

print(f"Input:   '{test_text}'")
print(f"Pieces:  {pieces}")
print(f"Decoded: '{decoded}'")
print(f"Match:   {test_text == decoded}")
```

### Step 2: Check individual tokens

```python
# See what each token in your vocabulary looks like
for i in range(min(100, sp.get_piece_size())):
    print(f"Token {i}: '{sp.id_to_piece(i)}'")
```

### Step 3: Verify with a known example

```python
# Try a simple example
simple = "ܗܠܠ"  # or any word you know
pieces = sp.encode_as_pieces(simple)
decoded = sp.decode_pieces(pieces)
print(f"'{simple}' -> {pieces} -> '{decoded}'")
```

## Understanding minbpe vs SentencePiece

This repository (minbpe) implements **byte-level BPE**, which is different from SentencePiece:

| Feature | minbpe (BPE) | SentencePiece |
|---------|--------------|---------------|
| Works on | UTF-8 bytes | Unicode code points |
| Space handling | Byte 32 | ▁ character |
| Output format | Token IDs [104, 101, ...] | Token strings ['▁hel', 'lo'] |
| Token boundaries | Byte pairs | Code point pairs |

If you want to understand how minbpe works with your Syriac text:

```bash
# See how untrained tokenizer handles Syriac
python visualize_tokenizer.py --text "ܐܲܒ݂ܪܵܗܵܡ" --bytes

# Train on Syriac corpus and see improved tokenization
python train_on_syriac.py  # you'd need to create this
python visualize_tokenizer.py --model models/syriac.model --text "ܐܲܒ݂ܪܵܗܵܡ"
```

## Summary

- **Encoded pieces are vocabulary tokens**, not your input text
- **The ▁ character** marks word boundaries (space replacement)
- **Verify encoding is lossless**: decode(encode(text)) == text
- **Different tools, different formats**: 
  - SentencePiece shows token strings
  - minbpe shows token IDs
- **Vocabulary reflects training data**, not test data

If your encoding/decoding is lossless, your tokenizer is working correctly, even if the pieces look confusing.

## Need More Help?

1. Read [TOKENIZATION_GUIDE.md](TOKENIZATION_GUIDE.md) for comprehensive documentation
2. Run `python examples_tokenization.py` to see interactive examples
3. Use `python visualize_tokenizer.py` to debug your tokenization (works with minbpe tokenizers)
4. Check SentencePiece documentation for SentencePiece-specific features
