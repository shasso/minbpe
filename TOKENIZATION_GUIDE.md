# Understanding Tokenizer Output: A Guide

This guide helps you understand how tokenizers work and how to debug tokenization issues, particularly when working with different scripts and languages.

## Table of Contents
1. [BPE vs SentencePiece](#bpe-vs-sentencepiece)
2. [Understanding Tokenizer Output](#understanding-tokenizer-output)
3. [Debugging Tokenization Issues](#debugging-tokenization-issues)
4. [Using the Visualization Tool](#using-the-visualization-tool)
5. [Working with Non-Latin Scripts](#working-with-non-latin-scripts)

## BPE vs SentencePiece

### Byte Pair Encoding (BPE) - Used in minbpe

**How it works:**
- Operates on **UTF-8 encoded bytes**
- Starts with 256 base tokens (one for each byte value 0-255)
- Learns merges by finding the most frequent byte pairs
- The final vocabulary contains both individual bytes and merged tokens

**Example:**
```python
from minbpe import BasicTokenizer

tokenizer = BasicTokenizer()
tokenizer.train("hello hello hello", 256 + 3)
# Creates tokens for common byte pairs like "he", "ll", "lo"
```

**Advantages:**
- Always lossless (can encode any byte sequence)
- No out-of-vocabulary tokens
- Works well with any language/script

**Disadvantages:**
- Can create many tokens for languages with large character sets
- Token boundaries don't always align with character boundaries

### SentencePiece (Not implemented in minbpe)

**How it works:**
- Operates on **Unicode code points** directly
- Can use BPE or Unigram language model
- Uses special markers like `▁` (U+2581) to represent spaces
- Treats spaces as regular tokens

**Example (conceptual):**
```
Input: "hello world"
SentencePiece output: ["▁hello", "▁world"]
or with more merges: ["▁hel", "lo", "▁wor", "ld"]
```

**Key differences:**
1. **Space handling**: SentencePiece uses `▁` to mark word boundaries
2. **Unicode**: Works on code points, not bytes
3. **No guaranteed lossless encoding**: May have unknown tokens

## Understanding Tokenizer Output

### Reading minbpe Output

When you encode text with minbpe:

```python
from minbpe import BasicTokenizer

tokenizer = BasicTokenizer()
text = "hello"
token_ids = tokenizer.encode(text)
# Output: [104, 101, 108, 108, 111]
# These are the ASCII values of 'h', 'e', 'l', 'l', 'o'
```

**Token ID meanings:**
- **0-255**: Individual bytes (base vocabulary)
- **256+**: Merged tokens learned during training

### Reading SentencePiece Output

SentencePiece output includes the `▁` character (not underscore!) to mark spaces:

```
Input: "hello world"
Output: ["▁hello", "▁world"]
```

The `▁` character (U+2581 LOWER ONE EIGHTH BLOCK) indicates that this token appeared at the start of a word.

## Debugging Tokenization Issues

### Common Issues

#### 1. "My tokenizer output doesn't match my input"

**Possible causes:**
- You're comparing encoded token IDs with expected text
- The tokenizer was trained on different data
- Character encoding issues (UTF-8 vs other encodings)

**Solution:** Use the visualization tool to see what each token represents:
```bash
python visualize_tokenizer.py --model your_model.model --text "your text"
```

#### 2. "I see � (replacement character) in the output"

**Cause:** Individual tokens may be partial UTF-8 byte sequences that don't form valid Unicode characters.

**This is normal!** BPE works on bytes, not characters. The full sequence decodes correctly even if individual tokens don't.

**Example:**
```
Korean "안녕" in UTF-8: ec 95 88 eb 85 95
Token 236 (0xec): � (invalid alone)
Token 149 (0x95): � (invalid alone)  
Token 136 (0x88): � (invalid alone)
Together: 안 (valid!)
```

#### 3. "My tokens don't align with words"

**Cause:** BPE doesn't understand linguistic boundaries. It only knows byte frequencies.

**This is expected!** Tokens represent common byte patterns, not semantic units.

## Using the Visualization Tool

The `visualize_tokenizer.py` script helps you understand how your tokenizer processes text.

### Basic Usage

```bash
# Visualize how text is tokenized (untrained tokenizer)
python visualize_tokenizer.py --text "hello world"

# Use a trained model
python visualize_tokenizer.py --model models/basic.model --text "hello world"

# Show byte-level details
python visualize_tokenizer.py --text "hello" --bytes

# Compare different texts
python visualize_tokenizer.py --model models/basic.model --compare "hello" "world" "hello world"

# Analyze token distribution
python visualize_tokenizer.py --model models/basic.model --analyze "your long text here"

# Explain a specific token
python visualize_tokenizer.py --model models/basic.model --explain-token 258
```

### Programmatic Usage

```python
from minbpe import BasicTokenizer
from visualize_tokenizer import TokenizerVisualizer

# Create and train a tokenizer
tokenizer = BasicTokenizer()
tokenizer.train("your training text", vocab_size=512)

# Create visualizer
visualizer = TokenizerVisualizer(tokenizer)

# Visualize encoding
visualizer.visualize_encoding("test text")

# Analyze token distribution
visualizer.analyze_token_distribution("test text")

# Compare texts
visualizer.compare_texts(["text 1", "text 2", "text 3"])

# Explain a token
visualizer.explain_token(258)
```

## Working with Non-Latin Scripts

### Understanding Multi-Byte Characters

Non-Latin scripts (Arabic, Hebrew, CJK, etc.) use multiple bytes per character in UTF-8:

| Script | Example | Bytes per char |
|--------|---------|----------------|
| English | "a" | 1 byte |
| Hebrew | "א" | 2 bytes |
| Arabic | "ا" | 2 bytes |
| Syriac | "ܐ" | 3 bytes |
| CJK | "中" | 3 bytes |
| Emoji | "😊" | 4 bytes |

### Example: Syriac Text

Let's say you have Syriac text like the user's example:
```
ܐܲܒ݂ܪܵܗܵܡ ܡܘܼܠܸܕ
```

When tokenized with an **untrained** BPE tokenizer:
- Each UTF-8 byte becomes a separate token
- Syriac characters use 3 bytes each
- Diacritics add more bytes
- Result: Many tokens per character

With a **trained** tokenizer on Syriac text:
- Common byte sequences merge into single tokens
- Might learn tokens for common letter combinations
- More efficient encoding

### Training Tips for Non-Latin Scripts

1. **Train on your target language**: A tokenizer trained on English won't efficiently encode Syriac
2. **Use sufficient vocabulary size**: Complex scripts need more tokens
3. **Consider RegexTokenizer**: Splits by script type for better handling

```python
from minbpe import RegexTokenizer

tokenizer = RegexTokenizer()
# Train on your Syriac corpus
tokenizer.train(syriac_text, vocab_size=4096)
# This will learn common Syriac patterns
```

## Comparing with SentencePiece Output

If you're comparing minbpe output with SentencePiece:

**Key differences to remember:**

1. **Space markers**: SentencePiece uses `▁`, minbpe encodes space as byte 32
2. **Token IDs**: Different vocabularies = different IDs for same text
3. **Byte vs Code Point**: minbpe works on bytes, SentencePiece on Unicode
4. **Training data**: Different training creates different merges

**Why your output might not match:**

If you trained a SentencePiece model and are looking at its output:
- The encoded pieces are **token strings**, not the original text
- They represent the tokenizer's learned vocabulary
- The `▁` markers show word boundaries
- They may not align with the input text character-by-character

**To understand SentencePiece output:**
1. The output shows learned tokens from the vocabulary
2. Decode the tokens back to text to verify correctness
3. Use SentencePiece's own visualization tools

## Example Workflow: Debugging Tokenization

Let's walk through debugging a tokenization issue:

### Problem
"I encoded this text but the tokens don't look right"

### Step 1: Visualize
```bash
python visualize_tokenizer.py --model models/basic.model --text "your text"
```

Look at:
- Are the token IDs reasonable?
- Does it decode back to the original text?
- How many tokens per character?

### Step 2: Check Individual Tokens
```bash
python visualize_tokenizer.py --model models/basic.model --explain-token 258
```

Understand what each token represents.

### Step 3: Compare with Expected
```python
from visualize_tokenizer import TokenizerVisualizer
from minbpe import BasicTokenizer

tokenizer = BasicTokenizer()
tokenizer.load("models/basic.model")

# Check if encoding is lossless
text = "your text"
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)
assert text == decoded, "Encoding is lossy!"
```

### Step 4: Analyze Distribution
```bash
python visualize_tokenizer.py --model models/basic.model --analyze "your long text"
```

Check if tokens are being used efficiently.

## Additional Resources

- **minbpe README**: `/README.md` - Overview of the tokenizers
- **Training script**: `train.py` - Example of training tokenizers
- **Tests**: `tests/test_tokenizer.py` - More usage examples
- **SentencePiece**: https://github.com/google/sentencepiece - Original SentencePiece implementation

## Summary

- **minbpe** implements byte-level BPE (works on UTF-8 bytes)
- **SentencePiece** implements Unicode-level BPE (works on code points)
- Use the visualization tool to understand how text is tokenized
- Individual tokens may not be valid Unicode characters (but the full sequence is)
- Train on your target language for efficient tokenization
- Always verify encode/decode is lossless for your use case
