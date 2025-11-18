"""
Understanding Tokenizer Output - Example Script

This script demonstrates how to understand tokenizer output,
particularly useful when you're confused about what tokens represent.

This addresses common questions like:
- "Why don't my encoded tokens match my input text?"
- "What do these token IDs mean?"
- "How do I debug tokenization issues?"
"""

from minbpe import BasicTokenizer, RegexTokenizer
from visualize_tokenizer import TokenizerVisualizer


def example_1_basic_encoding():
    """
    Example 1: Understanding basic encoding
    Shows how text is encoded into token IDs and what each ID represents
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Encoding")
    print("="*80)
    
    tokenizer = BasicTokenizer()
    text = "hello world"
    
    print(f"\nInput text: '{text}'")
    
    # Encode
    token_ids = tokenizer.encode(text)
    print(f"Token IDs: {token_ids}")
    
    # What each token represents
    print(f"\nWhat each token represents:")
    for token_id in token_ids:
        token_bytes = tokenizer.vocab[token_id]
        token_str = token_bytes.decode('utf-8', errors='replace')
        print(f"  Token {token_id}: '{token_str}' (byte: {token_bytes[0] if len(token_bytes) == 1 else token_bytes})")
    
    # Decode back
    decoded = tokenizer.decode(token_ids)
    print(f"\nDecoded text: '{decoded}'")
    print(f"Match: {text == decoded}")


def example_2_trained_tokenizer():
    """
    Example 2: Understanding a trained tokenizer
    Shows how tokens change after training on a corpus
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Trained Tokenizer")
    print("="*80)
    
    # Train on some text
    training_text = "hello hello world world hello world " * 10
    tokenizer = BasicTokenizer()
    tokenizer.train(training_text, vocab_size=256 + 10, verbose=False)
    
    text = "hello world"
    print(f"\nInput text: '{text}'")
    
    # Encode with trained tokenizer
    token_ids = tokenizer.encode(text)
    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(token_ids)} (vs {len(text)} bytes for untrained)")
    
    print(f"\nWhat each token represents:")
    for token_id in token_ids:
        token_bytes = tokenizer.vocab[token_id]
        token_str = token_bytes.decode('utf-8', errors='replace')
        
        # Check if it's a merged token
        if token_id >= 256:
            print(f"  Token {token_id}: '{token_str}' (MERGED TOKEN)")
        else:
            print(f"  Token {token_id}: '{token_str}' (base byte)")
    
    # Decode back
    decoded = tokenizer.decode(token_ids)
    print(f"\nDecoded text: '{decoded}'")
    print(f"Match: {text == decoded}")


def example_3_non_latin_script():
    """
    Example 3: Understanding non-Latin scripts
    Shows how multi-byte characters are handled
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Non-Latin Scripts (Korean)")
    print("="*80)
    
    tokenizer = BasicTokenizer()
    text = "안녕하세요"  # "Hello" in Korean
    
    print(f"\nInput text: '{text}'")
    print(f"Number of characters: {len(text)}")
    print(f"Number of UTF-8 bytes: {len(text.encode('utf-8'))}")
    
    # Encode
    token_ids = tokenizer.encode(text)
    print(f"\nToken IDs: {token_ids}")
    print(f"Number of tokens: {len(token_ids)}")
    
    print(f"\nKey insight: Each Korean character uses 3 UTF-8 bytes")
    print(f"Without training, each byte becomes a separate token")
    
    # Show bytes
    print(f"\nUTF-8 bytes of '{text}':")
    for i, byte in enumerate(text.encode('utf-8')):
        print(f"  Byte {i}: 0x{byte:02x} (decimal {byte})")
    
    # Decode back
    decoded = tokenizer.decode(token_ids)
    print(f"\nDecoded text: '{decoded}'")
    print(f"Match: {text == decoded}")
    
    print(f"\n💡 Note: Individual tokens show as '�' but together they form valid text!")


def example_4_comparing_outputs():
    """
    Example 4: Why encoded output doesn't match input
    Clarifies the difference between token IDs and token strings
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Understanding Token IDs vs Token Strings")
    print("="*80)
    
    print("""
Common confusion: "I encoded text and got [104, 101, 108, 108, 111]
but I expected to see the actual text!"

Explanation:
- encode() returns TOKEN IDS (integers), not text
- Token IDs are indices into the tokenizer's vocabulary
- To see what each ID represents, look it up in the vocabulary
- To convert back to text, use decode()
""")
    
    tokenizer = BasicTokenizer()
    text = "hello"
    
    print(f"\nInput text: '{text}'")
    
    # Encode
    token_ids = tokenizer.encode(text)
    print(f"\nStep 1 - Encode:")
    print(f"  Result: {token_ids}")
    print(f"  These are TOKEN IDs, not the text itself!")
    
    # Look up what each token represents
    print(f"\nStep 2 - Look up each token ID in vocabulary:")
    for token_id in token_ids:
        token_bytes = tokenizer.vocab[token_id]
        token_str = token_bytes.decode('utf-8', errors='replace')
        print(f"  Token ID {token_id} → '{token_str}'")
    
    # Decode
    decoded = tokenizer.decode(token_ids)
    print(f"\nStep 3 - Decode:")
    print(f"  Result: '{decoded}'")
    print(f"  Now we have the text back!")


def example_5_sentencepiece_vs_bpe():
    """
    Example 5: Understanding SentencePiece vs BPE output
    Explains the key differences
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: SentencePiece vs BPE (minbpe)")
    print("="*80)
    
    print("""
SentencePiece Output Example:
  Input:  "hello world"
  Output: ["▁hello", "▁world"]
  
  The ▁ character marks word boundaries (it's not an underscore!)
  The output shows TOKEN STRINGS from the vocabulary

minbpe (BPE) Output:
  Input:  "hello world"
  Output: [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
  
  The output shows TOKEN IDS (integers)
  Token 32 represents the space character
  No special space markers are used

Key Differences:
1. SentencePiece works on Unicode code points
2. minbpe works on UTF-8 bytes
3. SentencePiece uses ▁ for space markers
4. minbpe uses byte 32 for spaces
5. SentencePiece output shows token strings
6. minbpe output shows token IDs
""")
    
    # Demonstrate with minbpe
    tokenizer = BasicTokenizer()
    text = "hello world"
    
    print(f"\nminbpe demonstration:")
    print(f"Input: '{text}'")
    
    token_ids = tokenizer.encode(text)
    print(f"Encoded: {token_ids}")
    
    # Show what token 32 (space) is
    space_token = tokenizer.vocab[32]
    print(f"\nToken 32 represents: {repr(space_token.decode('utf-8'))}")
    
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded: '{decoded}'")


def example_6_debugging_workflow():
    """
    Example 6: Step-by-step debugging workflow
    Shows how to investigate tokenization issues
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Debugging Tokenization Issues")
    print("="*80)
    
    print("""
Problem: "My tokenizer output doesn't look right"

Step-by-step debugging:
""")
    
    # Simulate a trained tokenizer
    tokenizer = BasicTokenizer()
    training_text = "the quick brown fox jumps over the lazy dog " * 20
    tokenizer.train(training_text, vocab_size=300, verbose=False)
    
    text = "the fox"
    
    print(f"\n1. Check basic encoding/decoding:")
    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)
    print(f"   Input:   '{text}'")
    print(f"   Encoded: {token_ids}")
    print(f"   Decoded: '{decoded}'")
    print(f"   Match: {text == decoded} {'✓' if text == decoded else '✗'}")
    
    print(f"\n2. Inspect each token:")
    for i, token_id in enumerate(token_ids):
        token_bytes = tokenizer.vocab[token_id]
        token_str = token_bytes.decode('utf-8', errors='replace')
        token_type = "merged" if token_id >= 256 else "byte"
        print(f"   Token {i}: ID={token_id} Value='{token_str}' Type={token_type}")
    
    print(f"\n3. Check compression:")
    original_bytes = len(text.encode('utf-8'))
    token_count = len(token_ids)
    compression = original_bytes / token_count if token_count > 0 else 0
    print(f"   Original bytes: {original_bytes}")
    print(f"   Token count: {token_count}")
    print(f"   Compression ratio: {compression:.2f} bytes/token")
    
    print(f"\n4. Use visualization tool for detailed analysis:")
    print(f"   python visualize_tokenizer.py --model your_model.model --text \"{text}\"")


def example_7_using_visualizer():
    """
    Example 7: Using the TokenizerVisualizer programmatically
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: Using TokenizerVisualizer")
    print("="*80)
    
    # Train a simple tokenizer
    tokenizer = BasicTokenizer()
    training_text = "hello world " * 50
    tokenizer.train(training_text, vocab_size=270, verbose=False)
    
    # Create visualizer
    visualizer = TokenizerVisualizer(tokenizer)
    
    # Visualize encoding
    visualizer.visualize_encoding("hello world")
    
    # Compare multiple texts
    print(f"\n{'='*80}")
    visualizer.compare_texts(["hello", "world", "hello world"])


def main():
    """Run all examples"""
    print("="*80)
    print("UNDERSTANDING TOKENIZER OUTPUT - EXAMPLES")
    print("="*80)
    print("""
This script demonstrates common tokenization concepts and how to debug issues.
Each example focuses on a different aspect of tokenization.
""")
    
    examples = [
        ("Basic Encoding", example_1_basic_encoding),
        ("Trained Tokenizer", example_2_trained_tokenizer),
        ("Non-Latin Scripts", example_3_non_latin_script),
        ("Token IDs vs Strings", example_4_comparing_outputs),
        ("SentencePiece vs BPE", example_5_sentencepiece_vs_bpe),
        ("Debugging Workflow", example_6_debugging_workflow),
        ("Using Visualizer", example_7_using_visualizer),
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        try:
            func()
        except Exception as e:
            print(f"\nError in example {i}: {e}")
        
        if i < len(examples):
            input(f"\nPress Enter to continue to next example...")
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)
    print("""
For more information:
- Read TOKENIZATION_GUIDE.md
- Use visualize_tokenizer.py for your own text
- Check the tests/ directory for more examples
""")


if __name__ == "__main__":
    main()
