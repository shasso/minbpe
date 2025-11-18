"""
Tokenizer Visualization and Debugging Utility

This module provides tools to understand how tokenizers encode text,
particularly useful for debugging tokenization issues and understanding
the relationship between input text and output tokens.

Usage:
    python visualize_tokenizer.py --text "your text here" --model models/basic.model
    
Or use it programmatically:
    from visualize_tokenizer import TokenizerVisualizer
    visualizer = TokenizerVisualizer(tokenizer)
    visualizer.visualize_encoding("hello world")
"""

import argparse
from typing import List, Tuple, Optional
from minbpe import BasicTokenizer, RegexTokenizer


class TokenizerVisualizer:
    """
    A utility class to visualize and debug tokenizer behavior.
    Helps understand the mapping between input text and encoded tokens.
    """
    
    def __init__(self, tokenizer):
        """
        Initialize the visualizer with a tokenizer.
        
        Args:
            tokenizer: A tokenizer instance (BasicTokenizer, RegexTokenizer, etc.)
        """
        self.tokenizer = tokenizer
    
    def visualize_encoding(self, text: str, show_bytes: bool = False) -> None:
        """
        Visualize how the tokenizer encodes the given text.
        Shows the mapping between tokens and their corresponding text.
        
        Args:
            text: The text to encode and visualize
            show_bytes: If True, also show the byte representation
        """
        print(f"\n{'='*80}")
        print(f"INPUT TEXT:")
        print(f"{'='*80}")
        print(f"{text}")
        print(f"\nLength: {len(text)} characters")
        print(f"UTF-8 bytes: {len(text.encode('utf-8'))} bytes")
        
        # Encode the text
        token_ids = self.tokenizer.encode(text)
        
        print(f"\n{'='*80}")
        print(f"ENCODED TOKENS:")
        print(f"{'='*80}")
        print(f"Token IDs: {token_ids}")
        print(f"Number of tokens: {len(token_ids)}")
        
        # Decode each token individually to show what each represents
        print(f"\n{'='*80}")
        print(f"TOKEN BREAKDOWN:")
        print(f"{'='*80}")
        print(f"{'ID':<8} {'Token':<30} {'Bytes':<40}")
        print(f"{'-'*8} {'-'*30} {'-'*40}")
        
        for i, token_id in enumerate(token_ids):
            token_bytes = self.tokenizer.vocab.get(token_id, b'<UNK>')
            try:
                token_text = token_bytes.decode('utf-8', errors='replace')
            except:
                token_text = '<DECODE_ERROR>'
            
            if show_bytes:
                byte_repr = ' '.join(f'{b:02x}' for b in token_bytes)
                print(f"{token_id:<8} {repr(token_text):<30} {byte_repr:<40}")
            else:
                print(f"{token_id:<8} {repr(token_text):<30}")
        
        # Verify decoding
        decoded_text = self.tokenizer.decode(token_ids)
        
        print(f"\n{'='*80}")
        print(f"DECODED TEXT:")
        print(f"{'='*80}")
        print(f"{decoded_text}")
        
        # Check if encoding/decoding is lossless
        if text == decoded_text:
            print(f"\n✓ Encoding/Decoding is LOSSLESS (perfect match)")
        else:
            print(f"\n✗ WARNING: Encoding/Decoding is LOSSY (text changed)")
            print(f"  Original length: {len(text)}")
            print(f"  Decoded length: {len(decoded_text)}")
            self._show_differences(text, decoded_text)
    
    def _show_differences(self, original: str, decoded: str) -> None:
        """Show differences between original and decoded text."""
        print(f"\n  Differences:")
        max_len = max(len(original), len(decoded))
        for i in range(min(20, max_len)):  # Show first 20 characters
            if i < len(original) and i < len(decoded):
                if original[i] != decoded[i]:
                    print(f"    Position {i}: '{original[i]}' → '{decoded[i]}'")
            elif i >= len(original):
                print(f"    Position {i}: <missing> → '{decoded[i]}'")
            elif i >= len(decoded):
                print(f"    Position {i}: '{original[i]}' → <missing>")
    
    def analyze_token_distribution(self, text: str) -> None:
        """
        Analyze the distribution of tokens in the encoded text.
        Shows statistics about token usage.
        
        Args:
            text: The text to analyze
        """
        token_ids = self.tokenizer.encode(text)
        
        print(f"\n{'='*80}")
        print(f"TOKEN DISTRIBUTION ANALYSIS:")
        print(f"{'='*80}")
        
        # Count token frequencies
        from collections import Counter
        token_counts = Counter(token_ids)
        
        print(f"\nTotal tokens: {len(token_ids)}")
        print(f"Unique tokens: {len(token_counts)}")
        print(f"Vocabulary size: {len(self.tokenizer.vocab)}")
        
        # Show most common tokens
        print(f"\nMost common tokens:")
        print(f"{'Rank':<6} {'ID':<8} {'Count':<8} {'Token':<30}")
        print(f"{'-'*6} {'-'*8} {'-'*8} {'-'*30}")
        
        for rank, (token_id, count) in enumerate(token_counts.most_common(10), 1):
            token_bytes = self.tokenizer.vocab.get(token_id, b'<UNK>')
            try:
                token_text = token_bytes.decode('utf-8', errors='replace')
            except:
                token_text = '<DECODE_ERROR>'
            print(f"{rank:<6} {token_id:<8} {count:<8} {repr(token_text):<30}")
    
    def compare_texts(self, texts: List[str]) -> None:
        """
        Compare how multiple texts are tokenized.
        Useful for understanding tokenizer behavior across different inputs.
        
        Args:
            texts: List of texts to compare
        """
        print(f"\n{'='*80}")
        print(f"TOKENIZATION COMPARISON:")
        print(f"{'='*80}")
        
        for i, text in enumerate(texts, 1):
            token_ids = self.tokenizer.encode(text)
            print(f"\nText {i}: {repr(text)}")
            print(f"  Length: {len(text)} chars, {len(text.encode('utf-8'))} bytes")
            print(f"  Tokens: {len(token_ids)}")
            print(f"  Token IDs: {token_ids}")
            print(f"  Compression ratio: {len(text.encode('utf-8')) / len(token_ids):.2f} bytes/token")
    
    def explain_token(self, token_id: int) -> None:
        """
        Explain what a specific token represents.
        
        Args:
            token_id: The token ID to explain
        """
        print(f"\n{'='*80}")
        print(f"TOKEN EXPLANATION:")
        print(f"{'='*80}")
        
        if token_id not in self.tokenizer.vocab:
            print(f"Token ID {token_id} not found in vocabulary!")
            print(f"Vocabulary size: {len(self.tokenizer.vocab)}")
            return
        
        token_bytes = self.tokenizer.vocab[token_id]
        
        print(f"Token ID: {token_id}")
        print(f"Bytes: {' '.join(f'{b:02x}' for b in token_bytes)}")
        print(f"Length: {len(token_bytes)} bytes")
        
        try:
            token_text = token_bytes.decode('utf-8', errors='replace')
            print(f"Text: {repr(token_text)}")
        except:
            print(f"Text: <DECODE_ERROR>")
        
        # Check if it's a byte token or merge token
        if token_id < 256:
            print(f"Type: Byte token (base vocabulary)")
            print(f"ASCII: {chr(token_id) if 32 <= token_id < 127 else '<non-printable>'}")
        else:
            print(f"Type: Merge token (learned from training)")
            # Try to find the merge that created this token
            if hasattr(self.tokenizer, 'merges'):
                for (p0, p1), idx in self.tokenizer.merges.items():
                    if idx == token_id:
                        print(f"Merge: [{p0}] + [{p1}] → [{token_id}]")
                        if p0 in self.tokenizer.vocab and p1 in self.tokenizer.vocab:
                            try:
                                left = self.tokenizer.vocab[p0].decode('utf-8', errors='replace')
                                right = self.tokenizer.vocab[p1].decode('utf-8', errors='replace')
                                print(f"       {repr(left)} + {repr(right)} → {repr(token_text)}")
                            except:
                                pass
                        break


def main():
    parser = argparse.ArgumentParser(
        description='Visualize and debug tokenizer behavior',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize encoding of a simple text
  python visualize_tokenizer.py --text "hello world"
  
  # Load a trained model and analyze text
  python visualize_tokenizer.py --model models/basic.model --text "hello world"
  
  # Show byte-level details
  python visualize_tokenizer.py --text "hello" --bytes
  
  # Compare multiple texts
  python visualize_tokenizer.py --compare "hello" "world" "hello world"
  
  # Explain a specific token
  python visualize_tokenizer.py --model models/basic.model --explain-token 258
        """
    )
    
    parser.add_argument('--text', type=str, help='Text to encode and visualize')
    parser.add_argument('--model', type=str, help='Path to trained tokenizer model')
    parser.add_argument('--tokenizer', type=str, choices=['basic', 'regex'], 
                        default='basic', help='Tokenizer type to use (default: basic)')
    parser.add_argument('--bytes', action='store_true', 
                        help='Show byte-level representation')
    parser.add_argument('--compare', nargs='+', help='Compare multiple texts')
    parser.add_argument('--analyze', type=str, help='Analyze token distribution in text')
    parser.add_argument('--explain-token', type=int, help='Explain a specific token ID')
    
    args = parser.parse_args()
    
    # Create tokenizer
    if args.tokenizer == 'basic':
        tokenizer = BasicTokenizer()
    else:
        tokenizer = RegexTokenizer()
    
    # Load model if specified
    if args.model:
        print(f"Loading tokenizer from {args.model}...")
        tokenizer.load(args.model)
        print(f"Loaded vocabulary of size {len(tokenizer.vocab)}")
    
    visualizer = TokenizerVisualizer(tokenizer)
    
    # Execute requested action
    if args.compare:
        visualizer.compare_texts(args.compare)
    elif args.analyze:
        visualizer.analyze_token_distribution(args.analyze)
    elif args.explain_token is not None:
        visualizer.explain_token(args.explain_token)
    elif args.text:
        visualizer.visualize_encoding(args.text, show_bytes=args.bytes)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
