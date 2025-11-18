"""
Tests for the tokenizer visualization and debugging utilities.
"""

import pytest
import io
import sys
from minbpe import BasicTokenizer, RegexTokenizer
from visualize_tokenizer import TokenizerVisualizer


class TestTokenizerVisualizer:
    """Tests for the TokenizerVisualizer class"""
    
    def test_visualizer_initialization(self):
        """Test that visualizer can be initialized with a tokenizer"""
        tokenizer = BasicTokenizer()
        visualizer = TokenizerVisualizer(tokenizer)
        assert visualizer.tokenizer == tokenizer
    
    def test_visualize_encoding_basic(self):
        """Test basic visualization of encoding"""
        tokenizer = BasicTokenizer()
        visualizer = TokenizerVisualizer(tokenizer)
        
        # Capture output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        visualizer.visualize_encoding("hello")
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Check that output contains expected sections
        assert "INPUT TEXT:" in output
        assert "ENCODED TOKENS:" in output
        assert "TOKEN BREAKDOWN:" in output
        assert "DECODED TEXT:" in output
        assert "hello" in output
        assert "[104, 101, 108, 108, 111]" in output
    
    def test_visualize_encoding_with_bytes(self):
        """Test visualization with byte display"""
        tokenizer = BasicTokenizer()
        visualizer = TokenizerVisualizer(tokenizer)
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        visualizer.visualize_encoding("hi", show_bytes=True)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Check that byte representation is shown
        assert "68" in output  # 'h' in hex
        assert "69" in output  # 'i' in hex
    
    def test_visualize_encoding_non_latin(self):
        """Test visualization with non-Latin scripts"""
        tokenizer = BasicTokenizer()
        visualizer = TokenizerVisualizer(tokenizer)
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Korean text
        visualizer.visualize_encoding("안녕")
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Should show that it's multi-byte
        assert "UTF-8 bytes: 6 bytes" in output
        assert "안녕" in output
    
    def test_analyze_token_distribution(self):
        """Test token distribution analysis"""
        tokenizer = BasicTokenizer()
        text = "hello hello world " * 50
        tokenizer.train(text, vocab_size=270, verbose=False)
        
        visualizer = TokenizerVisualizer(tokenizer)
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        visualizer.analyze_token_distribution("hello hello")
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        assert "TOKEN DISTRIBUTION ANALYSIS:" in output
        assert "Total tokens:" in output
        assert "Unique tokens:" in output
        assert "Most common tokens:" in output
    
    def test_compare_texts(self):
        """Test comparison of multiple texts"""
        tokenizer = BasicTokenizer()
        visualizer = TokenizerVisualizer(tokenizer)
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        visualizer.compare_texts(["hello", "world", "hello world"])
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        assert "TOKENIZATION COMPARISON:" in output
        assert "Text 1:" in output
        assert "Text 2:" in output
        assert "Text 3:" in output
        assert "Compression ratio:" in output
    
    def test_explain_token_byte(self):
        """Test explanation of a byte token"""
        tokenizer = BasicTokenizer()
        visualizer = TokenizerVisualizer(tokenizer)
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        visualizer.explain_token(104)  # 'h'
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        assert "TOKEN EXPLANATION:" in output
        assert "Token ID: 104" in output
        assert "Type: Byte token" in output
    
    def test_explain_token_merged(self):
        """Test explanation of a merged token"""
        tokenizer = BasicTokenizer()
        text = "hello hello hello"
        tokenizer.train(text, vocab_size=260, verbose=False)
        
        visualizer = TokenizerVisualizer(tokenizer)
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Token 256 should be the first merge
        visualizer.explain_token(256)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        assert "TOKEN EXPLANATION:" in output
        assert "Token ID: 256" in output
        assert "Type: Merge token" in output
        assert "Merge:" in output
    
    def test_explain_token_not_found(self):
        """Test explanation of a non-existent token"""
        tokenizer = BasicTokenizer()
        visualizer = TokenizerVisualizer(tokenizer)
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        visualizer.explain_token(9999)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        assert "not found in vocabulary" in output
    
    def test_lossless_encoding_detection(self):
        """Test detection of lossless encoding"""
        tokenizer = BasicTokenizer()
        visualizer = TokenizerVisualizer(tokenizer)
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        visualizer.visualize_encoding("test")
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Should indicate lossless encoding
        assert "LOSSLESS" in output
    
    def test_with_trained_tokenizer(self):
        """Test visualization with a trained tokenizer"""
        tokenizer = BasicTokenizer()
        training_text = "the quick brown fox jumps over the lazy dog " * 50
        tokenizer.train(training_text, vocab_size=280, verbose=False)
        
        visualizer = TokenizerVisualizer(tokenizer)
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        visualizer.visualize_encoding("the fox")
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Should have fewer tokens than bytes due to merges
        tokens = tokenizer.encode("the fox")
        assert len(tokens) < len("the fox")
        
        assert "INPUT TEXT:" in output
        assert "the fox" in output
    
    def test_regex_tokenizer_support(self):
        """Test that visualizer works with RegexTokenizer"""
        tokenizer = RegexTokenizer()
        visualizer = TokenizerVisualizer(tokenizer)
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        visualizer.visualize_encoding("Hello, world!")
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        assert "INPUT TEXT:" in output
        assert "Hello, world!" in output
    
    def test_empty_string(self):
        """Test visualization with empty string"""
        tokenizer = BasicTokenizer()
        visualizer = TokenizerVisualizer(tokenizer)
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        visualizer.visualize_encoding("")
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        assert "Length: 0 characters" in output
        assert "Number of tokens: 0" in output
    
    def test_special_characters(self):
        """Test visualization with special characters"""
        tokenizer = BasicTokenizer()
        visualizer = TokenizerVisualizer(tokenizer)
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        visualizer.visualize_encoding("Hello\nWorld\t!")
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Should handle newlines and tabs
        assert "INPUT TEXT:" in output
    
    def test_emoji_handling(self):
        """Test visualization with emoji (multi-byte characters)"""
        tokenizer = BasicTokenizer()
        visualizer = TokenizerVisualizer(tokenizer)
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        visualizer.visualize_encoding("Hello 😊")
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        assert "INPUT TEXT:" in output
        # Emoji is 4 bytes in UTF-8
        assert "UTF-8 bytes: 10 bytes" in output


class TestVisualizerIntegration:
    """Integration tests for the visualization tool"""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow: train -> visualize -> analyze"""
        # Train a tokenizer
        tokenizer = BasicTokenizer()
        training_text = "hello world " * 50
        tokenizer.train(training_text, vocab_size=270, verbose=False)
        
        # Create visualizer
        visualizer = TokenizerVisualizer(tokenizer)
        
        # Visualize encoding
        test_text = "hello world"
        token_ids = tokenizer.encode(test_text)
        decoded = tokenizer.decode(token_ids)
        
        # Verify lossless
        assert test_text == decoded
        
        # Verify compression (should have fewer tokens than bytes)
        assert len(token_ids) <= len(test_text)
    
    def test_multiple_visualizers(self):
        """Test using multiple visualizers with different tokenizers"""
        tokenizer1 = BasicTokenizer()
        tokenizer2 = BasicTokenizer()
        
        text = "test test test " * 50
        tokenizer2.train(text, vocab_size=270, verbose=False)
        
        viz1 = TokenizerVisualizer(tokenizer1)
        viz2 = TokenizerVisualizer(tokenizer2)
        
        # Untrained should have more tokens
        ids1 = tokenizer1.encode(text)
        ids2 = tokenizer2.encode(text)
        
        assert len(ids1) > len(ids2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
