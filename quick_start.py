from minbpe import BasicTokenizer
tokenizer = BasicTokenizer()
text = "ܡܥܲܕܪܢܘܼܬܵܐ"
tokenizer.train(text, 256 + 3) # 256 are the byte tokens, then do 3 merges
print(tokenizer.encode(text))
# [258, 100, 258, 97, 99]
# print(tokenizer.decode([258, 100, 258, 97, 99]))
print(tokenizer.decode([258, 220, 178, 220, 149, 220, 170, 220, 162, 220, 152, 220, 188, 220, 172, 220, 181, 220, 144]))
# aaabdaaabac
tokenizer.save("toy")
# writes two files: toy.model (for loading) and toy.vocab (for viewing)