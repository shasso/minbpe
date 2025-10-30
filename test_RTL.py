from minbpe import BasicTokenizer
from minbpe.utils import codepoints, display_for_console, wrap_rle, add_rlm
t = BasicTokenizer()
s = "ܡܥܲܕܪܢܘܼܬܵܐ"  # example Syriac string
t.train(s, 256 + 3)
ids = t.encode(s)
decoded = t.decode(ids)

print("original repr:", repr(s))
print("decoded repr: ", repr(decoded))
print("original codepoints:", [hex(ord(c)) for c in s])
print("decoded codepoints: ", [hex(ord(c)) for c in decoded])

# visual helpers / examples
print("\n--- visual helpers ---")
print("wrap_rle: ", repr(wrap_rle(decoded)))
print("add_rlm:  ", repr(add_rlm(decoded)))
print("display_for_console (fallback RLE): ", repr(display_for_console(decoded, use_bidi=False)))
try:
	# if python-bidi is installed, show the get_display output
	print("display_for_console (python-bidi): ", repr(display_for_console(decoded, use_bidi=True)))
except Exception as e:
	print("display_for_console (python-bidi) raised:", repr(e))