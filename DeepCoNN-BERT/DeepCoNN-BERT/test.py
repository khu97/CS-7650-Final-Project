from transformers import AutoTokenizer
import time

tok = AutoTokenizer.from_pretrained("bert-mini")
text = "this is a test sentence" * 20

start = time.time()
for _ in range(5000):
    tok(text, truncation=True, padding="max_length", max_length=256)
print("5000 tokenizations took:", time.time() - start, "seconds")