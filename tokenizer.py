import re

with open("verdict.txt","r",encoding="utf-8") as f:
    raw_text = f.read()

print("total number of characters", len(raw_text))
print(raw_text[:99])

tokenizer_list = re.split(r'([,.:;?_!"()\']|--|\s)',raw_text)
tokenizer_list = [item for item in tokenizer_list if item.strip()]
print(len(tokenizer_list))

unique_words = sorted(set(tokenizer_list))
unique_words.extend(["<|endoftext|>", "<|unk|>"])
print("unique words",len(unique_words))

vocab = {token:integer for integer,token in enumerate(unique_words)}
for i , item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

class SimpleTokenizerV1:

    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self,text):
        preprocess_text = re.split(r'([,.:;?_!"()\']|--|\s)',text)
        preprocess_text = [item.strip() for item in preprocess_text if item.strip()]

        preprocess_text = [item if item in self.str_to_int else "<|unk|>" for item in preprocess_text]

        ids = [self.str_to_int[i] for i in preprocess_text]
        return ids
    
    def decode(self,ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r"\s+([,.?!\"()\\'])", r"\1", text)
        return text
    

tokenizer = SimpleTokenizerV1(vocab)
# text = """"It's the last he painted, you know,"
# Mrs. Gisburn said with pardonable pride."""
# ids = tokenizer.encode(text)
# print(ids)

# print(tokenizer.decode(ids))

text3 = "Hello, do you like tea?"
print(tokenizer.encode(text3))

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

ids = tokenizer.encode(text)

print(ids)

print(tokenizer.decode(ids))