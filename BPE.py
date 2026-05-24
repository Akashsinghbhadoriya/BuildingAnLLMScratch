import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
tokenizer = tiktoken.get_encoding("gpt2")

# text = (
# "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
# "of someunknownPlace."
# )
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)

# strings = tokenizer.decode(integers)
# print(strings)

with open("verdict.txt","r",encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
# print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 5

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    # print(context, "--->", desired)
    # print(tokenizer.decode(context), "--->", tokenizer.decode([desired]))

#dataset class using pytorch
class GPTDatasetV1(Dataset) :
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt) #encode using BPE tiktoken library

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+1+max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, key):
        return self.input_ids[key], self.target_ids[key]
    
#dataloader function for iterable on dataset
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt,tokenizer,max_length,stride)
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
        )
    
    return dataloader

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)

data_iter = iter(dataloader)
input, target = next(data_iter)
# print("input-->",input)
# print("\ntarget",target)

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size,output_dim)

input_token_embedding = token_embedding_layer(input)
print(input_token_embedding.size())
# print(token_embedding)

target_token_embedding = token_embedding_layer(target)
print(target_token_embedding.size())

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length,output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length)) # torch.arange array from 0,1,...,context_length-1
print(pos_embeddings.size())
# print(pos_embeddings)

input_embedding = input_token_embedding + pos_embeddings
target_embedding = target_token_embedding + pos_embeddings

print(input_embedding)