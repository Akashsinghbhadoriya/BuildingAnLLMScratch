import torch
import tiktoken
from llm_architecture import GPTModel, generate_text_simple


#reduced the context length compared to the gpt2 small model for training on a laptop
GPT_CONFIG_124M = {
"vocab_size": 50257,
"context_length": 256, #1
"emb_dim": 768,
"n_heads": 12,
"n_layers": 12,
"drop_rate": 0.1, #2
"qkv_bias": False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

# complete architecutre from text to token conversion to token ID to text 

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

# print("output text:\n",token_ids_to_text(token_ids,tokenizer))

#calculating text generation loss for training process
# we will do backpropogation to update the weights so that we get the desired output for the input tokens
#we will use log probabilities for calculating the losses as it is much easier

#In deep learning the term for turning the negative value is called cross entropy loss it is a build in function in pytorch
#cross entropy loss is a difference between two probability distributions 
# typically true distribution of labels and predicted distribution of labels,
# perplexity is the measure often used alongside cross entropy loss to evaluate model performance in language modeling
#Perplexity provides a more interpretable way to understand the uncertainity of a model in predicting the next token in a sequence
# perplexity = torch.exp(loss) which returns tensor(48725.8203) this means model is not sure about which among the 48725 tokens in the vocabulary to generate as the next token
# this example is listed below

input_text1 = "every effort moves"
input_text2 = "I really like"

target_text1 = "effort moves you"
target_text2 = "really like chocolate"

output_token_ids1 = generate_text_simple(
    model=model,
    idx=text_to_token_ids(input_text1,tokenizer),
    max_new_tokens=1,
    context_size=GPT_CONFIG_124M["context_length"]
)
output_token_ids2 = generate_text_simple(
    model=model,
    idx=text_to_token_ids(input_text2,tokenizer),
    max_new_tokens=1,
    context_size=GPT_CONFIG_124M["context_length"]
)

print(output_token_ids1)
print(output_token_ids2)

output_text1 = token_ids_to_text(output_token_ids1,tokenizer)
output_text2 = token_ids_to_text(output_token_ids2, tokenizer)

print(output_text1)
print(output_text2)