#For Using this class first finetune the gpt model on the classification data run the classfication_finetuning.py file for this
from pretraining import GPTModel, GPT_CONFIG_124M
import torch
import tiktoken
from classification_finetuning import classify_review

NEW_CONFIG = GPT_CONFIG_124M
NEW_CONFIG.update(context_length=1024, qkv_bias=True)

model = GPTModel(NEW_CONFIG)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("model_and_optimizer_classfier.pth", map_location=device)

model.out_head = torch.nn.Linear(
        in_features=NEW_CONFIG["emb_dim"],
        out_features=2)
model.load_state_dict(checkpoint["model_state_dict"])

tokenizer = tiktoken.get_encoding("gpt2")

text_1 = (
"You are a winner you have been specially"
" selected to receive $1000 cash or a $2000 award."
)
print(classify_review(
    text_1, model, tokenizer, device, max_length=120
))

text_2 = (
"Hey, just wanted to check if we're still on"
" for dinner tonight? Let me know!"
)
print(classify_review(
text_2, model, tokenizer, device, max_length=120
))
