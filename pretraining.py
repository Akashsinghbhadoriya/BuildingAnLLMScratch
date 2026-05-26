import torch
import tiktoken
from llm_architecture import GPTModel, generate_text_simple
from BPE import create_dataloader_v1

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

# complete architecutre from text to token conversion to token ID to text 

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

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


#utility function for calculating the cross-entropy loss
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

#building a data loader for calculating the losses for the batch
def calc_loss_loader(data_loader, model, device, num_batches=None) :
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader) :
        if i < num_batches:
            loss = calc_loss_batch(input_batch,target_batch,model,device)
            total_loss += loss.item()
        else:
            break
    return total_loss/num_batches


#implementing the training model function
def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer) :
    train_losses, val_losses , track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs) :
        model.train() #sets the model in training mode to use dropout and batch normalization
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # clear existing gradients
            loss = calc_loss_batch(input_batch,target_batch,model,device)
            loss.backward() #triggers backpropogation
            optimizer.step() # update the model weights during training
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(model,tokenizer,device,start_context)
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() #enables evaluation of the model dropout is disabled during this stage 
    with torch.no_grad():  #also disable gradient tracking during evaluation
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

# modifiying the text generation to use probabilistic sampling, temperature scaling and top-k sampling

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # take the last data based on the token size if 4 taking the last four columns
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] # takes the last token from the logits keeps the vocabulary size and the batch size

        # using top k for smapling the top k tokens with highest probablity and masking the remaining ones
        if top_k is not None:
            top_logits, _ = torch.topk(logits,top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        
        # implementing temperature scaling on the generate output
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # implementing probabilistic sampling
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx

if __name__ == "__main__":
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    # print("output text:\n",token_ids_to_text(token_ids,tokenizer))
    input_text1 = "every effort moves"
    input_text2 = "I really like"

    target_text1 = "effort moves you"
    target_text2 = "really like chocolate"

    # print("input text1 token::",text_to_token_ids(input_text1,tokenizer))
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

    # print(output_token_ids1[:, -3:])
    # print(output_token_ids2[:, -3:])

    output_text1 = token_ids_to_text(output_token_ids1[:, -3:],tokenizer)
    output_text2 = token_ids_to_text(output_token_ids2[:, -3:], tokenizer)

    # print(output_text1)
    # print(output_text2)

    #calculating the training and validation set losses using cross entropy
    #we will be using "The Verdict" short story for training the model

    file_path = "verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    # print("Characters:",total_characters)
    # print("Tokens:", total_tokens)

    # splitting the training data into training and validation
    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    #creating data loader for train anv val data
    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    # print("train loader\n")
    # for x,y in train_loader:
    #     print(x.shape,y.shape)
    # print("val loader:\n")
    # for x,y in val_loader:
    #     print(x.shape,y.shape)

    # applying the calc_loss_loader to training and validation function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad(): #disables gradient calculation
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader,model,device)

    print("training loss:",train_loss)
    print("validation loss:", val_loss)

    # train a gpt model for 10 epochs
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, weight_decay=0.1
    )
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer = tokenizer
    )

    #saving the generated model .pth is the convention for pytorch files
    torch.save(model.state_dict(), "model.pth")

    # saving the ADAMW optimizer is also important as it uses the historical data to adjust the learning rates without saving it the optimizer resets and the model may learn suboptimally
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        },
        "model_and_optimizer.pth"
    )

    # after above training data the training loss goes from 10.xx to 0.39 bu validation loss goes from 10.xx to 6.xx this means the model is overfitting on the training data model is memorizing the training data
    # this memorization is expected as we are working on very small amount of training dataset


    # strategies for controlling randomness
    model.to("cpu") # for inference we only require cpu
    model.eval() # change the model to evaluation mode to remove unecessary things such as dropout

    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text_simple(
        model=model,
        idx = text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("output text:\n", token_ids_to_text(token_ids, tokenizer))

    torch.manual_seed(123)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )
    print("output text:\n",token_ids_to_text(token_ids, tokenizer))


