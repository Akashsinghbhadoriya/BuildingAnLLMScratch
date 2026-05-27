## Dummy class for GPT 

import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
from selfatten import MultiheadAttention

GPT_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of layers
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False # Query-Key-Value bias
}

class DummyGPTModel(nn.Module) :
    def __init__(self, cfg): 
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx) :
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self,x):
        return x
    
class DummyLayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


#actual layer normalization class
#Parameter wraps the tensor for auto matically tracking and updating it during backpropogation
#scale and shift are trainable parameters that llm adjust during training
class LayerNorm(nn.Module) :
    def __init__(self, emb_dim) :
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim)) #weight which controls the variance and scaling of normalized embeddings
        self.shift = nn.Parameter(torch.zeros(emb_dim)) #bias which controls the offset/mean of the normalized embeddings

    def forward(self, x) :
        mean = x.mean(dim =-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    

#Implementation of GeLU
class GELU(nn.Module) :
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh( torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x,3))))


#feedforward neural network where we expand the dimension for richer calculation and then resizing to the original dimension
class FeedForward(nn.Module) :
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 *  cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x) :
        return self.layers(x)
    


#neural network illustrating shortcut connections
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x) :
        for layer in self.layers :
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x
    

def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


#connecting the attention and the linear layer in a transformer block which is repeated a dozen times in the 124-million-block
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiheadAttention(
            d_in = cfg['emb_dim'],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg['qkv_bias'])
        
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x) :
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
    

##The final GPT Model architecture implementation
class GPTModel(nn.Module) :
    def __init__(self, cfg) :
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self,in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def generate_text_simple(model, idx, max_new_tokens, context_size) :
    for _ in range(max_new_tokens) :
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:,-1,:]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas,dim=-1,keepdim=True)
        idx = torch.cat((idx, idx_next), dim=-1)
    
    return idx


if __name__ == "__main__" :
    #usage of dummy gpt class 
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))

    batch = torch.stack(batch, dim=0)
    # print(batch)

    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    # print("Output shape:",logits.shape)
    # print(logits)

    #example layer normalization

    torch.manual_seed(123)
    batch_example = torch.randn(2,5)
    layer = nn.Sequential(nn.Linear(5,6), nn.ReLU()) # using a non-linear activation function Rectified Linear unit
    out = layer(batch_example)
    # print(out)

    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)
    # print("Mean:\n",mean)
    # print("Var:\n",var)

    # applying layer normalization to the output
    out_norm = (out - mean) / torch.sqrt(var)
    norm_mean = out_norm.mean(dim=-1, keepdim=True)
    norm_var = out_norm.var(dim = -1, keepdim=True)
    # print("Normalized layer outputs:\n", out_norm)
    # print("Mean:\n",norm_mean)
    # print("Variance:\n",norm_var)

    #normalization example 
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    mean_ln = out_ln.mean(dim=-1,keepdim=True)
    var_ln = out_ln.var(dim =-1, unbiased=False, keepdim=True)
    # print("Mean:\n",mean_ln)
    # print("var:\n",var_ln)

    #plotting gelu and relu parallelly

    gelu, relu = GELU(), nn.ReLU()

    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)
    plt.figure(figsize=(8,3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label} (x)")
        plt.grid(True)
    plt.tight_layout()
    # plt.show()

    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2,3,768)
    out = ffn(x)
    # print(out.shape)

    layer_sizes = [3,3,3,3,3,1]
    sample_input = torch.tensor([[1., 0., -1.]])
    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes,use_shortcut=False)

    # print_gradients(model_without_shortcut, sample_input)

    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=True
    )
    # print_gradients(model_with_shortcut, sample_input)

    torch.manual_seed(123)
    x = torch.rand(2,4,768)
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)

    # print("Input Shape:",x.shape)
    # print("Output shape:", output.shape)

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    out = model(batch)
    # print("Input batch:\n",batch)
    # print("\nOutput Shape:",out.shape)
    # print(out)
    total_params = sum(p.numel() for p in model.parameters()) # Sum of all the parameters in the model
    # print(f"total number of parameters: {total_params:,}") #163,009,536 why we got 163 million rather than 124 million is because of weight tying
    #Weight tying because the original gpt-2 architecture reuses the  the weights from the token embedding layer in its output layer so if we subtract this we get 124 million
    total_params_gpt2 = (
        total_params - sum(p.numel() for p in model.out_head.parameters())
    )
    # print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

    total_size_bytes = total_params * 4 #1
    total_size_mb = total_size_bytes / (1024 * 1024) #2
    # print(f"Total size of the model: {total_size_mb:.2f} MB") #each parameter is 32-bit float taking up 4 bytes

    #gpt-2 medium 
    GPT_2_MEDIUM = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 1024, # Embedding dimension
    "n_heads": 16, # Number of attention heads
    "n_layers": 24, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
    }

    #commented for training uncomment and test if required
    # model_2 = GPTModel(GPT_2_MEDIUM)
    # total_params_2 = sum(p.numel() for p in model_2.parameters())
    # total_params_gpt2medium = (total_params_2 - sum(p.numel() for p in model_2.out_head.parameters()))
    # print(f"{total_params_gpt2medium:,}")

    #gpt-2 large 
    GPT_2_Large = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 1280, # Embedding dimension
    "n_heads": 20, # Number of attention heads
    "n_layers": 36, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
    }
    #commented for training uncomment and test if required
    # model_3 = GPTModel(GPT_2_Large)
    # total_params_3 = sum(p.numel() for p in model_3.parameters())
    # total_params_gpt2large = (total_params_3 - sum(p.numel() for p in model_3.out_head.parameters()))
    # print(f"{total_params_gpt2large:,}")

    #gpt 2 XL
    GPT_2_XL = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 1600, # Embedding dimension
    "n_heads": 25, # Number of attention heads
    "n_layers": 48, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
    }

    #commented for training uncomment and test if required
    # model_4 = GPTModel(GPT_2_XL)
    # total_params_4= sum(p.numel() for p in model_4.parameters())
    # total_params_gpt2XL = (total_params_4 - sum(p.numel() for p in model_4.out_head.parameters()))
    # print(f"{total_params_gpt2XL:,}")

    #Generating the text from the generated tokens
    #steps of converting the tokens back to text are below
    #1] extract the last vector from the output response which will correspond to the next token
    #2] convert the logits into probability distribution using the softmax function
    #3] Identify the index pos of the largest value which also represent the token ID
    #4] append the token to the previous inputs for the next round
    #simple class for generating the text from the generated logits output of the gpt model
    #model is calculating the most likely next token so it is called greedy decoding

    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    # print("ecoded:",encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    # print("encoded tensor:",encoded_tensor)

    model.eval()
    out = generate_text_simple(
        model=model,
        idx = encoded_tensor,
        max_new_tokens=8,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    # print("output:",out)
    # print("output length:", len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    # print("decoded_text:",decoded_text)