from gpt_download import download_and_load_gpt2
from pretraining import GPTModel
import torch
import numpy as np
import tiktoken
from pretraining import generate, text_to_token_ids, token_ids_to_text, create_dataloader_v1, calc_loss_loader


# utility function for checking the dimensions
def assign(left, right) :
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. left: {left.shape}, right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

#function for loading weights into gpt
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split( 
        (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
        gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
        gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
        gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        q_b, k_b, v_b = np.split(
        (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
        gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
        gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
        gpt.trf_blocks[b].att.W_value.bias, v_b)
        gpt.trf_blocks[b].att.out_proj.weight = assign(
        gpt.trf_blocks[b].att.out_proj.weight,
        params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
        gpt.trf_blocks[b].att.out_proj.bias,
        params["blocks"][b]["attn"]["c_proj"]["b"])
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
        gpt.trf_blocks[b].ff.layers[0].weight,
        params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
        gpt.trf_blocks[b].ff.layers[0].bias,
        params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
        gpt.trf_blocks[b].ff.layers[2].weight,
        params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
        gpt.trf_blocks[b].ff.layers[2].bias,
        params["blocks"][b]["mlp"]["c_proj"]["b"])
        gpt.trf_blocks[b].norm1.scale = assign(
        gpt.trf_blocks[b].norm1.scale,
        params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
        gpt.trf_blocks[b].norm1.shift,
        params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
        gpt.trf_blocks[b].norm2.scale,
        params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
        gpt.trf_blocks[b].norm2.shift,
        params["blocks"][b]["ln_2"]["b"])
    
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


if __name__ == "__main__" :

    settings, params = download_and_load_gpt2(
        model_size="124M", models_dir="gpt2"
    )

    # print("Settings:",settings)
    # print("Parameter dictionary:",params.keys())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    model_name = "gpt2-small (124M)"
    NEW_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024, #for gpt2 small model context length is 1024 not 256
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1, #2
    "qkv_bias": True #qkv bias is true for gpt model
    }

    #initializing the gpt model
    gpt = GPTModel(NEW_CONFIG)
    gpt.eval()

    #loading weights into gpt
    load_weights_into_gpt(gpt, params)
    gpt.to(device)

    #testing the output with the loaded weights
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    # token_ids = generate(
    #     model = gpt,
    #     idx = text_to_token_ids("Every effort moves you", tokenizer).to(device),
    #     max_new_tokens=25,
    #     context_size= NEW_CONFIG["context_length"],
    #     top_k=50,
    #     temperature=1.5
    # )
    # print("output text:\n",token_ids_to_text(token_ids, tokenizer))

    with open("verdict.txt", "r", encoding="utf-8") as f:
        text_data = f.read()

    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=256,
        stride=256,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=256,
        stride= 256,
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    gpt.to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader,gpt,device)
        val_loss = calc_loss_loader(val_loader,gpt,device)

    print("training loss:",train_loss)
    print("validation loss:",val_loss)

