from pretraining import GPTModel, text_to_token_ids, token_ids_to_text, generate
from instruction_finetuning import format_input
from openai import OpenAI
from tqdm import tqdm
import tiktoken
import torch
import json
import os
from dotenv import load_dotenv
load_dotenv()


def generate_test_responses(test_data, model, tokenizer, device, BASE_CONFIG):
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )

        generated_text = token_ids_to_text(token_ids, tokenizer)

        reponse_text = (
            generated_text[len(input_text):]
            .replace("### Reponse:", "")
            .strip()
        )
        test_data[i]["model_response"] = reponse_text

    with open("instruction-data-with-response.json", "w") as file:
        json.dump(test_data, file, indent=4)


def evaluate_with_gpt(test_data, api_key, model="gpt-4o-mini"):
    client = OpenAI(api_key=api_key)
    scores = []

    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        prompt = (
            "You are evaluating a language model's response to an instruction.\n"
            f"Instruction: {entry['instruction']}\n"
            f"Input: {entry.get('input', '')}\n"
            f"Expected output: {entry['output']}\n"
            f"Model response: {entry['model_response']}\n\n"
            "Score the model response from 0 to 100, where 100 is a perfect match "
            "in meaning and quality to the expected output. "
            'Respond with only a JSON object: {"score": <integer>}'
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        score = json.loads(response.choices[0].message.content)["score"]
        test_data[i]["gpt_score"] = score
        scores.append(score)

    print(f"Mean: {sum(scores) / len(scores):.1f}  Min: {min(scores)}  Max: {max(scores)}")

    with open("instruction-data-with-response.json", "w") as file:
        json.dump(test_data, file, indent=4)


if __name__ == "__main__":
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    tokenizer = tiktoken.get_encoding("gpt2")
    with open("instruction-data.json", "r") as file:
        data = json.load(file)

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)

    test_data = data[train_portion : train_portion + test_portion]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(BASE_CONFIG)
    model_state = torch.load("instruction.pth", map_location=device)
    model.load_state_dict(model_state)
    model.eval()

    # generate_test_responses(test_data, model, tokenizer, device, BASE_CONFIG)

    with open("instruction-data-with-response.json", "r") as file:
        test_data_with_response = json.load(file)

    api_key = os.environ.get("OPENAI_API_KEY")
    evaluate_with_gpt(test_data_with_response, api_key)
