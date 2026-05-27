import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from gpt_download import download_and_load_gpt2
from loading_gpt2 import load_weights_into_gpt
from pretraining import GPTModel

#creating a balanced dataset using the data where the spam and not spam are equal
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123
    )
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
    ])
    return balanced_df

# splitting the data into training validation and testing data
def random_split(df, train_frac, validation_frac) :
    
    df = df.sample(
        frac=1, random_state=123
    ).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

def download_classification_dataset(train_df, validation_df, test_df):
    for filename, df in [
        ("train.csv", train_df),
        ("validation.csv", validation_df),
        ("test.csv", test_df),
    ]:
        if Path(filename).exists():
            print(f"{filename} already exists. Skipping.")
        else:
            df.to_csv(filename, index=None)
            print(f"Saved {filename}.")

# for padding or truncate the dataset to form uniform length
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

            self.encoded_texts = [
                encoded_text[:self.max_length] for encoded_text in self.encoded_texts
            ]
        
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_len = len(encoded_text)
            if encoded_len > max_length:
                max_length = encoded_len
        return max_length

if __name__ == "__main__" :
    extracted_path = "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
    df = pd.read_csv(
        data_file_path, sep="\t", header=None, names=["Label", "Text"]
    )
    # print(df)
    # print(df["Label"].value_counts())
    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())

    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    # splitting the data
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    
    #downloading the classification data
    download_classification_dataset(train_df, validation_df, test_df)

    tokenizer = tiktoken.get_encoding("gpt2")
    #We need to create dataset and dataloaders for that we need to make the shorter text equal to the longer once by adding endoftext to the shorter once
    train_dataset = SpamDataset(
        csv_file="train.csv",
        max_length=None,
        tokenizer=tokenizer
    )

    val_dataset = SpamDataset(
        csv_file="validation.csv",
        max_length=None,
        tokenizer=tokenizer
    )

    test_dataset = SpamDataset(
        csv_file="test.csv",
        max_length=None,
        tokenizer=tokenizer
    )

    print(train_dataset.max_length)

    # creating the dataloader for fine-tuning here the targets represent the class labels and not next token in the sequence
    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
    )

    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")

    #Initialize a model with pretrained weights

    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"
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

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    #loading weights of the gpt model publically available
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    
    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir="gpt2"
    )
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    # printing the model to check the configurations
    # print(model)  
    # now we will modify the out_head to utilize the model for classification finetuning
    # freeze all layers we will train only the last layers for fine-tuning it
    # 
    for param in model.parameters():
        param.requires_grad = False

    # modifying the out_head
    torch.manual_seed(123)
    num_classes = 2

    # this model out head has requires_grad = true by default which means training the outer layer is sufficient for getting the output but if we train further more layers it improve the preditive performance of the model
    model.out_head = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"],
        out_features=num_classes
    )  

    # making the Final LayerNorm and the last transformer block as trainable
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    


