import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from gpt_download import download_and_load_gpt2
from loading_gpt2 import load_weights_into_gpt
from pretraining import GPTModel
import time
import matplotlib.pyplot as plt

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
    
#calculating the classification accuracy
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:,-1,:]

            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += ((predicted_labels == target_batch).sum().item())

        else:
            break
    return correct_predictions / num_examples

#classification loss for a single batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:,-1,:]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

# classification loss loader for multiple batches
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch,target_batch,model,device)
            total_loss += loss
        else:
            break

    return total_loss / num_batches

#Finetuning the gpt 2 model to work as a classifier
def train_classfier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs, eval_freq, eval_iter
):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )
        
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")

        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    model.train()
    return train_loss, val_loss

#plotting the classification loss 
def plot_values(
    epochs_seen, examples_seen, train_values, val_values,
    label="loss"):
        fig, ax1 = plt.subplots(figsize=(5, 3))
        #1
        ax1.plot(epochs_seen, train_values, label=f"Training {label}")
        ax1.plot(
        epochs_seen, val_values, linestyle="-.",
        label=f"Validation {label}"
        )
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel(label.capitalize())
        ax1.legend()
        #2
        ax2 = ax1.twiny()
        ax2.plot(examples_seen, train_values, alpha=0) #3
        ax2.set_xlabel("Examples seen")
        fig.tight_layout() #4
        plt.savefig(f"{label}-plot.pdf")
        plt.show()

#classification of new texts using the finetuned gpt model
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]

    input_ids = input_ids[: min(max_length, supported_context_length)]

    input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_tensor = torch.tensor(
        input_ids, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:,-1,:]
    predicted_label = torch.argmax(logits,dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"

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

    #testing the classfier before training
    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    print("Inputs dimensions:", inputs.shape)

    with torch.no_grad():
        outputs = model(inputs)

    print("outputs:\n",outputs)
    print("outputs dimensions:", outputs.shape)

    #we will be using the last token as it has the compelete idead of the text before it because of self attention
    print("last output token:",outputs[:,-1,:])

    probas = torch.softmax(outputs[:,-1,:], dim=-1)
    label = torch.argmax(probas)
    print("Class label:", label.item())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    torch.manual_seed(123)
    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model,device, num_batches=5)
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")

    # fine tuning the gpt model for classification tasks using train_classfier_simple function

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5

    train_losses, val_losses, train_acc, val_acc , examples_seen = train_classfier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5
    )

    end_time = time.time()
    execution_time_min = (end_time - start_time) / 60
    print(f"Training complete in {execution_time_min:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

    epochs_tensor = torch.linspace(0, num_epochs, len(train_acc))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_acc))
    plot_values(
    epochs_tensor, examples_seen_tensor, train_acc, val_acc,
    label="accuracy"
    )

    #saving the generated model .pth is the convention for pytorch files
    torch.save(model.state_dict(), "classifier.pth")

    # saving the ADAMW optimizer is also important as it uses the historical data to adjust the learning rates without saving it the optimizer resets and the model may learn suboptimally
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        },
        "model_and_optimizer_classfier.pth"
    )