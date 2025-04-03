import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from models import plot_training_curve

######################### creating typo dataset ###############################

def get_random_words(file_path, num_words):
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    num_words = min(num_words, len(lines))
    random_lines = random.sample(lines, num_words)

    return {line.strip() for line in random_lines}


def introduce_typo(word):

        if len(word) <= 2:
            return word  
        
        typo_type = random.choice(["swap", "replace"])
        if typo_type == "swap" and len(word) > 1:
            i = random.randint(0, len(word) - 2)
            word = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        elif typo_type == "replace":
            i = random.randint(0, len(word) - 1)
            word = word[:i] + random.choice("abcdefghijklmnopqrstuvwxyz") + word[i + 1:]

        return word

def word_to_seq(word):
    vocab = "abcdefghijklmnopqrstuvwxyz"
    stoi = {char: idx + 1 for idx, char in enumerate(vocab)}
    stoi["<PAD>"] = 0
    return [stoi.get(char, 0) for char in word]

class TypoDataset(Dataset):
    def __init__(self, word_list, num_typos=5, correct_ratio=0.4):
      
        self.data = []
        total_correct = int(len(word_list) * correct_ratio)

        #add correct words
        for word in random.sample(word_list, total_correct):
            self.data.append((word_to_seq(word), word_to_seq(word)))

        #add typo variations
        for word in word_list:
            for _ in range(num_typos):
                typo = introduce_typo(word)
                self.data.append((word_to_seq(typo), word_to_seq(word)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        typo_seq, correct_seq = self.data[idx]
        return torch.tensor(typo_seq), torch.tensor(correct_seq)


def collate_sequences(batch):

    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    return sequences_padded, labels_padded


def load_dataset(word_list, train_ratio=0.6, val_ratio=0.2, batch_size=50):

    dataset = TypoDataset(word_list) 
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_sequences)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)
    return train_loader, val_loader, test_loader


##########################GRU model########################################

class GRU(nn.Module):
    def __init__(self, name, hidden_size, vocab_size, n_layers=1):
        super(GRU, self).__init__()
        self.name = name
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        self.ident = torch.eye(vocab_size)
        self.gru = nn.GRU(vocab_size, hidden_size, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden=None):
        x = self.ident[x]  # One-hot encoding
        output, hidden = self.gru(x, hidden)
        output = self.fc(output)
        return output


##########################training & evaluation functions ############################

def evaluate_model(model, loader, criterion, device):

    model.eval()

    total_loss, total_errors, total_chars = 0, 0, 0
    with torch.no_grad():

        for typo_batch, correct_batch in loader:

            typo_batch, correct_batch = typo_batch.to(device), correct_batch.to(device)

            outputs = model(typo_batch)
            loss = criterion(outputs.view(-1, model.vocab_size), correct_batch.view(-1))

            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=-1)
            mask = correct_batch != 0

            total_errors += (predictions[mask] != correct_batch[mask]).sum().item()
            total_chars += mask.sum().item()

    return total_loss / len(loader), 1 - (total_errors / total_chars)


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, checkpoint_dir="gru_checkpoints"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

     # Arrays to store training/validation metrics
    train_loss = np.zeros(num_epochs)
    train_acc = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)

    for epoch in range(num_epochs):

        model.train()

        total_loss, total_errors, total_chars = 0, 0, 0

        for typo_batch, correct_batch in train_loader:

            typo_batch, correct_batch = typo_batch.to(device), correct_batch.to(device)
            optimizer.zero_grad()
            outputs = model(typo_batch)
            loss = criterion(outputs.view(-1, model.vocab_size), correct_batch.view(-1))
            predictions = torch.argmax(outputs, dim=-1)
            mask = correct_batch != 0

            total_errors += (predictions[mask] != correct_batch[mask]).sum().item()
            total_chars += mask.sum().item()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = 1 - (total_errors / total_chars)
        avg_val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        train_loss[epoch] = avg_train_loss
        train_acc[epoch] = train_accuracy
        val_loss[epoch] = avg_val_loss
        val_acc[epoch] = val_accuracy

        #save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss[epoch],
            'val_loss': val_loss[epoch],
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    print("Training complete!")

    #save models
    model_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), model_path)

    # save logs
    np.savetxt(os.path.join(checkpoint_dir, "train_acc.csv"), train_acc)
    np.savetxt(os.path.join(checkpoint_dir, "val_acc.csv"), val_acc)
    np.savetxt(os.path.join(checkpoint_dir, "train_loss.csv"), train_loss)
    np.savetxt(os.path.join(checkpoint_dir, "val_loss.csv"), val_loss)


############################ train and save gru ###################################

def main():
    word_list = get_random_words("words.txt", 1000)  # Adjust the number of words as needed

    train_loader, val_loader, test_loader = load_dataset(word_list, batch_size=50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRU(name="GRU", hidden_size=128, vocab_size=27, n_layers=2)
    train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, device=device, checkpoint_dir="gru_checkpoints")

    test_loss, test_accuracy = evaluate_model(model, test_loader, nn.CrossEntropyLoss(ignore_index=0), device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Plot training and validation curves
    plot_training_curve("gru_checkpoints")

    #example input and output
    example_input = "hlelo"
    example_input_seq = word_to_seq(example_input)
    example_input_tensor = torch.tensor(example_input_seq).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        model.eval()
        output = model(example_input_tensor)
        predicted_seq = torch.argmax(output, dim=-1).squeeze().cpu().numpy()

        predicted_word = ''.join([chr(i + 96) for i in predicted_seq if i > 0])  # Convert to letters
        print(f"Input: {example_input} | Predicted: {predicted_word}")

if __name__ == "__main__":
    main()
