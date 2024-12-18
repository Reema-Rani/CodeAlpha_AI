import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pretty_midi

# Step 1: Preprocess MIDI Data
def preprocess_midi(file_path):
    midi = pretty_midi.PrettyMIDI(file_path)
    notes = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                notes.append(note.pitch)  
    return np.array(notes)

# Step 2: Create Dataset
class MusicDataset(Dataset):
    def __init__(self, sequences, sequence_length):
        self.sequences = sequences
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.sequences) - self.sequence_length

    def __getitem__(self, idx):
        return (
            self.sequences[idx:idx + self.sequence_length],  # Input sequence
            self.sequences[idx + 1:idx + self.sequence_length + 1]  # Target sequence
        )

# Step 3: Define the Model
class MusicGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(MusicGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden

# Step 4: Train the Model
def train(model, dataloader, criterion, optimizer, epochs, vocab_size, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# Step 5: Generate Music
def generate(model, seed_sequence, length, vocab_size, device):
    model.eval()
    generated = seed_sequence.tolist()
    input_seq = torch.tensor(seed_sequence, device=device).unsqueeze(0)
    hidden = None
    for _ in range(length):
        with torch.no_grad():
            outputs, hidden = model(input_seq, hidden)
            next_note = torch.argmax(outputs[:, -1, :], dim=-1).item()
            generated.append(next_note)
            input_seq = torch.tensor([[next_note]], device=device)
    return generated

# Step 6: Save Generated Music to MIDI
def save_to_midi(notes, output_path):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    start = 0
    for pitch in notes:
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start + 0.5)
        instrument.notes.append(note)
        start += 0.5
    midi.instruments.append(instrument)
    midi.write(output_path)

# Main Function
if __name__ == "__main__":
    # Hyperparameters
    sequence_length = 50
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    epochs = 20
    batch_size = 32
    learning_rate = 0.001

    # Load and preprocess MIDI data
    midi_file = "example.mid"  # Replace with your MIDI file
    data = preprocess_midi(midi_file)
    vocab_size = len(set(data))  # Unique number of notes
    dataset = MusicDataset(data, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicGenerator(vocab_size, embedding_dim, hidden_dim, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, dataloader, criterion, optimizer, epochs, vocab_size, device)

    # Generate music
    seed_sequence = data[:sequence_length]
    generated_notes = generate(model, seed_sequence, length=200, vocab_size=vocab_size, device=device)

    # Save the generated sequence to MIDI
    save_to_midi(generated_notes, "generated_music.mid")
    print("Generated music saved as 'generated_music.mid'")
