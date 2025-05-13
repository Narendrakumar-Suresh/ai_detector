import streamlit as st
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
import nltk
import pickle
nltk.download('punkt_tab')

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

def tokenize(text):
    """Tokenize text using word_tokenize."""
    return word_tokenize(text.lower())

def encode(tokens):
    """Convert tokens to numerical indices using vocab."""
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=3, dropout=0.3):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        x = hidden[-1]  # Get the last hidden state from LSTM
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(1)


device = torch.device("cpu")
vocab_size = len(vocab)
model = LSTM(vocab_size).to(device)

model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

st.title("AI-Generated Text Detector")
st.write("Upload a text file or paste your text below:")

text_input = st.text_area("Enter Text")
uploaded_file = st.file_uploader("Or Upload a .txt File", type=["txt"])

if uploaded_file:
    text_input = uploaded_file.read().decode("utf-8")

if st.button("Detect"):
    if not text_input.strip():
        st.warning("Please enter or upload some text.")
    else:
        tokens = tokenize(text_input)
        ids = encode(tokens)

        if len(ids) == 0:
            st.error("The input contains no recognizable tokens. Try entering more meaningful text.")
        else:
            input_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output).item()
                percentage = round(prob * 100, 2)

            st.success(f"AI-Generated Probability: **{percentage}%**")
