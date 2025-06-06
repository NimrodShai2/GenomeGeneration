import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from preprocess_fasta import extract_chunks_from_fasta, build_kmer_vocab, tokenize_chunks
import json
import os
from tqdm import tqdm

# -------------------------------
# Config
# -------------------------------
FASTA_FILE = "P_Dorei_Genomes/P_Dorei_genome_sequence.fasta"
CHUNK_SIZE = 1000
STRIDE = 500
K = 6
LATENT_DIM = 32
EMBED_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "saved"
os.makedirs(SAVE_DIR, exist_ok=True)


# -------------------------------
# Dataset
# -------------------------------
class KmerDataset(Dataset):
    """Dataset for k-mer tokenized DNA sequences."""

    def __init__(self, tokenized_sequences):
        self.data = tokenized_sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        return x


# -------------------------------
# VAE Model
# -------------------------------
class VAE(nn.Module):
    """
    Variational Autoencoder for DNA sequences using k-mer embeddings.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim, seq_len):
        super(VAE, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim)  # +1 for unknown
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim * seq_len, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # mean of the latent space
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # log variance of the latent space
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim * seq_len),
        )
        self.output = nn.Linear(embed_dim, vocab_size)  # token-by-token

        self.seq_len = seq_len  # Length of the sequence after k-mer extraction
        self.embed_dim = embed_dim  # Dimension of the embedding space

    def encode(self, x):
        """Encode input sequence into latent space parameters."""
        embedded = self.embedding(x).view(x.size(0), -1)
        h = self.encoder(embedded)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from the latent space."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent space sample back to sequence."""
        out = self.decoder_fc(z)
        out = out.view(-1, self.seq_len, self.embed_dim)
        return self.output(out)

    def forward(self, x):
        """Forward pass through the VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# -------------------------------
# Loss function
# -------------------------------
def loss_fn(recon_logits, x, mu, logvar):
    recon_loss = nn.CrossEntropyLoss()(recon_logits.view(-1, recon_logits.size(-1)), x.view(-1))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_loss


# -------------------------------
# Training loop
# -------------------------------
def train():
    print("Preprocessing...", flush=True)
    chunks = extract_chunks_from_fasta(FASTA_FILE, chunk_size=CHUNK_SIZE, stride=STRIDE)
    vocab = build_kmer_vocab(K)
    tokenized = tokenize_chunks(chunks, K, vocab)

    dataset = KmerDataset(tokenized)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = VAE(vocab_size=len(vocab), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                latent_dim=LATENT_DIM, seq_len=CHUNK_SIZE - K + 1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Training...", flush=True)
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(loader):
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon_logits, mu, logvar = model(batch)
            loss = loss_fn(recon_logits, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}", flush=True)

    print("Saving model and vocab...", flush=True)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pt"))
    with open(os.path.join(SAVE_DIR, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:
        json.dump({
            "chunk_size": CHUNK_SIZE,
            "k": K,
            "latent_dim": LATENT_DIM,
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "seq_len": CHUNK_SIZE - K + 1,
        }, f)


if __name__ == "__main__":
    train()
