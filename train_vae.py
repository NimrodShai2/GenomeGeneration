import argparse
import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from preprocess_fasta import extract_chunks_from_fasta, build_kmer_vocab, tokenize_chunks
from vae_model import ConvVAE


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
        return torch.tensor(self.data[idx], dtype=torch.long)


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
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # ensure (chunk_size - k + 1) is divisible by 4
    adjusted_seq_len = (args.chunk_size - args.k + 1)
    remainder = adjusted_seq_len % 4
    if remainder != 0:
        adjustment = 4 - remainder
        args.chunk_size += adjustment
        print(
            f"[!] Adjusted chunk_size to {args.chunk_size} so that seq_len={args.chunk_size - args.k + 1} is divisible by 4.")

    print("Preprocessing...", flush=True)
    chunks = extract_chunks_from_fasta(args.fasta_file,
                                       chunk_size=args.chunk_size,
                                       stride=args.stride)
    vocab = build_kmer_vocab(args.k)
    tokenized = tokenize_chunks(chunks, args.k, vocab)

    dataset = KmerDataset(tokenized)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = ConvVAE(vocab_size=len(vocab),
                    embed_dim=args.embed_dim,
                    latent_dim=args.latent_dim,
                    seq_len=args.chunk_size - args.k + 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Training...", flush=True)
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_logits, mu, logvar = model(batch)
            loss = loss_fn(recon_logits, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{args.epochs}, Loss: {total_loss:.4f}", flush=True)

    print("Saving model and vocab...", flush=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, "model.pt"))
    with open(os.path.join(args.save_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    config = {
        "fasta_file": args.fasta_file,
        "chunk_size": args.chunk_size,
        "stride": args.stride,
        "k": args.k,
        "latent_dim": args.latent_dim,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "seq_len": args.chunk_size - args.k + 1,
        "vocab_size": len(vocab),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "device": str(device)
    }
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE on k-mer tokenized genome chunks")
    parser.add_argument("--fasta_file", type=str,
                        default="P_Dorei_Genomes/P_Dorei_genome_sequence.fasta",
                        help="Input multi-genome FASTA file")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Length of each DNA chunk (bp)")
    parser.add_argument("--stride", type=int, default=500,
                        help="Stride between chunks (bp)")
    parser.add_argument("--k", type=int, default=6,
                        help="k-mer size for tokenization")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="Dimensionality of latent space")
    parser.add_argument("--embed_dim", type=int, default=64,
                        help="Dimension of k-mer embeddings")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden layer size for encoder/decoder")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--save_dir", type=str, default="saved",
                        help="Directory to save model and config")
    args = parser.parse_args()
    train(args)
