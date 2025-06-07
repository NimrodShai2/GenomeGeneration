import torch
import torch.nn as nn
import json
import argparse
from tqdm import tqdm


# -------------------------------
# Reconstruct VAE Model
# -------------------------------
class VAE(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim, seq_len):
        super(VAE, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim * seq_len, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim * seq_len),
        )
        self.output = nn.Linear(embed_dim, vocab_size)
        self.seq_len = seq_len
        self.embed_dim = embed_dim

    def decode(self, z):
        out = self.decoder_fc(z)
        out = out.view(-1, self.seq_len, self.embed_dim)
        return self.output(out)


# -------------------------------
# Reconstruct DNA from tokens
# -------------------------------
def tokens_to_dna(tokens, inv_vocab, k):
    kmers = [inv_vocab.get(str(tok), 'N' * k) for tok in tokens]
    seq = kmers[0]
    for kmer in kmers[1:]:
        seq += kmer[-1]
    return seq


# -------------------------------
# Sample from logits
# -------------------------------
def sample_softmax(logits):
    probs = torch.softmax(logits, dim=-1)  # [seq_len, vocab_size]
    sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [seq_len]
    return sampled_indices


# -------------------------------
# Main Generation Function
# -------------------------------
def generate(num_genomes, genome_length, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config and vocab
    with open("saved/config.json") as f:
        cfg = json.load(f)
    with open("saved/vocab.json") as f:
        vocab = json.load(f)

    inv_vocab = {str(v): k for k, v in vocab.items()}
    k = cfg["k"]
    chunk_len = cfg["seq_len"] + k - 1
    chunks_per_genome = (genome_length + chunk_len - 1) // chunk_len

    model = VAE(
        vocab_size=len(vocab),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        seq_len=cfg["seq_len"]
    ).to(device)
    model.load_state_dict(torch.load("saved/model.pt", map_location=device))
    model.eval()

    with open(output_path, "w") as f:
        for i in tqdm(range(num_genomes), desc="Generating genomes"):
            dna_parts = []
            for _ in range(chunks_per_genome):
                with torch.no_grad():
                    z = torch.randn((1, cfg["latent_dim"])).to(device)
                    logits = model.decode(z).squeeze(0)  # [seq_len, vocab_size]
                    tokens = sample_softmax(logits).cpu().tolist()
                    chunk_dna = tokens_to_dna(tokens, inv_vocab, k)
                    dna_parts.append(chunk_dna)
            full_dna = ''.join(dna_parts)[:genome_length]
            f.write(f">genome_{i}\n{full_dna}\n")

    print(f"[✓] Generated {num_genomes} synthetic genomes (~{genome_length} bp each) to {output_path}")


# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_genomes", type=int, default=10, help="Number of synthetic genomes to generate")
    parser.add_argument("--length", type=int, default=50000, help="Target length per genome in base pairs")
    parser.add_argument("--output", type=str, default="generated.fasta", help="Output FASTA file path")
    args = parser.parse_args()
    generate(args.num_genomes, args.length, args.output)
