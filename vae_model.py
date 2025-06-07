import torch
import torch.nn as nn


class VAE(nn.Module):
    """Variational Autoencoder for DNA sequences."""

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

    def encode(self, x):
        """Encode input sequence into latent space."""
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
        """Decode latent vector back to sequence."""
        out = self.decoder_fc(z)
        out = out.view(-1, self.seq_len, self.embed_dim)
        return self.output(out)

    def forward(self, x):
        """Forward pass through the VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
