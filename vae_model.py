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


class ConvVAE(nn.Module):
    """Convolutional Variational Autoencoder for DNA sequences."""

    def __init__(self, vocab_size, embed_dim, latent_dim, seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim


        # Embedding layer
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim)

        # Encoder: Conv1d -> downsampling -> flatten -> linear to mu/logvar
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )

        # Calculate conv output size
        self.conv_out_len = seq_len // 4  # due to two stride-2 layers
        self.conv_out_dim = 256 * self.conv_out_len

        self.fc_mu = nn.Linear(self.conv_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.conv_out_dim, latent_dim)

        # Decoder: linear -> unflatten -> ConvTranspose1d
        self.decoder_fc = nn.Linear(latent_dim, self.conv_out_dim)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, embed_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def encode(self, x):
        x = self.embedding(x)  # [B, L, E]
        x = x.permute(0, 2, 1)  # [B, E, L] for Conv1d
        x = self.encoder_conv(x)  # [B, 256, L/4]
        x = x.view(x.size(0), -1)  # flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)  # [B, conv_out_dim]
        x = x.view(z.size(0), 256, self.conv_out_len)  # unflatten
        x = self.decoder_conv(x)  # [B, embed_dim, seq_len]
        x = x.permute(0, 2, 1)  # [B, seq_len, embed_dim]
        return self.output_layer(x)  # [B, seq_len, vocab_size]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
