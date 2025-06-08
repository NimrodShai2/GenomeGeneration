from vae_model import ConvVAE, VAE


def get_model(model_type, vocab_size, embed_dim, hidden_dim, latent_dim, seq_len):
    """Factory function to create a VAE model."""
    if model_type == "conv":
        return ConvVAE(vocab_size, embed_dim, latent_dim, seq_len)
    else:
        return VAE(vocab_size, embed_dim, hidden_dim, latent_dim, seq_len)
