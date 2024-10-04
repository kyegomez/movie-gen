
import torch
from loguru import logger
from movie_gen.tae import TemporalAutoencoder

# Assuming the TemporalAutoencoder and its dependencies have been defined/imported as provided above

def test_temporal_autoencoder():
    """
    Test the TemporalAutoencoder model with a dummy input tensor.
    This function creates a random input tensor representing a batch of videos,
    passes it through the model, and prints out the input and output shapes.
    """
    # Set the logger to display debug messages
    logger.add(lambda msg: print(msg, end=''))

    # Instantiate the model
    model = TemporalAutoencoder(in_channels=3, latent_channels=16)

    # Create a dummy input tensor representing a batch of videos
    # Batch size B=2, T0=16 frames, 3 channels (RGB), H0=64, W0=64
    B, T0, C_in, H0, W0 = 1, 16, 3, 64, 64
    x = torch.randn(B, T0, C_in, H0, W0)

    # Forward pass through the model
    recon = model(x)

    # Print the shapes
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed output shape: {recon.shape}")

if __name__ == "__main__":
    test_temporal_autoencoder()
