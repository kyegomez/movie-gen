
import torch
import torch.nn as nn
from loguru import logger


def get_num_heads(in_channels: int) -> int:
    """
    Calculate the appropriate number of attention heads.
    Args:
        in_channels (int): The number of input channels.
    Returns:
        int: The number of attention heads.
    """
    num_heads = min(in_channels, 8)
    while in_channels % num_heads != 0 and num_heads > 1:
        num_heads -= 1
    if in_channels % num_heads != 0:
        num_heads = 1  # Fallback to 1 if no divisor is found
    return num_heads


class ResNetBlock(nn.Module):
    """
    Residual block with spatial and temporal convolutions.
    This block implements the ResNet block described in Section 3.1.1 of the paper,
    where after each 2D spatial convolution, a 1D temporal convolution is added to 'inflate' the model.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(ResNetBlock, self).__init__()
        # 2D spatial convolution
        self.conv_spatial = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )
        # 1D temporal convolution with symmetrical replicate padding
        self.conv_temporal = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            padding_mode="replicate",
        )
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(
                in_channels, out_channels, kernel_size=1
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet block.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, T, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, T, H, W)
        """
        identity = self.shortcut(x)
        out = self.relu(self.conv_spatial(x))  # Spatial convolution
        out = self.relu(
            self.conv_temporal(out)
        )  # Temporal convolution with symmetrical replicate padding
        out += identity  # Residual connection
        out = self.relu(out)
        return out


class SpatialSelfAttention(nn.Module):
    """
    Spatial Self-Attention Block.
    Implements spatial attention as described in Section 3.1.1, where spatial attention is applied after spatial convolutions.
    """

    def __init__(self, in_channels: int):
        super(SpatialSelfAttention, self).__init__()
        self.in_channels = in_channels
        num_heads = self.get_num_heads(in_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels, num_heads=num_heads
        )

    def get_num_heads(self, in_channels: int) -> int:
        """
        Calculate the appropriate number of attention heads.
        Args:
            in_channels (int): The number of input channels.
        Returns:
            int: The number of attention heads.
        """
        num_heads = min(in_channels, 8)
        while in_channels % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        if in_channels % num_heads != 0:
            num_heads = 1  # Fallback to 1 if no divisor is found
        return num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the spatial self-attention block.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        x_reshaped = x.permute(0, 2, 3, 4, 1).reshape(
            B * T, H * W, C
        )  # (B*T, H*W, C)
        attn_output, _ = self.attention(
            x_reshaped, x_reshaped, x_reshaped
        )
        attn_output = attn_output.view(B, T, H, W, C).permute(
            0, 4, 1, 2, 3
        )
        return x + attn_output  # Residual connection


class TemporalSelfAttention(nn.Module):
    """
    Temporal Self-Attention Block.
    Implements temporal attention as described in Section 3.1.1, where temporal attention is applied after temporal convolutions.
    """

    def __init__(self, in_channels: int):
        super(TemporalSelfAttention, self).__init__()
        self.in_channels = in_channels
        heads = get_num_heads(in_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels, num_heads=heads
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the temporal self-attention block.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        x_reshaped = x.permute(0, 3, 4, 2, 1).reshape(
            B * H * W, T, C
        )  # (B*H*W, T, C)
        attn_output, _ = self.attention(
            x_reshaped, x_reshaped, x_reshaped
        )
        attn_output = attn_output.view(B, H, W, T, C).permute(
            0, 4, 3, 1, 2
        )
        return x + attn_output  # Residual connection


class TAEEncoder(nn.Module):
    """
    Temporal Autoencoder Encoder.
    Compresses input pixel space video V of shape (B, T0, 3, H0, W0) to latent X of shape (B, C, T, H, W).
    As described in Section 3.1.1, the encoder compresses the input 8x across each spatio-temporal dimension.
    """

    def __init__(
        self, in_channels: int = 3, latent_channels: int = 16
    ):
        super(TAEEncoder, self).__init__()
        self.latent_channels = latent_channels
        # Initial convolution to increase channel dimension
        self.initial_conv = nn.Conv3d(
            in_channels, 64, kernel_size=3, padding=1
        )
        current_channels = 64
        # Downsampling layers to achieve 8x compression
        self.downsampling_layers = nn.ModuleList()
        for _ in range(3):  # 8x compression over T, H, W
            self.downsampling_layers.append(
                nn.Sequential(
                    ResNetBlock(current_channels, current_channels),
                    SpatialSelfAttention(current_channels),
                    TemporalSelfAttention(current_channels),
                    # Temporal downsampling via strided convolution with stride of 2
                    nn.Conv3d(
                        current_channels,
                        current_channels * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                )
            )
            current_channels *= 2
        # Final block to get latent representation
        self.final_block = nn.Sequential(
            ResNetBlock(current_channels, latent_channels),
            SpatialSelfAttention(latent_channels),
            TemporalSelfAttention(latent_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TAE encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (B, T0, 3, H0, W0)
        Returns:
            torch.Tensor: Output tensor of shape (B, C, T, H, W)
        """
        logger.debug("Encoding input of shape {}", x.shape)
        B, T0, C_in, H0, W0 = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, C_in, T0, H0, W0)
        x = self.initial_conv(x)
        for layer in self.downsampling_layers:
            x = layer(x)
            logger.debug(
                "After downsampling layer, shape: {}", x.shape
            )
        x = self.final_block(x)
        logger.debug("Final latent shape: {}", x.shape)
        return x  # (B, latent_channels, T, H, W)


class TAEDecoder(nn.Module):
    """
    Temporal Autoencoder Decoder.
    Decodes latent X of shape (B, C, T, H, W) to reconstructed video V_hat of shape (B, T0, 3, H0, W0).
    Upsampling is performed via nearest-neighbor interpolation followed by convolution as described in Section 3.1.1.
    """

    def __init__(
        self, out_channels: int = 3, latent_channels: int = 16
    ):
        super(TAEDecoder, self).__init__()
        self.latent_channels = latent_channels
        current_channels = latent_channels
        # Initial block
        self.initial_block = nn.Sequential(
            ResNetBlock(current_channels, current_channels),
            SpatialSelfAttention(current_channels),
            TemporalSelfAttention(current_channels),
        )
        # Upsampling layers to reverse the compression
        self.upsampling_layers = nn.ModuleList()
        for _ in range(3):  # Reverse of encoder
            self.upsampling_layers.append(
                nn.Sequential(
                    # Upsampling via nearest-neighbor interpolation
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    ResNetBlock(
                        current_channels, current_channels // 2
                    ),
                    SpatialSelfAttention(current_channels // 2),
                    TemporalSelfAttention(current_channels // 2),
                )
            )
            current_channels = current_channels // 2
        # Final convolution to get output image
        self.final_conv = nn.Conv3d(
            current_channels, out_channels, kernel_size=3, padding=1
        )
        self.sigmoid = (
            nn.Sigmoid()
        )  # Assuming input images are normalized between 0 and 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TAE decoder.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (B, T0, 3, H0, W0)
        """
        logger.debug("Decoding latent of shape {}", x.shape)
        x = self.initial_block(x)
        for layer in self.upsampling_layers:
            x = layer(x)
            logger.debug("After upsampling layer, shape: {}", x.shape)
        x = self.final_conv(x)
        x = self.sigmoid(x)
        x = x.permute(0, 2, 1, 3, 4)  # (B, T0, C_out, H0, W0)
        logger.debug(
            "Reconstructed output shape before trimming: {}", x.shape
        )
        return x


class TemporalAutoencoder(nn.Module):
    """
    Temporal Autoencoder (TAE) model.
    This model combines the encoder and decoder, and handles variable-length videos as described in Section 3.1.1.
    """

    def __init__(
        self, in_channels: int = 3, latent_channels: int = 16
    ):
        super(TemporalAutoencoder, self).__init__()
        self.encoder = TAEEncoder(in_channels, latent_channels)
        self.decoder = TAEDecoder(in_channels, latent_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TAE.
        Args:
            x (torch.Tensor): Input tensor of shape (B, T0, 3, H0, W0)
        Returns:
            torch.Tensor: Reconstructed tensor of shape (B, T0, 3, H0, W0)
        """
        logger.debug("Starting encoding")
        latent = self.encoder(x)
        logger.debug("Encoding complete")
        logger.debug("Starting decoding")
        recon = self.decoder(latent)
        logger.debug("Decoding complete")
        # Discard spurious frames as shown in Figure 4 of the paper
        T0 = x.shape[1]
        T_decoded = recon.shape[1]
        if T_decoded > T0:
            recon = recon[:, :T0]
            logger.debug(
                "Discarded spurious frames, final output shape: {}",
                recon.shape,
            )
        else:
            logger.debug(
                "No spurious frames to discard, final output shape: {}",
                recon.shape,
            )
        return recon


# import torch
# from loguru import logger

# # Assuming the TemporalAutoencoder and its dependencies have been defined/imported as provided above

# def test_temporal_autoencoder():
#     """
#     Test the TemporalAutoencoder model with a dummy input tensor.
#     This function creates a random input tensor representing a batch of videos,
#     passes it through the model, and prints out the input and output shapes.
#     """
#     # Set the logger to display debug messages
#     logger.add(lambda msg: print(msg, end=''))

#     # Instantiate the model
#     model = TemporalAutoencoder(in_channels=3, latent_channels=16)

#     # Create a dummy input tensor representing a batch of videos
#     # Batch size B=2, T0=16 frames, 3 channels (RGB), H0=64, W0=64
#     B, T0, C_in, H0, W0 = 1, 16, 3, 64, 64
#     x = torch.randn(B, T0, C_in, H0, W0)

#     # Forward pass through the model
#     recon = model(x)

#     # Print the shapes
#     print(f"Input shape: {x.shape}")
#     print(f"Reconstructed output shape: {recon.shape}")

# if __name__ == "__main__":
#     test_temporal_autoencoder()
