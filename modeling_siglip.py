from typing import Optional, Tuple
import torch
import torch.nn as nn


class SiglipVisionConfig:

    def __init__(
            self,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels=3,
            image_size=224,
            patch_size=16,
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            num_image_tokens: int=None,
            **kwargs
    ):
        
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens
        

class SiglipMLP(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Intermediate_size]
        hidden_states = self.fc1(hidden_states)
        # [Batch_size, Num_patches, Intermediate_size] -> [Batch_size, Num_patches, Embed_dim]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Embed_dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states
        


class SiglipAttention(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5          # Equivalent to 1/sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        # Projection layers to transform input embeddings into queries, keys, and values
        self.k_proj   = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj   = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj   = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)


    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.size()

        # These projections map the input sequence to different learned spaces for query, key, and value
        # Input shape:  [Batch_size, Num_patches, Embed_dim]
        # Output shape: [Batch_size, Num_patches, Embed_dim]
        # The output shape does not change because the internal operation made in the linear layer is output = input * weight^T + bias
        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape and transpose the query, key, and value tensors
        # Input shape: [Batch_size, Num_patches, Embed_dim]
        # New shape after view: (batch_size, seq_len, num_heads, head_dim)
        # Transpose to: (batch_size, num_heads, seq_len, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        key_states   = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        # Compute scaled dot-product attention scores
        # query_states @ key_states.transpose: (batch_size, num_heads, seq_len, seq_len)
        # [batch_size, num_heads, seq_len, head_dim] @ [batch_size, num_heads, head_dim, seq_len] -> [batch_size, num_heads, seq_len, seq_len]
        # Scale the dot-product by sqrt(head_dim) to stabilize gradients
        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        
        # Apply softmax to attention weights to get probabilities, softmax along the last dimension (seq_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # Compute the attention output by applying the attention weights to the value states
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # Reshape and transpose attention output back to original input shape
        # Transpose: (batch_size, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1,2).contiguous()
        # Reshape to merge the heads back together: (batch_size, seq_len, embed_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # Apply a final linear projection to the attention output
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights

class SiglipEncoderLayer(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.embed_dim = config.hidden_state
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)

        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Layer Norm -> Attention layer -> Layer Norm -> MLP
        # Residual -----------------> Residual ----> Residual
        # Input shape: [Batch_size, Num_patches, Embed_dim] -> Output shape: [Batch_size, Num_patches, Embed_dim]
        # This module does not modify the shape of the input tensor.
        # This module will be repeated "num_hidden_layers" times.

        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )


    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds

        for encoder_layers in self.layers:
            # [Batch_size, Num_patches, Embed_dim] -> [Batch_size, Num_patches, Embed_dim]
            hidden_states = encoder_layers(hidden_states)
        
        return hidden_states

class SiglipVisionEmbeddings(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size


        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid"
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        
        # Learned vector of size (num_patches, embed_dim). 
        # Later it will be added to the patch embeddings, that is why it must have the same dimension.
        # The positional embeddings are learned in training
        self.position_embeddings = nn.Embedding(num_embeddings=self.num_positions, embedding_dim=self.embed_dim)

        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape # [Batch_size, Num_channels, Height, Width]
        
        if not (height == self.image_size and width == self.image_size):
            raise ValueError(f"Input image size ({height}*{width}) doesn't match the expected image size ({self.image_size}*{self.image_size}).")
        
        # The output of the patch embeddings layer is of shape [Batch_size, Embed_dim, Num_Patches_H, Num_Patches_W]
        patch_embeds = self.patch_embeddings(pixel_values)
        # We flatten the last two dimensions to get a tensor of shape [Batch_size, Embed_dim, Num_Patches_H * Num_Patches_W]
        embeddings = patch_embeds.flatten(2)
        # embeddings = embeddings.view(batch_size, self.num_patches, -1)  # Equivalent to embeddings = embeddings.flatten(2)
        # Shape: [Batch_size, Num_Patches_H * Num_Patches_W, Embed_dim]
        embeddings = embeddings.transpose(1, 2)
        # Adding the position embeddings to the patch embeddings
        embeddings = embeddings + self.position_embeddings(self.position_ids)
        # [Batch_size, Num_Patches_H * Num_Patches_W, Embed_dim]
        return embeddings
        
            

class SiglipVisionTransformer(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        embed_dim = config.hidden_size
        
        # Extracts the image patches using a convolution layer, flattens them and adds positional encodings.
        self.embeddings = SiglipVisionEmbeddings(config)
        # Transformer encoder
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.post_layernorm(self.encoder(self.embeddings(pixel_values)))


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.vision_model = SiglipVisionTransformer(config)


    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor]:
        # [Batch_size, Channels, Height, Width] -> [Batch_size, Num_patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)
        