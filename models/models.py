import math
import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, frame_size, channel_num=5, output_size=4, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(channel_num, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.linear1 = nn.Linear(hidden_size*num_layers, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        h = self.gru(x)[1].permute(1, 0, 2)
        h = h.reshape(h.size(0), -1)
        linear_out = self.linear1(h)
        linear_out = self.batch_norm1(linear_out)
        linear_out = torch.relu(linear_out)
        linear_out = self.dropout1(linear_out)
        
        linear_out = self.linear2(linear_out)
        linear_out = self.batch_norm2(linear_out)
        linear_out = torch.relu(linear_out)
        linear_out = self.dropout2(linear_out)
        
        out = self.linear3(linear_out)
        return out


class Autoencoder(nn.Module):
    def __init__(self, frame_size=32, bottleneck_dim=40):
        super(Autoencoder, self).__init__()

        ch1 = bottleneck_dim // 2  # 20 when bottleneck_dim=40
        ch2 = bottleneck_dim  # 40
        fc_dim = bottleneck_dim * 12  # 480 when bottleneck_dim=40
        input_channels = 5
        input_length = frame_size

        # === Encoder ===
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(
                input_channels,
                ch1,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 32 -> 16
            nn.Conv1d(
                ch1,
                ch2,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 16 -> 8
        )

        # After conv: [B, ch2, 8] → flattened: ch2 * 8
        self.encoder_fc = nn.Sequential(
            nn.Linear(ch2 * 8, fc_dim),
            nn.ReLU(True),
            nn.Linear(fc_dim, bottleneck_dim),  # bottleneck
        )

        # === Decoder ===
        self.decoder_fc = nn.Sequential(
            nn.Linear(bottleneck_dim, fc_dim),
            nn.ReLU(True),
            nn.Linear(fc_dim, ch2 * 8), 
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(
                ch2, ch1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 8→16
            nn.ReLU(True),
            nn.ConvTranspose1d(
                ch1,
                input_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # 16→32
            # For precise length restoration (ConvTranspose1d can sometimes add +1)
            nn.Linear(input_length, input_length),
        )

    def forward(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.encoder_fc(x)
        x = self.decoder_fc(x)
        x = x.view(x.size(0), -1, 8)  # Use -1 instead of fixed bottleneck_dim
        x = self.decoder_conv(x)
        return x


class MultiScaleAttentionCNN(nn.Module):
    def __init__(
        self, frame_size, channel_num=5, hidden_size=128, output_size=4, dropout=0.3
    ):
        super(MultiScaleAttentionCNN, self).__init__()
        self.frame_size = frame_size
        
        # Multi-scale feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv1d(channel_num, hidden_size // 4, kernel_size=3, padding=1, padding_mode="circular"),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(channel_num, hidden_size // 4, kernel_size=5, padding=2, padding_mode="circular"),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(channel_num, hidden_size // 4, kernel_size=7, padding=3, padding_mode="circular"),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )
        
        # Residual block
        self.residual_conv = nn.Sequential(
            nn.Conv1d(hidden_size // 4 * 3, hidden_size, kernel_size=1),
            nn.BatchNorm1d(hidden_size),
        )
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, padding_mode="circular"),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, padding_mode="circular"),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )
        
        # Global attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Calculate the size after convolutions
        conv_output_size = hidden_size * frame_size
        
        # Calculate combined feature size: fc3 output (hidden_size) + attention features (hidden_size)
        combined_features_size = hidden_size + hidden_size
        
        # Enhanced MLP with residual connections
        self.fc1 = nn.Linear(conv_output_size, hidden_size * 4)
        self.bn1 = nn.BatchNorm1d(hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(combined_features_size, output_size)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        # Input shape: [batch, time_steps, channels]
        x = x.permute(0, 2, 1)  # [batch, channels, time_steps]
        
        # Multi-scale feature extraction
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        
        # Concatenate multi-scale features
        x = torch.cat([x1, x2, x3], dim=1)
        
        # Residual connection
        residual = self.residual_conv(x)
        x = self.conv_block(residual) + residual
        
        # Global attention pooling
        batch_size, channels, time_steps = x.shape
        
        # Reshape for attention: [batch_size, time_steps, channels]
        x_permuted = x.permute(0, 2, 1)
        
        # Compute attention weights: [batch_size, time_steps, 1]
        attention_weights = self.attention(x_permuted.reshape(-1, channels))
        attention_weights = attention_weights.view(batch_size, time_steps, 1)
        
        # Apply attention: [batch_size, channels, time_steps] x [batch_size, time_steps, 1] = [batch_size, channels, 1]
        x_pooled = torch.bmm(x, attention_weights).squeeze(-1)  # [batch_size, channels]
        
        # Global average pooling
        x_avg = torch.mean(x, dim=2)  # [batch_size, channels]
        
        # Combine attention and average pooling
        x_combined = x_pooled + x_avg
        
        # Flatten convolutional features for MLP
        x_flat_conv = x.view(batch_size, -1)
        
        # Enhanced MLP with residual connections
        x_fc = self.fc1(x_flat_conv)
        x_fc = self.bn1(x_fc)
        x_fc = self.relu(x_fc)
        x_fc = self.dropout(x_fc)
        
        x_fc = self.fc2(x_fc)
        x_fc = self.bn2(x_fc)
        x_fc = self.relu(x_fc)
        x_fc = self.dropout(x_fc)
        
        x_fc = self.fc3(x_fc)
        x_fc = self.bn3(x_fc)
        x_fc = self.relu(x_fc)
        x_fc = self.dropout(x_fc)
        
        # Combine with attention features
        x_combined_features = torch.cat([x_fc, x_combined], dim=1)
        output = self.fc4(x_combined_features)
        
        return output


class MLP(nn.Module):
    def __init__(self, frame_size, channel_num=5, hidden_size=128, output_size=4, num_layers=3, dropout=0.3):
        super(MLP, self).__init__()
        
        layers = []
        input_dim = frame_size * channel_num
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU(True))
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU(True))
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.network(x)
        return x


class TransformerClassifier(nn.Module):
    def __init__(
        self, frame_size, channel_num=5, hidden_size=128, output_size=4,
        num_layers=3, num_heads=8, dropout=0.1
    ):
        super(TransformerClassifier, self).__init__()
        self.frame_size = frame_size
        self.hidden_size = hidden_size
        
        # Input projection to hidden dimension
        self.input_projection = nn.Linear(channel_num, hidden_size)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_size, dropout, max_len=frame_size)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        # Input shape: [batch, time_steps, channels]
        batch_size = x.size(0)
        
        # Project input to hidden dimension
        x = self.input_projection(x)  # [batch, time_steps, hidden_size]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoder expects [batch, seq_len, hidden_size]
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # [batch, time_steps, hidden_size]
        
        # Global average pooling over time dimension
        x = torch.mean(x, dim=1)  # [batch, hidden_size]
        
        # Classification
        x = self.classifier(x)  # [batch, output_size]
        
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class CNN(nn.Module):
    def __init__(
        self, frame_size, channel_num=5, hidden_size=64, output_size=4, dropout=0.3
    ):
        super(CNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(channel_num, hidden_size, kernel_size=7, padding=3, padding_mode="circular"),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=5, padding=2, padding_mode="circular"),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv1d(hidden_size * 2, hidden_size * 4, kernel_size=3, padding=1, padding_mode="circular"),
            nn.BatchNorm1d(hidden_size * 4),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )
        
        # Calculate the size after convolutions and pooling
        # After two max pools: frame_size -> frame_size/2 -> frame_size/4
        conv_output_size = (hidden_size * 4) * (frame_size // 4)
        
        self.classifier = nn.Sequential(
            nn.Linear(conv_output_size, hidden_size * 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        # Input shape: [batch, time_steps, channels]
        x = x.permute(0, 2, 1)  # [batch, channels, time_steps]
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Flatten for classification
        x = x.view(x.size(0), -1)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x