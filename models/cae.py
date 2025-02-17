import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import timm  # Make sure to install timm: pip install timm
from enum import Enum

class ConvAutoencoder(nn.Module):
    def __init__(self, encoder_channels=[16, 32, 64], decoder_channels=None, use_batchnorm=True, use_dropout=False,
                 backbone: Optional[str] = None):
        """
        이미지 이상 탐지를 위한 Convolutional Autoencoder.

        Args:
            encoder_channels (list): Encoder에서 사용할 각 Conv 레이어의 출력 채널 수 (backbone 사용 안 할 때).
            decoder_channels (list, optional): Decoder 채널을 명시적으로 지정 (backbone 사용 안 할 때).
            use_batchnorm (bool): Batch Normalization 사용 여부.
            use_dropout (bool): Dropout 사용 여부 (encoder에만 적용).
            backbone (str, optional):  Pretrained backbone 모델 이름 (timm 사용). None이면 자체 encoder 사용.
        """
        super(ConvAutoencoder, self).__init__()
        self.backbone_name = backbone

        if backbone:
            # Timm을 이용한 pretrained encoder
            self.encoder = timm.create_model(backbone, pretrained=True, features_only=True)
            # Get the number of output channels from the last feature map of the backbone
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)  # Example input size, adjust as needed
                out_features = self.encoder(dummy_input)
                in_channels = out_features[-1].shape[1] # Use the last feature map's channels
                
            # Freeze the backbone parameters
            for param in self.encoder.parameters():
                 param.requires_grad = False

            # Determine decoder channels based on the encoder's output channels.
            encoder_channels = [of.shape[1] for of in out_features] # Get the channel counts from the backbone outputs.
            if decoder_channels is None:
                decoder_channels = list(reversed(encoder_channels[:-1])) + [3]

        else:
            # 자체 encoder (backbone 사용 안 함)
            input_channels = 3
            encoder_layers = []
            in_channels = input_channels
            for out_channels in encoder_channels:
                encoder_layers.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
                )
                if use_batchnorm:
                    encoder_layers.append(nn.BatchNorm2d(out_channels))
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                if use_dropout:
                    encoder_layers.append(nn.Dropout(0.25))
                in_channels = out_channels
            self.encoder = nn.Sequential(*encoder_layers)

            if decoder_channels is None:
                decoder_channels = list(reversed(encoder_channels[:-1])) + [3]


        # Decoder (backbone 사용 여부와 관계없이 동일한 구조 사용)
        decoder_layers = []

        for i, out_channels in enumerate(decoder_channels):
            if i < len(decoder_channels) - 1:
                decoder_layers.append(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
                )
                if use_batchnorm:
                    decoder_layers.append(nn.BatchNorm2d(out_channels))
                decoder_layers.append(nn.ReLU(inplace=True))
            else:  # 마지막 레이어
                decoder_layers.append(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
                )
                decoder_layers.append(nn.Sigmoid())  # [0, 1]
            in_channels = out_channels
        self.decoder = nn.Sequential(*decoder_layers)


        # Placeholder for ReverseDistillation-specific attributes
        self.tiler = None
        self._criterion = self._default_criterion
        self.anomaly_map_generator = self._default_anomaly_map_generator

    def forward(self, x):
        if self.backbone_name:
            self.encoder.eval() # pretrained network는 evaluation 모드로 고정.
            encoded_features = self.encoder(x)
            encoded = encoded_features[-1]  # Use the last feature map from the backbone
        else:
            encoded = self.encoder(x) # Custom Encoder

        decoded = self.decoder(encoded)

        if self.backbone_name:
             return [encoded_features[-1]], [decoded], x # Return input x as well
        else:
            return [encoded], [decoded], x

    def get_latent(self, x):
        if self.backbone_name:
            self.encoder.eval()
            features = self.encoder(x)
            return features[-1]
        else:
            return self.encoder(x)

    def _default_criterion(self, reconstructed_x, x):
        return F.mse_loss(reconstructed_x, x)


    def _default_anomaly_map_generator(self, reconstructed_x, x):
        """Calculate the anomaly map as the absolute difference between input and reconstruction."""
        return torch.abs(x - reconstructed_x).mean(dim=1, keepdim=True)

    def criterion(self, outputs: tuple):
        (_, reconstructed_images, input_images) = outputs
        loss = self._default_criterion(reconstructed_images[0], input_images)
        return loss


    def get_score_map(self, outputs: tuple):
        (_, reconstructed_images, input_images) = outputs
        # Use the updated anomaly map generator
        output = self.anomaly_map_generator(reconstructed_images[0], input_images)
        return output

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)