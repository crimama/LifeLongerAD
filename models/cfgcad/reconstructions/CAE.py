import torch
import torch.nn as nn

class ReconstructionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 간단히 3-block 인코더/디코더 정의
        enc_ch = [272, 192, 128, 64]
        dec_ch = enc_ch[::-1]
        out_pads = [1, 0, 1]

        # Encoder
        encoder_layers = []
        for in_c, out_c in zip(enc_ch, enc_ch[1:]):
            encoder_layers += [
                nn.Conv2d(in_c, out_c, 3, 2, 1),
                nn.ReLU(inplace=True),
            ]
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for idx, (in_c, out_c) in enumerate(zip(dec_ch, dec_ch[1:])):
            decoder_layers += [
                nn.ConvTranspose2d(in_c, out_c, 3, 2, 1, output_padding=out_pads[idx]),
            ]
            # 마지막만 Sigmoid
            if idx < len(out_pads) - 1:
                decoder_layers.append(nn.ReLU(inplace=True))
            else:
                decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)      # latent feature
        recon = self.decoder(z)  # reconstruction
        return recon, z


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ae = ReconstructionNetwork()

    def forward(self, src, pos_embed=None, task_embedding=None):
        """
        src: (B, C=272, H=14, W=14)
        pos_embed, task_embedding는 사용하지 않지만,
        Transformer와 동일한 시그니처를 맞추기 위해 받습니다.
        """
        output_decoder, output_encoder = self.ae(src)
        return output_decoder, output_encoder