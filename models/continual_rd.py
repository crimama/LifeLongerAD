from anomalib.models.image.reverse_distillation.torch_model import ReverseDistillationModel
from anomalib.models.image.reverse_distillation.loss import ReverseDistillationLoss
from anomalib.models.image.reverse_distillation.anomaly_map import AnomalyMapGenerationMode
from torch import Tensor, nn
import torch
import torch.nn.functional as F


class RepresentationNormRegularizedLoss(nn.Module):
    """
    Combines the original Reverse Distillation loss with a representation norm-based regularization.
    """

    def __init__(self, alpha=1.0, beta=0.1):
        """
        Args:
            alpha (float): Weight for the original Reverse Distillation loss.
            beta (float):  Weight for the representation norm regularization.  Higher beta encourages
                         preserving/enhancing larger-norm representations and diminishing smaller-norm ones.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reverse_distillation_loss = ReverseDistillationLoss()  # Original loss

    def forward(self, encoder_features, decoder_features):
        """
        Calculates the combined loss.

        Args:
            encoder_features (list of Tensor):  Features from the encoder.
            decoder_features (list of Tensor): Features from the decoder.

        Returns:
            Tensor:  The combined loss value (scalar).
        """
        rd_loss = self.reverse_distillation_loss(encoder_features, decoder_features)

        # Calculate representation norms and regularization term.
        reg_loss = 0.0
        for dec_feat in decoder_features:  # Iterate through decoder features at different layers
            norm = torch.norm(dec_feat, p=2, dim=1)  # L2 norm along channel dimension (B, C, H, W) -> (B, H, W)
            # Inverted and scaled norm.  Smaller norms get larger values after this.
            # Adding a small constant (1e-6) for numerical stability (prevents division by zero).
            inverted_norm = 1.0 / (norm + 1e-6)
            reg_loss += torch.mean(inverted_norm)  # Mean over batch, height, and width.

        reg_loss /= len(decoder_features)  # Average regularization loss across layers.

        total_loss = self.alpha * rd_loss - self.beta * reg_loss # subtract, small norm loss term. 

        return total_loss



class ContinualReverseDistillation(ReverseDistillationModel):
    def __init__(self,
                 backbone: str,
                 input_size,
                 layers,
                 anomaly_map_mode: AnomalyMapGenerationMode,
                 pre_trained: bool = True,
                 alpha: float = 1.0,
                 beta: float = 0.1) -> None:

        super(ContinualReverseDistillation, self).__init__(
            backbone=backbone,
            input_size=input_size,
            layers=layers,
            anomaly_map_mode=anomaly_map_mode,
            pre_trained=pre_trained
        )

        # Use the combined custom loss
        self._criterion = RepresentationNormRegularizedLoss(alpha=alpha, beta=beta)
        for param in self.encoder.parameters():
            param.requires_grad = False  # Freeze encoder

        # Placeholder for old parameters (used for continual learning).  Initialized to None.
        self.old_decoder_params = None

    def forward(self, images: Tensor):
        self.encoder.eval()  # Keep encoder in eval mode

        if self.tiler:
            images = self.tiler.tile(images)
        encoder_features = self.encoder(images)
        encoder_features = list(encoder_features.values())
        decoder_features = self.decoder(self.bottleneck(encoder_features))

        if self.tiler:
            for i, features in enumerate(encoder_features):
                encoder_features[i] = self.tiler.untile(features)
            for i, features in enumerate(decoder_features):
                decoder_features[i] = self.tiler.untile(features)

        output = (encoder_features, decoder_features)

        return output

    def criterion(self, outputs: tuple):
        (encoder_features, decoder_features) = outputs
        loss = self._criterion(encoder_features, decoder_features)
        return loss


    def get_score_map(self, outputs: tuple):
        (encoder_features, decoder_features) = outputs
        output = self.anomaly_map_generator(encoder_features, decoder_features)
        return output
    
    def on_before_optimizer_step(self, optimizer):
        """
        Applies gradient modifications before each optimizer step.  This is where we
        implement the representation norm-based gradient scaling.
        """
        if self.old_decoder_params is None: #first learning, no update. 
            return

        with torch.no_grad():
            for (name, param), old_param in zip(self.decoder.named_parameters(), self.old_decoder_params):
                if param.grad is not None:
                    # Calculate the L2 norm of the *old* parameters.
                    param_norm = torch.norm(old_param, p=2)

                    # Scale the gradient.  Larger old parameter norms lead to larger gradients.
                    # Add a small epsilon to prevent division by zero.
                    epsilon = 1e-6
                    param.grad *= (param_norm + epsilon)


    def on_after_train_epoch_end(self, *args, **kwargs):
        """
        Store the decoder parameters *after* each epoch. These become the "old" parameters
        for the next task.  Deepcopy is important to prevent modification.
        """
        self.old_decoder_params = [param.clone().detach() for param in self.decoder.parameters()]
        #also, self.decoder should be set requires_grad = True. 

    def on_before_train_epoch_start(self, *args, **kwargs):
        # self.old_decoder_params = [param.clone().detach() for param in self.decoder.parameters()]
        for param in self.decoder.parameters():
            param.requires_grad = True