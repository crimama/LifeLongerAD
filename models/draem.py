import torch 
import torch.nn as nn 
from .anomalib.data.utils import Augmenter
from .anomalib.models.components import AnomalyModule
from .anomalib.models.draem.loss import DraemLoss
from .anomalib.models.draem.torch_model import DraemModel

class Draem(nn.Module):

    def __init__(
        self, backbone=None, enable_sspcab: bool = False, sspcab_lambda: float = 0.1, anomaly_source_path = None) -> None:
        super().__init__()

        self.augmenter = Augmenter(anomaly_source_path)
        self.model = DraemModel(sspcab=enable_sspcab)
        self._criterion = DraemLoss()
        self.sspcab = enable_sspcab

        if self.sspcab:
            self.sspcab_activations: dict = {}
            self.setup_sspcab()
            self.sspcab_loss = nn.MSELoss()
            self.sspcab_lambda = sspcab_lambda
        
    def setup_sspcab(self) -> None:
        """Prepare the model for the SSPCAB training step by adding forward hooks for the SSPCAB layer activations."""

        def get_activation(name: str) -> Callable:
            """Retrieves the activations.

            Args:
                name (str): Identifier for the retrieved activations.
            """

            def hook(_, __, output: Tensor) -> None:
                """Hook for retrieving the activations."""
                self.sspcab_activations[name] = output

            return hook

        self.model.reconstructive_subnetwork.encoder.mp4.register_forward_hook(get_activation("input"))
        self.model.reconstructive_subnetwork.encoder.block5.register_forward_hook(get_activation("output"))
        
    def forward(self, input_image: torch.Tensor) -> list:
        self.train()
        augmented_image, anomaly_mask = self.augmenter.augment_batch(input_image)
        reconstruction, prediction = self.model(augmented_image)
        outputs = [input_image, reconstruction, anomaly_mask, prediction]
        return outputs
    
    def criterion(self, outputs):
        [input_image, reconstruction, anomaly_mask, prediction] = outputs 
        loss = loss = self._criterion(input_image, reconstruction, anomaly_mask, prediction)
        return loss 
    
    def get_score_map(self, images):
        self.eval()
        prediction = self.model(images)
        return prediction 
