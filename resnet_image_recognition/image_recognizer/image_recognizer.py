from pathlib import Path
from PIL import Image as im
import torch
from torchvision import transforms
from ..resnet import ResNet50
from .imagenet_classes import IMAGENET_CLASSES


class ImageRecognizer:

    def __init__(self, model_path: Path) -> None:

        # Load the state dict
        state_dict = torch.load(model_path)

        # Create an empty model
        self._model = ResNet50()

        # Set the pretrained weights
        self._model.load_state_dict(state_dict)

        # Set evaluation mode
        self._model.eval()

    def recognize(self, image: im.Image) -> str:
        """
        Recognizes the class of an image.

        Parameters
        ----------
        image : Image
            The input image.

        Returns
        -------
        str
            The name of the class of the given image.
        """

        # Resize the image
        resized_image = transforms.Resize((224, 224))(image)

        # Convert to tensor
        image_tensor = transforms.ToTensor()(resized_image)

        # Add batch dimension
        image_tensor = torch.unsqueeze(image_tensor, 0)

        # Run inference
        # The probabilities for each class are returned
        probs = self._model(image_tensor)

        # Find the class with the highest probability
        class_index = torch.argmax(probs).item()

        # Get the class name
        class_name = IMAGENET_CLASSES[class_index]

        return class_name
