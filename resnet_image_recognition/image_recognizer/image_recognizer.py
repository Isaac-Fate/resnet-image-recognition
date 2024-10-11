from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image as im
from ..resnet import ResNet50
from .imagenet_class_list import IMAGENET_CLASS_LIST


class ImageRecognizer:

    def __init__(self, model_path: Path) -> None:

        self._model = ResNet50()

        state_dict = torch.load(model_path)

        self._model.load_state_dict(state_dict)

        self._model.eval()

    def recognize(self, image: im.Image) -> str:

        resized_image = transforms.Resize((224, 224))(image)
        image_tensor = transforms.ToTensor()(resized_image)

        # Add batch dimension
        image_tensor = torch.unsqueeze(image_tensor, 0)

        # Run inference
        probs = self._model(image_tensor)

        # Find the class with the highest probability
        class_index = torch.argmax(probs).item()

        class_name = IMAGENET_CLASS_LIST[class_index]

        return class_name
