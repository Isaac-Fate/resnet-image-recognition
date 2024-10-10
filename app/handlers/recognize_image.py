import io
from pathlib import Path
from PIL import Image as im
from pydantic import BaseModel
from fastapi import UploadFile

MODELS_DIR = Path.cwd().joinpath("models")
MODEL_PATH = MODELS_DIR.joinpath("resnet50-0676ba61.pth")


class RecognizeImageResponse(BaseModel):
    class_name: str


async def recognize_image(
    image: UploadFile,
) -> RecognizeImageResponse:

    # Read the image
    image_bytes = await image.read()

    # Create a byte stream from the raw bytes
    image_byte_stream = io.BytesIO(image_bytes)

    # Open the image
    image = im.open(image_byte_stream)

    from resnet_image_recognition import ImageRecognizer

    # Create the image recognizer
    image_recognizer = ImageRecognizer(MODEL_PATH)

    # Recognize
    class_name = image_recognizer.recognize(image)

    return RecognizeImageResponse(class_name=class_name)
