from fastapi import UploadFile
import io
from PIL import Image as im
from pathlib import Path
from pydantic import BaseModel
from resnet_image_recognition import ImageRecognizer


MODEL_PATH = Path.cwd().joinpath("models/resnet50-0676ba61.pth")


class RecognizeImageResponse(BaseModel):

    class_name: str


async def recognize_image(image: UploadFile) -> RecognizeImageResponse:

    image_bytes = await image.read()
    image_byte_stream = io.BytesIO(image_bytes)
    image = im.open(image_byte_stream)

    image_recognizer = ImageRecognizer(MODEL_PATH)

    class_name = image_recognizer.recognize(image)

    return RecognizeImageResponse(class_name=class_name)
