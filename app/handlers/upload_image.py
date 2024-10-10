from fastapi import UploadFile
from pydantic import BaseModel
from PIL import Image as im
import io


class UploadImageResponse(BaseModel):

    width: int
    height: int


async def upload_image(image: UploadFile) -> UploadImageResponse:

    # Read the image
    image_bytes = await image.read()

    # Create a byte stream from the raw bytes
    image_byte_stream = io.BytesIO(image_bytes)

    # Open the image
    image = im.open(image_byte_stream)

    # Get image size
    width, height = image.size

    return UploadImageResponse(
        width=width,
        height=height,
    )
