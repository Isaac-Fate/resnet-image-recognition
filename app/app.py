from fastapi import FastAPI, Depends
from mangum import Mangum
from .handlers import (
    health_check,
    greet,
    upload_image,
    UploadImageResponse,
    recognize_image,
    RecognizeImageResponse,
)
from .auth import validate_api_key

# Create an app instance
app = FastAPI()

# Create a handler
handler = Mangum(app)

# Register routes
app.get("/")(greet)
app.get("/health-check")(health_check)
app.post("/upload-image", response_model=UploadImageResponse)(upload_image)
app.post(
    "/recognize-image",
    response_model=RecognizeImageResponse,
    dependencies=[Depends(validate_api_key)],
)(recognize_image)
