from fastapi import FastAPI
from pydantic import BaseModel
from mangum import Mangum
from .handlers import health_check, recognize_image


# Create an app instance
app = FastAPI()

# Register routes
app.get("/health-check")(health_check)
app.post("/recognize-image")(recognize_image)

handler = Mangum(app)
