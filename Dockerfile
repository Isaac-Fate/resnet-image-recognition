FROM --platform=amd64 python:3.11


COPY ./resnet_image_recognition ./resnet_image_recognition

COPY ./app ./app

COPY ./models ./models


# Install dependencies
RUN pip install awslambdaric pydantic "fastapi[standard]" mangum

# Install pytorch
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu


# Set runtime interface client as default command for the container runtime
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]

# Pass the name of the function handler as an argument to the runtime
CMD [ "app.handler" ]
