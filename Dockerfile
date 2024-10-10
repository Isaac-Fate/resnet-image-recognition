# Base image
FROM --platform=linux/amd64 python:3.11


# Copy files

# Copy the package source
COPY ./resnet_image_recognition ./resnet_image_recognition

# Copy the FastAPI app source
COPY ./app ./app

# Copy model weights
COPY ./models ./models

# Copy API keys
COPY ./api-keys.txt .


# Install dependencies

# Install dependencies
RUN pip install awslambdaric pydantic "fastapi[standard]" mangum

# Install PyTroch for Linux
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu


# Copy the pyproject.toml for installing the custom package
# COPY ./pyproject.toml .

# Install the custom package
# RUN pip install -e .


# Set runtime interface client as default command for the container runtime
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]

# Pass the name of the function handler as an argument to the runtime
CMD [ "app.handler" ]
