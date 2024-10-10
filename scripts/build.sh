#!/bin/sh

# Exit on error
set -e

# Project name
PROJECT_NAME=resnet-image-recognition

# Build the Docker image
docker build --file=Dockerfile --tag=$PROJECT_NAME . 
