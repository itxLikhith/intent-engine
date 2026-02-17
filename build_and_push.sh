#!/bin/bash

# Script to build, tag, and push the Intent Engine API Docker image to Docker Hub

# Exit immediately if a command exits with a non-zero status
set -e

# Define variables
IMAGE_NAME="anony45/intent-engine-api"
TAG="latest"

# Print status message
echo "Building Docker image: $IMAGE_NAME:$TAG"

# Build the Docker image
docker build -t $IMAGE_NAME:$TAG -f Dockerfile .

# Tag the image
docker tag $IMAGE_NAME:$TAG $IMAGE_NAME:$TAG

# Push the image to Docker Hub
echo "Pushing image to Docker Hub: $IMAGE_NAME:$TAG"
docker push $IMAGE_NAME:$TAG

echo "Build and push completed successfully!"
