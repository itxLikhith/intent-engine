@echo off
setlocal enabledelayedexpansion

REM Script to build, tag, and push the Intent Engine API Docker image to Docker Hub

REM Define variables
set IMAGE_NAME=anony45/intent-engine-api
set TAG=latest

echo Building Docker image: !IMAGE_NAME!:!TAG!

REM Build the Docker image
docker build -t !IMAGE_NAME!:!TAG! -f Dockerfile .

REM Tag the image (same as build step in Docker)
docker tag !IMAGE_NAME!:!TAG! !IMAGE_NAME!:!TAG!

REM Push the image to Docker Hub
echo Pushing image to Docker Hub: !IMAGE_NAME!:!TAG!
docker push !IMAGE_NAME!:!TAG!

echo Build and push completed successfully!
pause
