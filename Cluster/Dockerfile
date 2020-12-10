# Using the PyTorch container
FROM pytorch/pytorch:latest
# FROM ubuntu:latest

# Metadata
LABEL maintainer="Sukumar Hakhoo (sh5661@nyu.edu)" \
    homework="Homework 5" \
    title="AlexNet_K8s"

# Expose any ports the app is expecting in the environment
ENV PORT 8002
EXPOSE $PORT

# Set up workspace
ADD endpoint-app .

# Install libraries
RUN apt-get update
# RUN apt-get install -y python3 python3-pip
RUN apt-get install -y libsm6 libxext6 libxrender-dev libglu1-mesa-dev libglib2.0-0 ca-certificates
RUN pip install -r requirements.txt

# Run the service
CMD [ "python", "endpoint.py" ]