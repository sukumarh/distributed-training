# Using the PyTorch container
FROM python:3.8-slim

# Metadata
LABEL maintainer="Sukumar Hakhoo (sh5661), Dibyajit (db4057)" \
    project="Project 2" \
    title="Suggestive Resource Allocation"

# Expose any ports the app is expecting in the environment
ENV PORT 8002
EXPOSE $PORT

# Set up workspace
ADD endpoint-app .

# Install libraries
RUN apt-get update
RUN apt-get install libgomp1
RUN apt-get install -y libsm6 libxext6 libxrender-dev libglu1-mesa-dev libglib2.0-0 ca-certificates
RUN pip install -r requirements.txt

# Run the service
CMD [ "python", "app.py" ]