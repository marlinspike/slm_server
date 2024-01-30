# Define the ARG before the FROM instruction
ARG MODEL_NAME="tinyllama-1.1b-chat-v1.0"

# Use an official Python runtime as a base image
FROM cgr.dev/chainguard/python:latest

# Redefine the ARG after the FROM instruction so we can use it in the rest of the Dockerfile
ARG MODEL_NAME

# Set the working directory in the container to /app
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model directory from your local file system to the container
COPY ./models/${MODEL_NAME}/ /app/models/${MODEL_NAME}/

# Copy the rest of your application's code to the container
COPY . .

# Make port 6001 available to the world outside this container
EXPOSE 6001

# Run app.py when the container launches
CMD ["python", "app.py"]
