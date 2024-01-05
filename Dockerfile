# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model directory from your local file system to the container
COPY ./models/phi-2/ /app/models/phi-2/

# Copy the rest of your application's code to the container
COPY . .

# Make port 6001 available to the world outside this container
EXPOSE 6001

# Define environment variable (if needed)
# ENV NAME Value

# Run app.py when the container launches
CMD ["python", "app.py"]
