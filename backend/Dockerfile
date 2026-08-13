# Use an official Python runtime as a parent image
# Python 3.11 is a good balance of features and compatibility
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# build-essential is often required for compiling C-extensions in some ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir keeps the Docker image smaller by not caching the downloaded packages
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the HuggingFace models during the Docker build process.
# This prevents the container from timing out on cold starts while trying to download models.
# (This step assumes you have a script or just let the app download it on first run, 
# but for production it's better to explicitly download them here if possible. 
# We will just copy the code and the app will download them on startup for now).

# Copy the rest of the backend code into the container
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
