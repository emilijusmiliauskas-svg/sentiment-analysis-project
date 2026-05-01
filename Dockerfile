# Layer 1: Use an official Python image as a base image
FROM python:3.11-slim

# Layer 2: Set the working directory inside the container
WORKDIR /app

# Layers 3&4: Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layers 5&6: Copy source code and trained model into the container
COPY src/ src/
COPY models/ models/

# Layer 7: Run a default sentiment prediction on container start
CMD ["python", "src/predict.py", "I love this product", "This is terrible"]
