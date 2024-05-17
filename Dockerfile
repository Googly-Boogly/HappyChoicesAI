# Use the official Python image from the Docker Hub
FROM python:3.11-alpine

# Set the working directory
WORKDIR /src

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the .env file
COPY .env .

# Copy the rest of the application code
COPY ./src .

# Specify the command to run the Python script
CMD ["python3", "main.py"]
