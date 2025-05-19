# Use official Python 3 slim image as base
FROM python:3-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
# Install git, pkg-config, and cmake for pip requirements
RUN apt-get update && apt-get install -y git pkg-config cmake && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port
EXPOSE 7860

# Set the default command
CMD ["python", "app.py", "--server", "0.0.0.0", "--port", "7860"]
