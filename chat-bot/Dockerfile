
# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Run collectstatic
RUN python manage.py collectstatic --no-input

# Expose the port
EXPOSE 8000

# Use gunicorn with daphne worker for ASGI/WebSockets
CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "backend.asgi:application"]
