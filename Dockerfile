FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV WORKER_COUNT=4
ENV MAX_WORKER_CONNECTIONS=1000

# Expose port 
EXPOSE 8000

# Command to run the application
CMD cd backend && gunicorn -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT -w $WORKER_COUNT --log-level info main:app
