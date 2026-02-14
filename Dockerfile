FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Chainlit port
EXPOSE 8000

# Start Chainlit
CMD ["chainlit", "run", "chatBot.py", "--host", "0.0.0.0", "--port", "8000"]
