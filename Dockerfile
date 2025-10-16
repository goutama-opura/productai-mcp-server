# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy code
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose your MCP server port
EXPOSE 8001

# Run the server
CMD ["python", "services/mcp_server/main.py"]