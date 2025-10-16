# ==========================================================
# ProductAI MCP Server - Multi-Agent Router (Dockerfile)
# ==========================================================

# Base Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy codebase
COPY . /app

# Install system dependencies (optional: for Mongo, FAISS, etc.)
RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose MCP port
EXPOSE 8001

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8001

# Start the router agent directly
CMD ["uvicorn", "services.mcp_server.router_agent:app", "--host", "0.0.0.0", "--port", "8001"]