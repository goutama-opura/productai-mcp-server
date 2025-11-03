import sys
import os

# Add the 'services/mcp_server' folder to sys.path so Python can find api_gateway module
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'mcp_server'))

from api_gateway import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
