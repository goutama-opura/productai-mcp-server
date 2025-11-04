import sys
import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastmcp import FastMCP

# Add project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.mcp_server.tools.retrieval_tool import answer_faq
from services.mcp_server.tools.reviews_tool import analyze_reviews
from services.mcp_server.router_agent import app as router_agent_app



# Load environment variables
load_dotenv()

HOST = os.getenv("MCP_HOST", "0.0.0.0")
PORT = int(os.getenv("MCP_PORT", 8001))

# Create FastMCP instance
mcp = FastMCP("ProductAI")

@mcp.tool()
def faq(query: str):
    """Retrieve answers from FAQ documents"""
    return answer_faq(query)

@mcp.tool()
def reviews(product_name: str):
    """Analyze or summarize reviews for a given product"""
    return analyze_reviews(product_name)

# Create the MCP FastAPI app with proper lifespan for mounting
mcp_app = mcp.http_app(path="/chat")

# Create a main FastAPI app and mount mcp_app under /chat
app = FastAPI()
app.mount("/chat", mcp_app)

if __name__ == "__main__":
    print(f"🚀 Starting MCP HTTP server on {HOST}:{PORT}")
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
