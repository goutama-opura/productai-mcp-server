import sys, os, asyncio
from dotenv import load_dotenv

# ✅ Add project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastmcp import FastMCP
from services.mcp_server.tools.retrieval_tool import answer_faq
from services.mcp_server.tools.reviews_tool import analyze_reviews

# Load environment variables
load_dotenv()

HOST = os.getenv("MCP_HOST", "0.0.0.0")
PORT = int(os.getenv("MCP_PORT", 8001))

# ✅ Create the FastMCP app
mcp = FastMCP("ProductAI")

@mcp.tool()
def faq(query: str):
    """Retrieve answers from FAQ documents"""
    return answer_faq(query)

@mcp.tool()
def reviews(product_name: str):
    """Analyze or summarize reviews for a given product"""
    return analyze_reviews(product_name)

if __name__ == "__main__":
    print(f"🚀 Starting MCP HTTP server on {HOST}:{PORT}")
    asyncio.run(mcp.run_http_async(host=HOST, port=PORT))