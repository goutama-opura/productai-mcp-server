import os
import sys
import asyncio
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from services.mcp_server.tools.retrieval_tool import answer_faq
from services.mcp_server.tools.reviews_tool import analyze_reviews
from services.mcp_server.tools.retriever_tool import retrieve_documents
from services.mcp_server.tools.generation_response_tool import generate_response

# Load env variables
load_dotenv()

HOST = os.getenv("MCP_HOST", "0.0.0.0")
PORT = int(os.getenv("MCP_PORT", 8001))

mcp = FastMCP("ProductAI")

@mcp.tool()
def faq(query: str):
    return answer_faq(query)

@mcp.tool()
def reviews(product_name: str):
    return analyze_reviews(product_name)

@mcp.tool()
def retriever(query: str, top_k: int = 5):
    return retrieve_documents(query, top_k=top_k)

@mcp.tool()
def generator(prompt: str, context: list):
    return generate_response(prompt, context)

if __name__ == "__main__":
    print(f"🚀 Starting MCP Server on {HOST}:{PORT}")
    asyncio.run(mcp.run_http_async(host=HOST, port=PORT))
