import os
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# === Import your tool logic ===
from services.mcp_server.tools.retrieval_tool import answer_faq
from services.mcp_server.tools.reviews_tool import analyze_reviews

# === Create FastAPI app ===
app = FastAPI(title="ProductAI - Multi-Agent MCP Gateway")

# === Logging ===
logger.add("logs/api_gateway.log", rotation="5 MB", level="INFO")
logger.info("🚀 Starting ProductAI API Gateway")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "ProductAI MCP Gateway is healthy"}

@app.post("/tools/faq")
async def faq_tool(request: Request):
    """Support Agent: Retrieve answers from FAQ documents"""
    try:
        body = await request.json()
        query = body.get("query")
        if not query:
            return {"error": "Missing 'query' field"}
        logger.info(f"[FAQ Tool] Query: {query}")
        response = answer_faq(query)
        logger.info(f"[FAQ Tool] Response: {response['answer'][:80]}...")
        return response
    except Exception as e:
        logger.exception("FAQ Tool failed")
        return {"error": str(e)}

@app.post("/tools/reviews")
async def reviews_tool(request: Request):
    """Reviews Agent: Summarize or analyze product reviews"""
    try:
        body = await request.json()
        product_name = body.get("product_name")
        if not product_name:
            return {"error": "Missing 'product_name' field"}
        logger.info(f"[Reviews Tool] Product: {product_name}")
        response = analyze_reviews(product_name)
        logger.info(f"[Reviews Tool] Response: {response['analysis'][:80]}...")
        return response
    except Exception as e:
        logger.exception("Reviews Tool failed")
        return {"error": str(e)}