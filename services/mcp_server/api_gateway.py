import os
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from services.mcp_server.tools.retrieval_tool import answer_faq
from services.mcp_server.tools.reviews_tool import analyze_reviews
from services.mcp_server.tools.retriever_tool import retrieve_documents
from services.mcp_server.tools.generation_response_tool import generate_response

app = FastAPI(title="ProductAI - Multi-Agent MCP Gateway")

logger.add("logs/api_gateway.log", rotation="5 MB", level="INFO")
logger.info("🚀 Starting ProductAI API Gateway")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "ProductAI MCP Gateway is healthy"}

@app.post("/tools/faq")
async def faq_tool(request: Request):
    body = await request.json()
    query = body.get("query")
    if not query:
        return {"error": "Missing 'query' field"}
    logger.info(f"[FAQ Tool] Query: {query}")
    response = answer_faq(query)
    logger.info(f"[FAQ Tool] Response preview: {str(response)[:80]}")
    return response

@app.post("/tools/reviews")
async def reviews_tool(request: Request):
    body = await request.json()
    product_name = body.get("product_name")
    if not product_name:
        return {"error": "Missing 'product_name' field"}
    logger.info(f"[Reviews Tool] Product: {product_name}")
    response = analyze_reviews(product_name)
    logger.info(f"[Reviews Tool] Response preview: {str(response)[:80]}")
    return response

@app.post("/tools/retriever")
async def retriever_tool(request: Request):
    body = await request.json()
    query = body.get("query")
    top_k = body.get("top_k", 5)
    logger.info(f"[Retriever Tool] Query: {query}, K: {top_k}")
    results = retrieve_documents(query, top_k=top_k)
    return results

@app.post("/tools/generation")
async def generation_tool(request: Request):
    body = await request.json()
    prompt = body.get("prompt")
    context = body.get("context", [])
    logger.info(f"[Generation Tool] Prompt: {prompt}, Context length: {len(context)}")
    answer = generate_response(prompt, context)
    return answer
