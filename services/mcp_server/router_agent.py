import os
import json
import requests
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

# === Load environment ===
load_dotenv()

# === Imports for tools ===
from services.mcp_server.tools.retrieval_tool import answer_faq
from services.mcp_server.tools.reviews_tool import analyze_reviews

# === Initialize app & logger ===
app = FastAPI(title="ProductAI - Unified Router & Tools")
logger.add("logs/unified_app.log", rotation="5 MB", level="INFO")

# === OpenAI client ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === System Prompt for Router Agent ===
SYSTEM_PROMPT = """
You are "ProductAI Assistant", the conversation manager for an e-commerce platform.
Your job is to decide whether the user's query belongs to:
1. FAQ-related queries (orders, returns, payments, account, etc.)
2. Reviews-related queries (opinions, ratings, customer feedback)
3. Or unrelated topics (fallback).

Respond in JSON ONLY with this structure:
{
  "intent": "<faq | reviews | fallback>",
  "tool_to_use": "<faq | reviews | none>",
  "reason": "<short reason>",
  "next_action": "<instruction>"
}
No additional text.
"""

# === Health Check ===
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Unified ProductAI service is running 🚀"}

# === FAQ Tool Endpoint ===
@app.post("/tools/faq")
async def faq_tool(request: Request):
    """Handles support & help-related queries"""
    try:
        body = await request.json()
        query = body.get("query")
        if not query:
            return {"error": "Missing 'query' field."}
        logger.info(f"[FAQ Tool] Query: {query}")
        response = answer_faq(query)
        return response
    except Exception as e:
        logger.exception("FAQ tool failed")
        return {"error": str(e)}

# === Reviews Tool Endpoint ===
@app.post("/tools/reviews")
async def reviews_tool(request: Request):
    """Handles product review analysis"""
    try:
        body = await request.json()
        product_name = body.get("product_name")
        if not product_name:
            return {"error": "Missing 'product_name' field."}
        logger.info(f"[Reviews Tool] Product: {product_name}")
        response = analyze_reviews(product_name)
        return response
    except Exception as e:
        logger.exception("Reviews tool failed")
        return {"error": str(e)}

# === Router Endpoint ===
@app.post("/chat")
async def chat_router(request: Request):
    """Routes conversation to the right tool (FAQ / Reviews / fallback)"""
    body = await request.json()
    user_message = body.get("message", "")

    if not user_message:
        return {"error": "Missing 'message' field."}

    logger.info(f"🧭 Incoming message: {user_message}")

    # Step 1: Ask LLM for routing decision
    completion = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0
    )

    raw_response = completion.choices[0].message.content.strip()
    logger.info(f"🔍 Router output: {raw_response}")

    # Step 2: Parse JSON safely
    try:
        decision = json.loads(raw_response)
    except json.JSONDecodeError:
        decision = {
            "intent": "fallback",
            "tool_to_use": "none",
            "reason": "Invalid LLM response",
            "next_action": "Fallback message"
        }

    intent = decision.get("tool_to_use", "none")

    # Step 3: Route request internally
    if intent == "faq":
        logger.info("📚 Routed → FAQ tool")
        response = answer_faq(user_message)
        return {"source": "faq", "intent": decision, "response": response}

    elif intent == "reviews":
        logger.info("⭐ Routed → Reviews tool")
        response = analyze_reviews(user_message)
        return {"source": "reviews", "intent": decision, "response": response}

    else:
        fallback = (
            "I'm here to help with your orders, returns, payments, or product reviews. "
            "Can you please ask about one of these topics?"
        )
        logger.info("💬 Routed → Fallback")
        return {"source": "fallback", "intent": decision, "response": {"message": fallback}}