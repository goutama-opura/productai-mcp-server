import os
import io
import json
import requests
from fastapi import FastAPI, Request, File, UploadFile
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from services.mcp_server.tools.retrieval_tool import answer_faq
from services.mcp_server.tools.reviews_tool import analyze_reviews
from services.mcp_server.tools.retriever_tool import Retriever
from services.mcp_server.tools.generation_response_tool import generate_answer

# Load environment
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="ProductAI Router Agent")

# Updated system prompt with new intents and tool_to_use options
SYSTEM_PROMPT = """
You are "ProductAI Assistant", the conversation manager for an e-commerce platform.
Your job is to decide whether the user's query belongs to:
1. FAQ-related queries (orders, returns, payments, account, etc.)
2. Reviews-related queries (opinions, ratings, customer feedback)
3. Product inquiry queries (features, specs, availability)
4. Or unrelated topics (fallback).

Respond in JSON ONLY with this structure:
{
  "intent": "<faq | reviews | product_inquiry | fallback>",
  "tool_to_use": "<faq | reviews | retriever | generation | none>",
  "confidence": <integer between 0 and 100>,
  "sentiment": "<happy | curious | frustrated | disappointed | neutral | casual>",
  "reason": "<short reason>",
  "next_action": "<instruction>"
}
No additional text.
"""

@app.post("/chat")
async def chat_router(request: Request):
    body = await request.json()
    user_message = body.get("message", "")

    if not user_message:
        return {"error": "Missing 'message' field."}

    logger.info(f" Incoming message: {user_message}")

    # Compose router prompt with user message embedded
    ROUTER_PROMPT = f"""
    You are "ProductAI Assistant", an intelligent e-commerce conversation manager.

    The user says: "{user_message}"

    Your tasks:
    1. Classify the user's **intent** as one of:
       - "product_inquiry" (questions about product features, specs, availability)
       - "faq" (orders, payments, returns, accounts, delivery tracking, etc.)
       - "reviews" (product opinions, feedback, comparisons)
       - "compare" (min 2 prods max 4 products)
       - "fallback" (unrelated or unclear)
    2. Estimate your **confidence score** between 0–100%.
    3. Detect the **sentiment** (happy, curious, frustrated, disappointed, neutral, casual).
    4. Specify **tool_to_use**: faq, reviews, retriever, generation, or none.
    5. Provide a **reason** and **next_action**.

    Respond ONLY in JSON following this structure.
    """

    completion = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "You are a professional conversation router for e-commerce AI."},
            {"role": "user", "content": ROUTER_PROMPT}
        ],
        temperature=0.2
    )

    raw_response = completion.choices[0].message.content.strip()
    logger.info(f"Router raw output: {raw_response}")

    try:
        decision = json.loads(raw_response)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON from LLM, fallback triggered.")
        decision = {
            "intent": "fallback",
            "tool_to_use": "none",
            "confidence": 50,
            "sentiment": "neutral",
            "reason": "Invalid LLM response format",
            "next_action": "Ask the user to rephrase"
        }

    intent = decision.get("intent", "fallback")
    tool_to_use = decision.get("tool_to_use", "none")
    confidence = decision.get("confidence", 0)
    sentiment = decision.get("sentiment", "neutral")

    logger.info(f"Parsed: Intent={intent}, Tool={tool_to_use}, Confidence={confidence}%, Sentiment={sentiment}")

    # Route based on intent and tool_to_use with confidence threshold
    if confidence >= 85:
        if intent == "faq" and tool_to_use == "faq":
            logger.info("Routed → FAQ tool")
            from services.mcp_server.tools.retrieval_tool import answer_faq
            response = answer_faq(user_message)
            return {"source": "faq", "intent": intent, "confidence": confidence, "sentiment": sentiment, "response": response}

        elif intent == "reviews" and tool_to_use == "reviews":
            logger.info(" Routed → Reviews tool")
            from services.mcp_server.tools.reviews_tool import analyze_reviews
            response = analyze_reviews(user_message)
            return {"source": "reviews", "intent": intent, "confidence": confidence, "sentiment": sentiment, "response": response}

        elif intent == "product_inquiry" and tool_to_use in ["retriever", "generation"]:
            logger.info(f"Routed → {tool_to_use.capitalize()} tool")
            # Here add calls to retriever or generation tools accordingly
            # For example:
            # if tool_to_use == "retriever":
            #     response = retrieve_documents(user_message)
            # else:
            #     response = generate_response(user_message)
            response = {"status": f"Called {tool_to_use} with user query"}  # placeholder
            return {"source": tool_to_use, "intent": intent, "confidence": confidence, "sentiment": sentiment, "response": response}

        elif intent == "compare":
            logger.info("Routed → Compare tool")
            return {"source": "compare", "intent": intent, "confidence": confidence, "sentiment": sentiment, "response": {"status": "yet to build"}}

    fallback_msg = f"I'm not sure what you meant ({confidence}% confidence). Please rephrase or ask about an order, product, or review."
    logger.info(" Routed → Fallback")

    return {"source": "fallback", "intent": intent, "tool_to_use": tool_to_use, "confidence": confidence, "sentiment": sentiment, "response": {"message": fallback_msg}}
