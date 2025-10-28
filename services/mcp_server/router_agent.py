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
# Move this prompt template to DB (Configure this per tenant =====================
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

# === Web UI Mount ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(os.path.dirname(BASE_DIR), "web_ui")  # one level up from mcp_server

print(f"🌐 Serving Web UI from: {WEB_DIR}")

app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")

@app.get("/web")
async def serve_index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))

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


from fastapi import File, UploadFile

from fastapi import UploadFile, File
import tempfile
import base64


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyzes an uploaded image using GPT-4o-mini.
    Returns a short description of what the image depicts.
    """
    try:
        logger.info(f"🖼️ Received image: {file.filename}")

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Send to GPT model
        with open(tmp_path, "rb") as image_file:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that describes images in simple, human-friendly language."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image briefly."},
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,"},
                        ],
                    },
                ],
                temperature=0.2,
            )

        description = completion.choices[0].message.content
        return {"description": description}

    except Exception as e:
        logger.exception("Image analysis failed")
        return {"error": str(e)}


@app.post("/speech-to-intent")
async def speech_to_intent(file: UploadFile = File(...)):
    """
    Transcribes user speech and returns text + intent.
    """
    try:
        logger.info(f"🎙️ Received audio: {file.filename}")

        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Step 1: Transcribe audio to text
        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",  # economical GPT model for transcription
                file=audio_file
            )

        text = transcript.text
        logger.info(f"🗣️ Transcribed text: {text}")

        # Step 2: Get intent using your same SYSTEM_PROMPT
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ]
        )

        decision = json.loads(completion.choices[0].message.content)
        return {
            "transcribed_text": text,
            "decision": decision
        }

    except Exception as e:
        logger.exception("Speech-to-intent failed")
        return {"error": str(e)}

# @app.post("/speech-to-intent")
# async def speech_to_intent(file: UploadFile = File(...)):
#     """Convert uploaded speech to text (memory-only)."""
#     try:
#         logger.info(f"🎙️ Received audio: {file.filename}")
#
#         # Read into memory
#         audio_bytes = await file.read()
#         audio_stream = io.BytesIO(audio_bytes)
#
#         # Step 1: Transcribe
#         transcript = client.audio.transcriptions.create(
#             model="gpt-4o-mini-transcribe",
#             file=audio_stream
#         )
#
#         text = transcript.text
#         logger.info(f"🗣️ Transcribed: {text}")
#
#         # Step 2: Analyze intent
#         completion = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": text},
#             ],
#         )
#
#         routing_decision = completion.choices[0].message.content.strip()
#         logger.info(f"🤖 Routing decision: {routing_decision}")
#
#         try:
#             decision = json.loads(routing_decision)
#         except json.JSONDecodeError:
#             decision = {"intent": "fallback", "reason": "Invalid LLM response"}
#
#         return {"transcribed_text": text, "decision": decision}
#
#     except Exception as e:
#         logger.exception("Speech processing failed")
#         return {"error": str(e)}
#

# === Router Endpoint ===
@app.post("/chat")
async def chat_router(request: Request):
    """Routes conversation to the right tool (FAQ / Reviews / fallback) with sentiment and confidence scoring."""
    body = await request.json()
    user_message = body.get("message", "")

    if not user_message:
        return {"error": "Missing 'message' field."}

    logger.info(f"🧭 Incoming message: {user_message}")

    # 🧠 Step 1: LLM-based routing + sentiment + confidence
    ROUTER_PROMPT = f"""
    You are "ProductAI Assistant", an intelligent e-commerce conversation manager.

    The user says: "{user_message}"

    Your tasks:
    1. Classify the user's **intent** as one of:
       - "faq" (for orders, payments, returns, accounts, delivery tracking, etc.)
       - "reviews" (for product opinions, customer feedback, comparisons)
       - "compare" (We need to have min of 2 prods and max of 4 products"
       - "fallback" (if unrelated or unclear)
    2. Estimate your **confidence score** between 0–100% for this classification.
    3. Detect the **sentiment** of the user's tone for shopping context.
       Use one of:
       - "happy" (positive, excited, praising a product)
       - "curious" (neutral but engaged or inquiring)
       - "frustrated" (angry, problem, delay)
       - "disappointed" (sad, poor service, regretful)
       - "neutral" (informational or generic)
       - "casual" (light small-talk or non-shopping topic)
    4. Give a **reason** and **next_action** (short and helpful for routing).

    Respond ONLY in JSON, following this structure:
    {{
      "intent": "<faq | reviews | fallback>",
      "confidence": <integer between 0 and 100>,
      "sentiment": "<happy | curious | frustrated | disappointed | neutral | casual>",
      "reason": "<short reason>",
      "next_action": "<instruction for next step>"
    }}
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
    logger.info(f"🔍 Router raw output: {raw_response}")

    # Step 2: Parse JSON safely
    try:
        decision = json.loads(raw_response)
    except json.JSONDecodeError:
        logger.warning("⚠️ LLM response not valid JSON. Using fallback.")
        decision = {
            "intent": "fallback",
            "confidence": 50,
            "sentiment": "neutral",
            "reason": "Invalid LLM response format",
            "next_action": "Ask the user to rephrase"
        }

    intent = decision.get("intent", "fallback")
    confidence = decision.get("confidence", 0)
    sentiment = decision.get("sentiment", "neutral")

    logger.info(f"🤖 Parsed: Intent={intent}, Confidence={confidence}%, Sentiment={sentiment}")

    # Step 3: Route or fallback based on confidence threshold
    if confidence >= 85:
        if intent == "faq":
            logger.info("📚 Routed → FAQ tool")
            response = answer_faq(user_message)
            return {
                "source": "faq",
                "intent": intent,
                "confidence": confidence,
                "sentiment": sentiment,
                "response": response
            }

        elif intent == "reviews":
            logger.info("⭐ Routed → Reviews tool")
            response = analyze_reviews(user_message)
            return {
                "source": "reviews",
                "intent": intent,
                "confidence": confidence,
                "sentiment": sentiment,
                "response": response
            }
        elif intent == "compare":
            logger.info("⭐ Routed → Compare tool")
            response = {"source": "compare",
                "intent": intent,
                "confidence": confidence,
                "sentiment": sentiment,
                "response": {"status": "yet to build"}}
            return response

    # Step 4: Handle low confidence or fallback cases
    fallback_message = (
        f"I'm not entirely sure what you meant ({confidence}% confidence). "
        "Can you please rephrase or ask about an order, product, or review?"
    )
    logger.info("💬 Routed → Fallback due to low confidence")

    return {
        "source": "fallback",
        "intent": intent,
        "confidence": confidence,
        "sentiment": sentiment,
        "response": {"message": fallback_message}
    }


