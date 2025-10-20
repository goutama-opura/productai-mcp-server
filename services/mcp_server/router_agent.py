import os
import json
import sys
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from services.mcp_server.tools.retrieval_tool import answer_faq
from services.mcp_server.tools.reviews_tool import analyze_reviews

try:
    from chains.router_chain import MultiAgentRouter
    from chains.embed_router import EmbeddingRouter
    from chains.fallback_chain import FallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain routing modules not found. Using fallback routing only.")
    LANGCHAIN_AVAILABLE = False

load_dotenv()

app = FastAPI(title="ProductAI - Unified 6-Agent Router")
logger.add("logs/unified_router.log", rotation="5 MB", level="INFO")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if LANGCHAIN_AVAILABLE:
    try:
        langchain_router = MultiAgentRouter()
        embed_router = EmbeddingRouter()
        fallback_handler = FallbackHandler()
        logger.info("✅ LangChain routers initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LangChain routers: {e}")
        LANGCHAIN_AVAILABLE = False

UNIFIED_SYSTEM_PROMPT = """
You are "ProductAI Assistant", an intelligent routing agent for an e-commerce platform.

Your job is to analyze the user's query and route it to the most appropriate agent from these 6 options:

1. **profiling** - Product recommendations, customer profiling, personalized suggestions, similar products
2. **ticketing** - Support tickets, complaints, returns, refunds, order issues
3. **troubleshooting** - Technical problems, errors, bugs, website/app not working
4. **customer_care** - General support, speak to human, escalation requests
5. **faq** - Questions about policies, shipping, payments, account management
6. **reviews** - Product reviews, ratings, customer feedback, pros/cons

Respond in JSON format:
{
  "agent": "agent_name",
  "confidence": 0.95,
  "reasoning": "Brief explanation"
}

Be precise and confident in your routing decision.
"""


def handle_profiling_agent(user_input: str, context: dict = None) -> dict:
    try:
        prompt = f"""You are a Product Recommendation Specialist.
        
User Query: {user_input}

Provide personalized product recommendations based on the query. Include:
- Recommended products/categories
- Reasoning for recommendations
- Personalization insights

Response:"""

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        return {
            "agent": "profiling",
            "response": response.choices[0].message.content,
            "success": True,
        }
    except Exception as e:
        logger.error(f"Profiling agent error: {e}")
        return {"agent": "profiling", "response": f"Error: {str(e)}", "success": False}


def handle_ticketing_agent(user_input: str, context: dict = None) -> dict:
    try:
        prompt = f"""You are a Customer Support Ticketing Specialist.

User Query: {user_input}

Handle this support request by:
- Creating a ticket summary
- Identifying the issue type
- Providing next steps
- Offering immediate assistance

Response:"""

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return {
            "agent": "ticketing",
            "response": response.choices[0].message.content,
            "success": True,
        }
    except Exception as e:
        logger.error(f"Ticketing agent error: {e}")
        return {"agent": "ticketing", "response": f"Error: {str(e)}", "success": False}


def handle_troubleshooting_agent(user_input: str, context: dict = None) -> dict:
    try:
        prompt = f"""You are a Technical Troubleshooting Specialist.

User Query: {user_input}

Diagnose and resolve this technical issue by:
- Identifying the problem
- Providing step-by-step solutions
- Offering workarounds if needed
- Explaining technical details clearly

Response:"""

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        return {
            "agent": "troubleshooting",
            "response": response.choices[0].message.content,
            "success": True,
        }
    except Exception as e:
        logger.error(f"Troubleshooting agent error: {e}")
        return {
            "agent": "troubleshooting",
            "response": f"Error: {str(e)}",
            "success": False,
        }


def handle_customer_care_agent(user_input: str, context: dict = None) -> dict:
    try:
        prompt = f"""You are a Customer Care Representative.

User Query: {user_input}

Provide friendly, helpful customer service by:
- Addressing their concern empathetically
- Offering relevant assistance
- Escalating to human if requested
- Maintaining professional tone

Response:"""

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )

        return {
            "agent": "customer_care",
            "response": response.choices[0].message.content,
            "success": True,
        }
    except Exception as e:
        logger.error(f"Customer care agent error: {e}")
        return {
            "agent": "customer_care",
            "response": f"Error: {str(e)}",
            "success": False,
        }


def handle_faq_tool(user_input: str, context: dict = None) -> dict:
    try:
        result = answer_faq(user_input)
        return {"agent": "faq", "response": result, "success": True}
    except Exception as e:
        logger.error(f"FAQ tool error: {e}")
        return {"agent": "faq", "response": f"Error: {str(e)}", "success": False}


def handle_reviews_tool(user_input: str, context: dict = None) -> dict:
    try:
        product_name = (
            context.get("product_name", user_input) if context else user_input
        )
        result = analyze_reviews(product_name)
        return {"agent": "reviews", "response": result, "success": True}
    except Exception as e:
        logger.error(f"Reviews tool error: {e}")
        return {"agent": "reviews", "response": f"Error: {str(e)}", "success": False}


AGENT_HANDLERS = {
    "profiling": handle_profiling_agent,
    "ticketing": handle_ticketing_agent,
    "troubleshooting": handle_troubleshooting_agent,
    "customer_care": handle_customer_care_agent,
    "faq": handle_faq_tool,
    "reviews": handle_reviews_tool,
}


def route_with_llm(user_input: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": UNIFIED_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1,
        )

        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        logger.error(f"LLM routing error: {e}")
        return {
            "agent": "customer_care",
            "confidence": 0.5,
            "reasoning": "Fallback due to routing error",
        }


def route_with_langchain(user_input: str, context: dict = None) -> dict:
    try:
        routing_result = langchain_router.route_request(user_input, context)

        return {
            "agent": routing_result.routing_decision.agent.value,
            "confidence": routing_result.routing_decision.confidence,
            "reasoning": routing_result.routing_decision.reasoning,
        }
    except Exception as e:
        logger.error(f"LangChain routing error: {e}")
        return route_with_llm(user_input)


@app.post("/chat")
async def unified_chat(request: Request):
    try:
        body = await request.json()
        user_input = body.get("message", "")
        context = body.get("context", {})

        if not user_input:
            return {"error": "Missing 'message' field"}

        logger.info(f"[CHAT] User: {user_input}")

        if LANGCHAIN_AVAILABLE:
            routing = route_with_langchain(user_input, context)
        else:
            routing = route_with_llm(user_input)

        selected_agent = routing["agent"]
        confidence = routing["confidence"]

        logger.info(f"[ROUTING] Agent: {selected_agent}, Confidence: {confidence:.2f}")

        handler = AGENT_HANDLERS.get(selected_agent, handle_customer_care_agent)
        result = handler(user_input, context)

        return {
            "routing": routing,
            "agent_response": result,
            "success": result["success"],
        }

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return {"error": str(e)}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "langchain_available": LANGCHAIN_AVAILABLE,
        "agents": list(AGENT_HANDLERS.keys()),
    }


@app.get("/")
def root():
    return {
        "message": "ProductAI Unified 6-Agent Router",
        "agents": {
            "profiling": "Product recommendations & customer profiling",
            "ticketing": "Support tickets, complaints, returns",
            "troubleshooting": "Technical issues & error resolution",
            "customer_care": "General support & human escalation",
            "faq": "FAQ retrieval using RAG",
            "reviews": "Product review summarization",
        },
        "endpoints": {
            "chat": "/chat (POST)",
            "health": "/health (GET)",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MCP_PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)