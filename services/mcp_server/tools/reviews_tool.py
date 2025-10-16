import os
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables
load_dotenv()

# Retrieve the API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found. Please set it in your .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)

def analyze_reviews(product_name: str, reviews: list[str] = None):
    """
    Summarizes reviews for a given product using LLM reasoning.
    """
    if not product_name:
        raise ValueError("Product name is required")

    base_prompt = f"""
    You are an expert product review analyst.
    Summarize customer feedback for the product "{product_name}".
    If reviews are provided, analyze them for sentiment, pros, cons, and recommendations.
    Format the response as:
    Summary:
    Pros:
    Cons:
    Sentiment:
    """

    if reviews:
        base_prompt += "\n\nHere are the reviews:\n" + "\n".join(reviews[:5])

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "You are a professional product analyst."},
            {"role": "user", "content": base_prompt},
        ],
        temperature=0.6,
    )

    return {
        "product": product_name,
        "analysis": response.choices[0].message.content
    }