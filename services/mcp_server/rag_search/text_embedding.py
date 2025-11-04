import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_text_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    if not text:
        return []
    try:
        response = openai.Embedding.create(input=text, model=model)
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating text embedding: {e}")
        return []
