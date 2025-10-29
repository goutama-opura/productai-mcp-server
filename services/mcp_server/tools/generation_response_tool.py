import openai

def generate_answer(query: str, retrieved_docs: list) -> dict:
    """
    Generate an LLM response for the user query based on retrieved product documents.
    Uses the OpenAI ChatCompletion API correctly to avoid JSON errors.
    """

    # Build a concise prompt string based on retrieved docs, fallback to empty string if none
    if not retrieved_docs:
        prompt = "No relevant products found."
    else:
        prompt_lines = []
        for doc in retrieved_docs:
            product_id = doc.get("product_id", "unknown")
            title = doc.get("title", "Unknown Product")
            confidence = 1.0 / (1.0 + doc.get("distance", 1.0))
            prompt_lines.append(
                f"Product '{title}' (ID: {product_id}) with confidence {confidence:.3f}."
            )
        prompt = "Based on your query, here are some relevant products:\n" + "\n".join(prompt_lines)

    # Create OpenAI API chat completion request
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # replace with your model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Query: {query}\n{prompt}"}
        ],
        max_tokens=150,
        temperature=0.5,
    )

    answer_text = response.choices[0].message.content.strip()

    # Return structured response
    return {
        "query": query,
        "answers": [
            {
                "text": answer_text,
                "source_documents": retrieved_docs
            }
        ]
    }
