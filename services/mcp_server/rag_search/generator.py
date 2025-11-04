import openai

def generate_answer(query: str, retrieved_docs: list) -> dict:
    """
    Generate an LLM response for the user query based on retrieved product documents.
    Uses the OpenAI ChatCompletion API correctly to avoid JSON errors.
    """

    # Build the context from retrieved documents, fallback if none
    if not retrieved_docs:
        context_text = "No relevant products found."
    else:
        product_lines = []
        for doc in retrieved_docs:
            product_id = doc.get("product_id", "unknown")
            title = doc.get("title", "Unknown Product")
            confidence = 1.0 / (1.0 + doc.get("distance", 1.0))
            product_lines.append(
                f"Product '{title}' (ID: {product_id}) with confidence {confidence:.3f}."
            )
        context_text = "Based on your query, here are some relevant products:\n" + "\n".join(product_lines)

    # Construct messages for ChatCompletion API, using system + user roles
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers user questions based on product "
                "information provided in the context. Use the context to generate accurate "
                "and concise responses while citing products by title and ID."
            )
        },
        {
            "role": "user",
            "content": f"User query: {query}\nContext:\n{context_text}\nAnswer:"
        }
    ]

    # Call OpenAI ChatCompletion API (example with openai library)
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # replace with your model name
        messages=messages,
        max_tokens=200,
        temperature=0.5,
        n=1,
        stop=None,
    )

    # Extract text answer
    answer_text = response.choices[0].message.content.strip()

    # Return a structured dict response
    return {
        "query": query,
        "answers": [
            {
                "text": answer_text,
                "source_documents": retrieved_docs
            }
        ]
    }
