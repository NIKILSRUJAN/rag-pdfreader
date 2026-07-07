import ollama

def answer(query, context, model_name="mistral"):
    prompt =f"""You are a helpful assistant. Answer the question using ONLY the context 
        provided below. If the answer isn't in the context, say you don't know 
        — do not make things up.

        Context:
        {context}

        Question:
        {query}

        Answer:"""
    try:
        response = ollama.chat(
            model = model_name,
            messages = [{"role" : "user", "content" : prompt}]
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"error with: {type(e).__name__}: {e}")
        raise
    