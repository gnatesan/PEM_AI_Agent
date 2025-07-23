# agent/reasoner.py
import openai

def generate_answer(query, top_docs, model="gpt-4"):
    context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(top_docs)])

    prompt = f"""You are a pediatric emergency medicine research assistant. 
Use the following documents to answer the question.
Cite document numbers in your answer if relevant.

Question: {query}

Documents:
{context}

Answer:"""

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )

    return response["choices"][0]["message"]["content"]

