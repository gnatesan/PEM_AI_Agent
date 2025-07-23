# agent/reasoner.py
from openai import OpenAI

client = OpenAI()

def generate_answer(query, top_docs, model="gpt-4o"):
    context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(top_docs)])

    prompt = f"""You are a pediatric emergency medicine research assistant. 
Use the following documents to answer the question.
Cite document numbers in your answer if relevant.

Question: {query}

Documents:
{context}

Answer:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )

    return response.choices[0].message.content

