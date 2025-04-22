
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def answer_question(user_query, context_block):
    prompt = f"""
You are a VC landscape assistant. A founder uploaded their business summary and we extracted summaries of top VCs.

Context:
{context_block}

Question:
{user_query}

Answer clearly and concisely using the context above.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
