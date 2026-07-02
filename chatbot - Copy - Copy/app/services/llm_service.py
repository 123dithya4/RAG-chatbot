import os
from groq import Groq
from sentence_transformers import SentenceTransformer
from app.core.config import GROQ_API_KEY

# Initialize Clients
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"Error initializing Groq: {e}")
    groq_client = None

def get_embedding(text_list):
    """
    Generates embeddings for a list of strings.
    """
    return embedding_model.encode(text_list)

def get_llm_response(prompt):
    """
    Sends the prompt to Groq API.
    Uses a standard Groq model and a temperature of 0.7 for varied, natural responses.
    """
    if not groq_client:
        raise Exception("Groq client not initialized. Check API Key.")
    
    response = groq_client.chat.completions.create(
        # We use Mixtral or Llama 3 for best results on Groq.
        # 'openai/gpt-oss-120b' is likely invalid; switching to a stable Groq model:
        model="openai/gpt-oss-120b", 
        messages=[
            {"role": "user", "content": prompt}
        ],
        # 0.7 ensures "whatsup" gets different answers like "Doing great!" vs "Fantastic!"
        temperature=0.7, 
        max_tokens=1024
    )
    return response.choices[0].message.content