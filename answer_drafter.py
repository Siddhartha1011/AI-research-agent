from llama_cpp import Llama
from typing import List
from langchain_core.documents import Document
import os
import logging

MODEL_PATH = os.path.join("models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        temperature=0.7,
        top_p=0.95,
        repeat_penalty=1.1,
        verbose=True
    )
    logger.info("✅ LLM loaded")
except Exception as e:
    logger.error(f"❌ LLM init failed: {str(e)}")
    raise

def draft_answer(chunks: List[Document], question: str) -> str:
    try:
        context = "\n\n".join([doc.page_content for doc in chunks])
        base_prompt = f"""<s>[INST] You are a helpful assistant. Based on the context below, answer the question clearly and concisely.

Context:
{{context}}

Question: {question}

Answer: [/INST]</s>"""
        
        reserved = 512
        token_estimate = len(base_prompt.format(context="")) // 4
        max_context_chars = (2048 - reserved - token_estimate) * 4
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "..."
        
        prompt = base_prompt.format(context=context)
        response = llm(prompt, max_tokens=512, stop=["</s>"])
        
        if not response or 'choices' not in response or not response['choices']:
            return "Error: Model returned empty response"
        return response['choices'][0]['text'].strip()
        
    except Exception as e:
        logger.error(f"❌ Error in draft_answer: {str(e)}")
        return f"Error: {str(e)}"