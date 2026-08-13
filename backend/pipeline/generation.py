import os
from typing import List, Dict, Any

class LLMProvider:
    def generate(self, query: str, context: List[str]) -> str:
        raise NotImplementedError

class GeminiProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model = None
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-2.0-flash")
                print("Initialized Gemini Provider (Live API)")
            except Exception as e:
                print(f"Gemini init failed: {e}. Falling back to extractive.")
                self.model = None
        else:
            print("No GEMINI_API_KEY found. Using extractive answer generation.")
        
    def generate(self, query: str, context: List[str]) -> str:
        if self.model:
            return self._generate_with_gemini(query, context)
        else:
            return self._extractive_answer(query, context)
    
    def _generate_with_gemini(self, query: str, context: List[str]) -> str:
        """Call real Gemini API for answer generation."""
        context_block = "\n---\n".join(context[:5])  # Top 5 passages
        
        prompt = f"""You are a highly accurate AI assistant. Answer the user's question ONLY using the provided context.

Rules:
1. Answer ONLY from the supplied context passages.
2. Do NOT invent or hallucinate any facts.
3. If the context is insufficient, say: "I couldn't find enough information in the dataset to answer that."
4. Keep the answer concise (1-3 sentences).
5. If the context is in Hindi, you may answer in Hindi or English.

Context:
{context_block}

Question: {query}

Answer:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini generation error: {e}")
            return self._extractive_answer(query, context)
    
    def _extractive_answer(self, query: str, context: List[str]) -> str:
        """Fallback: return the most relevant passage directly as the answer."""
        if not context:
            return "I couldn't find enough information in the dataset to answer that."
        
        # Return the first (highest-ranked) passage as the answer
        best_passage = context[0].strip()
        
        # Truncate if too long
        if len(best_passage) > 500:
            best_passage = best_passage[:500] + "..."
            
        return best_passage


class GenerationPipeline:
    def __init__(self, provider_name: str = "gemini"):
        self.provider_name = provider_name.lower()
        if self.provider_name == "gemini":
            self.provider = GeminiProvider()
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")
            
    def generate_answer(self, query: str, context_results: List[Dict[str, Any]]) -> str:
        """
        Generates an answer using the configured LLM and strict RAG prompt.
        """
        passages = [c['payload']['text'] for c in context_results]
        return self.provider.generate(query, passages)
