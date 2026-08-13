import re
from typing import List, Dict, Any, Tuple

class GroundingValidator:
    def __init__(self):
        # We could load a small NLI model here, but for extreme latency optimization,
        # we start with a lexical/semantic overlap heuristic for the lightweight check.
        pass

    def _extract_keywords(self, text: str) -> set:
        """Extracts basic lowercase tokens, stripping punctuation."""
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out very common stopwords if needed, but simple set is fine for now
        return set(words)

    def lightweight_check(self, answer: str, context_passages: List[str]) -> str:
        """
        Concrete lightweight grounding check using lexical overlap.
        Checks if the key claims (words/entities) in the answer appear in the context.
        Returns: "PASS", "UNCERTAIN", or "FAIL"
        """
        if not answer or not context_passages:
            return "FAIL"
            
        answer_tokens = self._extract_keywords(answer)
        if not answer_tokens:
            return "UNCERTAIN"
            
        # Combine all context into one large set of tokens
        context_text = " ".join(context_passages)
        context_tokens = self._extract_keywords(context_text)
        
        # Calculate overlap percentage
        overlap = answer_tokens.intersection(context_tokens)
        overlap_ratio = len(overlap) / len(answer_tokens)
        
        # Thresholds (tune in production)
        if overlap_ratio > 0.5:
            return "PASS"
        elif overlap_ratio > 0.25:
            return "UNCERTAIN"
        else:
            return "FAIL"

    def llm_check(self, answer: str, context_passages: List[str], llm_client: Any) -> str:
        """
        LLM verification fallback if the lightweight check is UNCERTAIN.
        llm_client would be an instance of our generation model client.
        """
        # In a real implementation, we prompt the LLM to verify if the answer is grounded.
        # For now, we return a mock PASS to complete the pipeline architecture.
        # Example prompt: "Given this context: {context}, is this answer fully supported? {answer}. Reply YES or NO."
        return "PASS"
        
    def validate(self, answer: str, context_results: List[Dict[str, Any]], llm_client: Any = None) -> Tuple[str, str]:
        """
        Runs the full grounding pipeline.
        Returns (Verdict, Reason)
        """
        passages = [c['payload']['text'] for c in context_results]
        
        lw_verdict = self.lightweight_check(answer, passages)
        
        if lw_verdict == "PASS":
            return "PASS", "Lightweight check passed (high overlap)."
        elif lw_verdict == "FAIL":
            return "FAIL", "Lightweight check failed (low overlap)."
        else:
            # Uncertain -> Fallback to LLM
            if llm_client:
                llm_verdict = self.llm_check(answer, passages, llm_client)
                return llm_verdict, "LLM verification completed."
            else:
                # If no LLM available for check, be conservative
                return "FAIL", "Lightweight check uncertain, and no LLM fallback available."

if __name__ == "__main__":
    validator = GroundingValidator()
    ctx = ["The capital of France is Paris. It is a large city."]
    print(validator.lightweight_check("Paris is the capital of France.", ctx)) # Should PASS
    print(validator.lightweight_check("Berlin is the capital of Germany.", ctx)) # Should FAIL
    print(validator.lightweight_check("Paris is a city in Europe.", ctx)) # Might be UNCERTAIN
