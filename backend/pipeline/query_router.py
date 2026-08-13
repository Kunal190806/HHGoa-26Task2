import re
from typing import Dict, Any

class QueryRouter:
    """
    Analyzes the query and determines:
    - Language (English, Hindi, Marathi, Code-mixed)
    - Intent (Factual, Entity-heavy, Comparative, Multi-hop)
    - Complexity (Simple, Moderate, Complex)
    - Routing Strategy (Weightings for Dense vs Sparse, Chunking Strategy Preference)
    """
    
    def __init__(self):
        # Basic Devanagari range for Hindi/Marathi detection
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]+')
        self.latin_pattern = re.compile(r'[a-zA-Z]+')
        
    def detect_language(self, query: str) -> str:
        has_devanagari = bool(self.devanagari_pattern.search(query))
        has_latin = bool(self.latin_pattern.search(query))
        
        if has_devanagari and has_latin:
            return "Code-mixed"
        elif has_devanagari:
            # We can't trivially distinguish Hindi and Marathi without a language model,
            # but MSMARCO-XI often treats 'hi' as the dominant one. We'll label as Hindi/Marathi.
            # In a full production system we'd use fasttext language id.
            if "आहे" in query or "काय" in query: # Simple Marathi heuristic
                return "Marathi"
            return "Hindi"
        else:
            return "English"
            
    def detect_intent(self, query: str) -> str:
        query_lower = query.lower().strip()
        
        # Chitchat / Greeting detection — match whole words only
        chitchat_starters = [
            r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bhow are you\b", 
            r"\bhow's it going\b", r"\bgood morning\b", r"\bgood evening\b",
            r"\bgood night\b", r"\bwhat's up\b", r"\bthanks\b", r"\bthank you\b",
            r"\bbye\b", r"\bgoodbye\b", r"\bsee you\b",
            r"\bनमस्ते\b", r"\bनमस्कार\b", r"\bकैसे हो\b", r"\bधन्यवाद\b"
        ]
        # Only classify as chitchat if the query is SHORT and matches a greeting
        if len(query_lower.split()) <= 5:
            for pattern in chitchat_starters:
                if re.search(pattern, query_lower):
                    return "Chitchat"
        
        comparative_keywords = ["vs", "difference", "compare", "better", "तुलना", "अंतर", "विरुद्ध"]
        if any(kw in query_lower for kw in comparative_keywords):
            return "Comparative"
            
        multihop_keywords = ["because", "after", "before", "result of", "कारण", "नंतर"]
        if any(kw in query_lower for kw in multihop_keywords):
            return "Multi-hop"
            
        # Very crude entity detection: Capitalized words in English, or specific keywords
        # In a real system, we'd use Spacy NER or a lightweight model.
        words = query.split()
        capitalized = [w for w in words if w.istitle()]
        if len(capitalized) > 2 or len(words) < 4: 
            return "Entity-heavy"
            
        return "Factual"
        
    def detect_complexity(self, query: str) -> str:
        word_count = len(query.split())
        if word_count < 6:
            return "Simple"
        elif word_count < 15:
            return "Moderate"
        else:
            return "Complex"
            
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Main entry point for routing logic.
        """
        lang = self.detect_language(query)
        intent = self.detect_intent(query)
        complexity = self.detect_complexity(query)
        
        # Determine retrieval strategy based on analysis
        strategy = {
            "dense_weight": 0.5,
            "sparse_weight": 0.5,
            "preferred_chunk_strategy": "sentence", # default
            "top_k_retrieve": 10, # hybrid retrieve candidate pool
            "top_k_rerank": 0     # final context size (0 = disable cross-encoder reranker)
        }
        
        if intent == "Entity-heavy":
            strategy["dense_weight"] = 0.3
            strategy["sparse_weight"] = 0.7 # Favor exact keyword matches
            strategy["preferred_chunk_strategy"] = "passage" # Needs more context for entities
            
        elif intent == "Comparative" or intent == "Multi-hop" or complexity == "Complex":
            strategy["dense_weight"] = 0.7
            strategy["sparse_weight"] = 0.3 # Favor semantic similarity for complex reasoning
            strategy["preferred_chunk_strategy"] = "parent" # Needs full passage or Parent-Child
            strategy["top_k_retrieve"] = 20 # Cast a wider net
            strategy["top_k_rerank"] = 0
            
        elif lang == "Code-mixed":
            # Code-mixed often struggles with dense semantic models unless they are specifically trained on code-mixed
            # So we balance or slightly favor sparse if words are phonetic
            strategy["dense_weight"] = 0.5
            strategy["sparse_weight"] = 0.5
            
        return {
            "language": lang,
            "intent": intent,
            "complexity": complexity,
            "strategy": strategy
        }

if __name__ == "__main__":
    router = QueryRouter()
    print(router.route_query("What is the capital of India?"))
    print(router.route_query("एमएस धोनी vs विराट कोहली in stats?"))
    print(router.route_query("भारत की राजधानी क्या है?"))
