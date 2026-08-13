# HH Goa 2026 - Task 2: Multilingual Voice RAG

## Current Status (Working)
- **Voice Input**: Real Sarvam STT integration is complete and working.
- **Microphone UI**: Premium WebGL "Strands" animation plays while recording.
- **Query Routing**: Correctly handles chitchat/greetings directly, bypassing RAG for casual conversation.
- **Retrieval**: Hybrid Search (BGE-M3 Dense + BM25 Sparse) + RRF is live and benchmarked (45.4ms P50 latency).
- **Generation & Grounding**: LLM pipeline is wired up with an extractive fallback (returns the top passage directly) when no `GEMINI_API_KEY` is present.

## TODO / Next Steps

### 1. LLM Integration (High Priority)
- [ ] Obtain a Google Gemini API Key.
- [ ] Add `GEMINI_API_KEY` to the `.env` file or environment variables on the backend.
- [ ] Verify that real generative answers are being produced instead of the extractive fallback.

### 2. Router Enhancements
- [ ] Replace basic regex/heuristic language detection in `QueryRouter` with a lightweight model (e.g., `fasttext` langid) for accurate Marathi vs Hindi distinction.
- [ ] Add more robust Entity Recognition (NER) for the "Entity-heavy" routing intent.

### 3. Deployment
- [ ] Containerize the FastAPI backend with Docker.
- [ ] Deploy backend to a scalable cloud service (e.g., GCP Cloud Run, AWS App Runner).
- [ ] Deploy the Next.js frontend to Vercel or Netlify.
- [ ] Configure CORS in `main.py` to only allow the production frontend URL.

### 4. UI/UX Polish
- [ ] Add visual indicators for the routing intent (e.g., show if the query was treated as "Chitchat", "Factual", or "Multi-hop").
- [ ] Handle network disconnects gracefully on the frontend with a toast notification.
- [ ] Support stopping playback of the TTS audio if implemented.

### 5. Production Hardening
- [ ] Move the hardcoded Sarvam API key in `sarvam_client.py` strictly to an environment variable.
- [ ] Tune the grounding overlap thresholds based on a larger evaluation dataset.
- [ ] Add API rate limiting to the FastAPI endpoints.
