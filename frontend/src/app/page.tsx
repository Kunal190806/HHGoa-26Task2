"use client";

import { useState, useRef } from "react";
import dynamic from "next/dynamic";
import "./globals.css";

const Strands = dynamic(() => import("./components/Strands"), { ssr: false });

// Interface for API Response matching FastAPI
interface RAGResponse {
  status: string;
  query: string;
  answer?: string;
  message?: string;
  sources?: any[];
  routing?: any;
  retrieval_confidence?: string;
  grounding?: { status: string; reason: string };
  latency_metrics?: Record<string, number>;
}

export default function Home() {
  const [isRecording, setIsRecording] = useState(false);
  const [messages, setMessages] = useState<{role: "user" | "agent", content: string | RAGResponse}[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: "audio/wav" });
        await sendAudio(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      alert("Microphone access is required to use this feature.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
    }
  };

  const sendAudio = async (audioBlob: Blob) => {
    setIsProcessing(true);
    
    // Add a placeholder message for user
    setMessages(prev => [...prev, { role: "user", content: "..." }]);
    
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.wav");

    try {
      const response = await fetch("http://localhost:8000/api/voice_ask", {
        method: "POST",
        body: formData,
      });
      
      const data: RAGResponse = await response.json();
      
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = { role: "user", content: data.query || "Could not transcribe." };
        return [...updated, { role: "agent", content: data }];
      });
      
    } catch (error) {
      console.error("API Error:", error);
      setMessages(prev => [
        ...prev.slice(0, -1),
        { role: "user", content: "Error communicating with server." },
        { role: "agent", content: { status: "error", query: "", message: "Failed to connect to backend API." } }
      ]);
    } finally {
      setIsProcessing(false);
    }
  };

  const renderAgentMessage = (data: RAGResponse) => {
    if (data.status === "error") {
      return (
        <div className="message agent" style={{ borderColor: 'var(--error-color)' }}>
          <p>{data.message}</p>
        </div>
      );
    }

    return (
      <div className="message agent glass-container" style={{ padding: '24px' }}>
        {data.answer ? (
          <div style={{ marginBottom: '20px' }}>
            <p style={{ fontSize: '1.1rem' }}>{data.answer}</p>
          </div>
        ) : (
          <div style={{ marginBottom: '20px', color: 'var(--warning-color)' }}>
            <p><strong>Refused:</strong> {data.message}</p>
          </div>
        )}
        
        {/* Detailed Metrics Dashboard */}
        <div className="metrics-grid">
          {data.latency_metrics?.total_e2e_ms && (
            <div className="metric-card" style={{ gridColumn: 'span 2', background: 'rgba(59, 130, 246, 0.15)', borderColor: 'rgba(59, 130, 246, 0.3)', border: '1px solid' }}>
              <span className="metric-value">{data.latency_metrics.total_e2e_ms.toFixed(1)}ms</span>
              <span className="metric-label">VOICE → ANSWER (Total)</span>
            </div>
          )}

          {data.latency_metrics?.total_retrieval_ms && (
            <div className="metric-card" style={{ gridColumn: 'span 2', background: 'rgba(139, 92, 246, 0.15)', borderColor: 'rgba(139, 92, 246, 0.3)', border: '1px solid' }}>
              <span className="metric-value">{data.latency_metrics.total_retrieval_ms.toFixed(1)}ms</span>
              <span className="metric-label">Actual Retrieval</span>
            </div>
          )}
          
          {data.latency_metrics?.stt_ms && (
             <div className="metric-card">
               <span className="metric-value">{data.latency_metrics.stt_ms.toFixed(1)}ms</span>
               <span className="metric-label">STT</span>
             </div>
          )}
          
          {data.latency_metrics?.query_routing_ms && (
             <div className="metric-card">
               <span className="metric-value">{data.latency_metrics.query_routing_ms.toFixed(1)}ms</span>
               <span className="metric-label">Router</span>
             </div>
          )}

          {data.latency_metrics?.query_embedding_ms !== undefined && (
            <div className="metric-card">
              <span className="metric-value">{data.latency_metrics.query_embedding_ms.toFixed(1)}ms</span>
              <span className="metric-label">Embed</span>
            </div>
          )}

          {data.latency_metrics?.dense_retrieval_ms !== undefined && (
            <div className="metric-card">
              <span className="metric-value">{data.latency_metrics.dense_retrieval_ms.toFixed(1)}ms</span>
              <span className="metric-label">Dense</span>
            </div>
          )}

          {data.latency_metrics?.sparse_retrieval_ms !== undefined && (
            <div className="metric-card">
              <span className="metric-value">{data.latency_metrics.sparse_retrieval_ms.toFixed(1)}ms</span>
              <span className="metric-label">Sparse</span>
            </div>
          )}

          {data.latency_metrics?.rrf_fusion_ms !== undefined && (
            <div className="metric-card">
              <span className="metric-value">{data.latency_metrics.rrf_fusion_ms.toFixed(1)}ms</span>
              <span className="metric-label">RRF</span>
            </div>
          )}
          
          {data.latency_metrics?.generation_ms !== undefined && (
            <div className="metric-card">
              <span className="metric-value">{data.latency_metrics.generation_ms.toFixed(1)}ms</span>
              <span className="metric-label">LLM Gen</span>
            </div>
          )}
          
          {data.latency_metrics?.grounding_ms !== undefined && (
            <div className="metric-card">
              <span className="metric-value">{data.latency_metrics.grounding_ms.toFixed(1)}ms</span>
              <span className="metric-label">Grounding</span>
            </div>
          )}
        </div>
        
        {/* Routing & Guardrails Display */}
        <div className="metrics-grid" style={{ marginTop: '12px', paddingTop: '12px' }}>
          {data.routing && (
            <div className="metric-card">
              <span className="metric-value" style={{ fontSize: '0.9rem', color: 'var(--text-primary)' }}>
                {data.routing.language}
              </span>
              <span className="metric-label">Detected Lang</span>
            </div>
          )}
          
          {data.routing && (
            <div className="metric-card">
              <span className="metric-value" style={{ fontSize: '0.9rem', color: 'var(--text-primary)' }}>
                {data.routing.intent}
              </span>
              <span className="metric-label">Intent</span>
            </div>
          )}

          {data.retrieval_confidence && (
            <div className="metric-card">
              <span className={`badge ${data.retrieval_confidence}`}>
                {data.retrieval_confidence}
              </span>
              <span className="metric-label">Confidence</span>
            </div>
          )}

          {data.grounding && (
            <div className="metric-card">
              <span className={`badge ${data.grounding.status}`}>
                {data.grounding.status}
              </span>
              <span className="metric-label">Grounding</span>
            </div>
          )}
        </div>
        
        {/* Source Citations */}
        {data.sources && data.sources.length > 0 && (
          <div style={{ marginTop: '20px', borderTop: '1px solid var(--surface-border)', paddingTop: '16px' }}>
            <h4 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '8px', textTransform: 'uppercase' }}>
              Sources Retrieved
            </h4>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {data.sources.slice(0, 3).map((src, i) => (
                <div key={i} style={{ 
                  background: 'rgba(255,255,255,0.05)', 
                  padding: '6px 12px', 
                  borderRadius: '16px',
                  fontSize: '0.8rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}>
                  <span style={{ color: 'var(--accent-color)' }}>★ {src.relevance}</span>
                  <span>{src.strategy}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <main className="app-container">
      <div className="header">
        <h1 className="gradient-text">HH Goa 2026</h1>
        <p>Voice-Enabled Multilingual RAG</p>
      </div>
      
      {/* Benchmark Banner */}
      <div className="benchmark-banner">
        <div>
          <h3 style={{ fontSize: '1.1rem', marginBottom: '4px' }}>Production Pipeline Evaluated</h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Verified 100-query Hindi IndicMSMARCO evaluation (Benchmark-Subset Results).<br/>
            Configuration: <strong>Hybrid RRF (BGE-M3 + BM25)</strong>
          </p>
        </div>
        <div className="benchmark-stats">
          <div className="bench-stat">
            <span className="bench-val">80.0%</span>
            <span className="bench-label">Recall@10</span>
          </div>
          <div className="bench-stat">
            <span className="bench-val">50.4ms</span>
            <span className="bench-label">Retrieval P50</span>
          </div>
          <div className="bench-stat">
            <span className="bench-val" style={{ color: 'var(--warning-color)'}}>54.9ms</span>
            <span className="bench-label">Retrieval P70</span>
          </div>
          <div className="bench-stat">
            <span className="bench-val" style={{ color: 'var(--error-color)'}}>65.4ms</span>
            <span className="bench-label">Retrieval P100</span>
          </div>
        </div>
      </div>

      <div className="glass-container" style={{ minHeight: '60vh', display: 'flex', flexDirection: 'column' }}>
        
        {/* Chat Area */}
        <div className="chat-window" style={{ flexGrow: 1, paddingBottom: '100px' }}>
          {messages.length === 0 ? (
            <div style={{ textAlign: 'center', color: 'var(--text-secondary)', marginTop: '40px' }}>
              <p>Tap the microphone and ask a question.</p>
              <p style={{ fontSize: '0.85rem', marginTop: '8px' }}>E.g., "What is the capital of India?" or "भारत की राजधानी क्या है?"</p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              msg.role === "user" ? (
                <div key={idx} className="message user">
                  {typeof msg.content === 'string' ? msg.content : "..."}
                </div>
              ) : (
                <div key={idx} style={{ width: '100%' }}>
                  {renderAgentMessage(msg.content as RAGResponse)}
                </div>
              )
            ))
          )}
          
          {isProcessing && (
            <div className="message agent" style={{ alignSelf: 'flex-start', padding: '12px 20px', display: 'flex', gap: '8px', alignItems: 'center', width: 'fit-content' }}>
              <div className="spinner" style={{ width: '16px', height: '16px', border: '2px solid var(--accent-color)', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Processing pipeline...</span>
            </div>
          )}
        </div>

        {/* Floating Controls */}
        <div style={{ 
          position: 'fixed', 
          bottom: '40px', 
          left: '50%', 
          transform: 'translateX(-50%)',
          zIndex: 10 
        }}>
          {/* Strands animation backdrop while recording */}
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: '280px',
            height: '280px',
            borderRadius: '50%',
            overflow: 'hidden',
            opacity: isRecording ? 1 : 0,
            transition: 'opacity 0.4s ease',
            pointerEvents: 'none',
            zIndex: 0
          }}>
            {isRecording && (
              <Strands
                colors={["#8B5CF6", "#3B82F6", "#06B6D4"]}
                count={4}
                speed={0.8}
                amplitude={1.2}
                waviness={1.2}
                thickness={0.8}
                glow={3.0}
                taper={2}
                spread={1.2}
                intensity={0.8}
                saturation={1.8}
                opacity={1}
                scale={1.5}
                glass={false}
                style={{}}
              />
            )}
          </div>

          <div className={`mic-btn-wrapper ${isRecording ? 'recording' : ''}`} style={{ position: 'relative', zIndex: 2 }}>
            {isRecording && <div className="ripple"></div>}
            <button 
              className={`mic-btn ${isRecording ? 'recording' : ''}`}
              onMouseDown={startRecording}
              onMouseUp={stopRecording}
              onTouchStart={startRecording}
              onTouchEnd={stopRecording}
              disabled={isProcessing}
            >
              <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
              </svg>
            </button>
          </div>
          <div style={{ textAlign: 'center', fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '8px', position: 'relative', zIndex: 2 }}>
            {isRecording ? "Release to Send" : "Hold to Speak"}
          </div>
        </div>

      </div>
      
      <style dangerouslySetInnerHTML={{__html: `
        @keyframes spin { 100% { transform: rotate(360deg); } }
      `}} />
    </main>
  );
}
