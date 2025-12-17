import { useState, useRef, useEffect } from "react";
import type { FormEvent } from "react";

type Role = "user" | "assistant";
type Mode = "vanilla" | "romance";

interface Message {
  role: Role;
  content: string;
}

interface WolfResponse {
  reply: string;
  audio_url?: string | null;
}

const API_URL = "http://localhost:8000/wolf";

const WolfChat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("vanilla");

  const chatEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const newUserMessage: Message = { role: "user", content: input.trim() };
    const newMessages = [...messages, newUserMessage];

    setMessages(newMessages);
    setInput("");
    setLoading(true);
    setAudioUrl(null);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: newMessages, mode }),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const data: WolfResponse = await res.json();

      const wolfMessage: Message = {
        role: "assistant",
        content: data.reply,
      };

      setMessages((prev) => [...prev, wolfMessage]);

      if (data.audio_url) {
        setAudioUrl(`http://localhost:8000${data.audio_url}`);
      }
    } catch (err) {
      console.error("Wolf API error:", err);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "I tried to answer, but something went wrong with my connection to the den.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="wolf-container">
      <header className="wolf-header">
        <div className="wolf-avatar">🐺</div>
        <div>
          <h1>Chat with Wolf</h1>
          <p>Yautja counselor · chaos philosopher · your problem now.</p>
        </div>
        <div className="wolf-mode-toggle">
          <label className="wolf-switch">
            <input
              type="checkbox"
              checked={mode === "romance"}
              onChange={(e) =>
                setMode(e.target.checked ? "romance" : "vanilla")
              }
            />
            <span className="wolf-slider" />
          </label>
          <span className="wolf-mode-label">
            {mode === "romance" ? "Romance mode" : "Standard"}
          </span>
        </div>
      </header>

      <main className="wolf-chat-area">
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={`wolf-message-row ${
              m.role === "user" ? "wolf-user" : "wolf-assistant"
            }`}
          >
            <div className="wolf-bubble">{m.content}</div>
          </div>
        ))}

        {loading && (
          <div className="wolf-message-row wolf-assistant">
            <div className="wolf-bubble wolf-loading">Wolf is thinking…</div>
          </div>
        )}

        <div ref={chatEndRef} />
      </main>

      <footer className="wolf-footer">
        {audioUrl && (
          <div className="wolf-audio">
            <p>Latest voice reply:</p>
            <audio src={audioUrl} controls autoPlay />
          </div>
        )}

        <form onSubmit={handleSubmit} className="wolf-input-row">
          <input
            className="wolf-input"
            placeholder="Ask Wolf something..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="wolf-send"
          >
            {loading ? "Sending…" : "Send"}
          </button>
        </form>
      </footer>
    </div>
  );
};

export default WolfChat;
