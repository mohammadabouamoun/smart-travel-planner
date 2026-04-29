import React, { useState, useRef, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import ToolTimeline from './ToolTimeline';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface ToolEvent {
  name: string;
  type: 'start' | 'end';
  output?: string;
}

// Simple error boundary to catch any remaining render errors
class ChatErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean }
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  render() {
    if (this.state.hasError) {
      return <div style={{ padding: 20 }}>Something went wrong. Please refresh the page and try again.</div>;
    }
    return this.props.children;
  }
}

// Helper: safely extract a single string from any content that the model might send
function extractText(content: any): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map((item) => {
        if (typeof item === 'string') return item;
        if (typeof item === 'object' && item !== null) {
          // Gemini sometimes sends {text: "..."}
          if ('text' in item) return (item as any).text || '';
          // fallback: any other string-like key
          return String(item);
        }
        return String(item);
      })
      .join('');
  }
  if (typeof content === 'object' && content !== null) {
    // single object with a text field
    if ('text' in content) return (content as any).text || '';
    return JSON.stringify(content);
  }
  return String(content || '');
}

const ChatContent = () => {
  const { user, logout } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [toolEvents, setToolEvents] = useState<ToolEvent[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages, loading]);

  const sendStream = () => {
    if (!input.trim()) return;
    const userMsg: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    setToolEvents([]);
    let streamedTokens = false;

    const token = localStorage.getItem('access_token');
    fetch('http://localhost:8000/chat/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({ query: userMsg.content }),
    })
      .then(response => {
        if (!response.ok) throw new Error('Network response was not ok');
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let assistantIndex: number | null = null;

        const read = () => {
          reader?.read().then(({ done, value }) => {
            if (done) {
              setLoading(false);
              return;
            }
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const jsonStr = line.slice(6).trim();
                if (!jsonStr) continue;
                try {
                  const data = JSON.parse(jsonStr);
                  switch (data.type) {
                    case 'token': {
                      const text = extractText(data.content);
                      if (text) {
                        streamedTokens = true;
                        setMessages(prev => {
                          const updated = [...prev];
                          if (assistantIndex === null) {
                            assistantIndex = updated.length;
                            updated.push({ role: 'assistant', content: text });
                          } else {
                            const currentMsg = updated[assistantIndex];
                            if (!currentMsg || currentMsg.role !== 'assistant') {
                              assistantIndex = updated.length;
                              updated.push({ role: 'assistant', content: text });
                            } else {
                              updated[assistantIndex] = {
                                ...currentMsg,
                                content: currentMsg.content + text,
                              };
                            }
                          }
                          return updated;
                        });
                      }
                      break;
                    }
                    case 'tool_start':
                      setToolEvents(prev => [...prev, { name: data.name, type: 'start' }]);
                      break;
                    case 'tool_end':
                      setToolEvents(prev => [...prev, { name: data.name, type: 'end', output: data.output }]);
                      break;
                    case 'done':
                      assistantIndex = null;
                      if (!streamedTokens && typeof data?.answer === 'string' && data.answer.length > 0) {
                        setMessages(prev => [...prev, { role: 'assistant', content: data.answer }]);
                      }
                      break;
                    case 'error':
                      console.error('Server error:', data.message);
                      break;
                  }
                } catch (parseError) {
                  console.error('Failed to parse SSE data:', jsonStr, parseError);
                }
              }
            }
            read();
          }).catch(err => {
            console.error('Stream read error:', err);
            setLoading(false);
          });
        };
        read();
      })
      .catch(err => {
        console.error('Fetch error:', err);
        setLoading(false);
      });
  };

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2>Smart Travel Planner</h2>
        <div>
          <span>{user?.email}</span>
          <button onClick={logout} style={{ marginLeft: 10 }}>Logout</button>
        </div>
      </div>

      <div style={{
        border: '1px solid #ccc',
        borderRadius: 8,
        padding: 10,
        height: 400,
        overflowY: 'auto',
        marginBottom: 10,
      }}>
        {messages.map((msg, i) => (
          <div key={i} style={{ marginBottom: 12 }}>
            <strong>{msg.role === 'user' ? 'You' : 'Agent'}</strong>
            <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
          </div>
        ))}
        {loading && <div>Thinking...</div>}
        <div ref={messagesEndRef} />
      </div>

      <ToolTimeline events={toolEvents} />

      <form onSubmit={(e) => { e.preventDefault(); sendStream(); }} style={{ marginTop: 10, display: 'flex' }}>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask about travel..."
          disabled={loading}
          style={{ flex: 1, padding: 8 }}
        />
        <button type="submit" disabled={loading || !input.trim()} style={{ padding: 8 }}>
          Send
        </button>
      </form>
    </div>
  );
};

const Chat = () => (
  <ChatErrorBoundary>
    <ChatContent />
  </ChatErrorBoundary>
);

export default Chat;