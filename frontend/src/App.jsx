import React, { useState, useEffect, useRef } from 'react';
import { 
  MessageSquare, 
  Bell, 
  Volume2, 
  VolumeX, 
  Mic, 
  Send, 
  ThumbsUp, 
  ThumbsDown, 
  X,
  Brain,
  Palette
} from 'lucide-react';
import './App.css';

const BACKEND_URL = "http://127.0.0.1:8000";
const WS_SCHEME = window.location.protocol === "https:" ? "wss" : "ws";
const WS_URL = `${WS_SCHEME}://127.0.0.1:8000/ws/chat/`;

const RECOMMENDED_CHIPS = [
  { label: "Timings for 1st Year", query: "What are the timings for first year?" },
  { label: "Highest Package offered?", query: "What is the highest package offered?" },
  { label: "Attendance Rules", query: "What is the minimum attendance required?" },
  { label: "Internships details", query: "Does the college provide internship opportunities?" },
  { label: "COMEDK Fees", query: "What is the fee structure for COMEDK?" }
];

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("scb.theme") || "slate"); // 'slate' | 'cyberpunk'
  
  // Chat state
  const [messages, setMessages] = useState([
    { id: 1, text: "Hello! I'm your Smart Campus Bot. Ask me anything about the college.", who: "bot" }
  ]);
  const [inputVal, setInputVal] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [typingText, setTypingText] = useState('');
  const [isWsOnline, setIsWsOnline] = useState(false);
  
  // TTS & STT state
  const [ttsEnabled, setTtsEnabled] = useState(() => localStorage.getItem("ttsOn") !== "false");
  const [isListening, setIsListening] = useState(false);
  
  // Reminders state
  const [showRemindersModal, setShowRemindersModal] = useState(false);
  const [remTitle, setRemTitle] = useState('');
  const [remWhen, setRemWhen] = useState('');
  const [reminders, setReminders] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("scb.reminders") || "[]");
    } catch {
      return [];
    }
  });


  // Toast state
  const [toasts, setToasts] = useState([]);

  // Refs
  const socketRef = useRef(null);
  const messagesEndRef = useRef(null);
  const recognitionRef = useRef(null);

  // Connect WebSocket and initialize on mount
  useEffect(() => {
    connectWebSocket();
    reminders.filter(r => !r.done).forEach(scheduleReminderAlarm);

    // Setup speech recognition
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRec) {
      const rec = new SpeechRec();
      rec.lang = 'en-IN';
      rec.interimResults = false;
      rec.continuous = false;
      
      rec.onstart = () => {
        setIsListening(true);
      };
      
      rec.onresult = (e) => {
        const text = e.results[0][0].transcript;
        setInputVal(text);
        handleSendMessage(text);
      };
      
      rec.onend = () => {
        setIsListening(false);
      };
      
      recognitionRef.current = rec;
    }

    if ("Notification" in window && Notification.permission === "default") {
      Notification.requestPermission();
    }

    return () => {
      if (socketRef.current) socketRef.current.close();
    };
  }, []);

  // Save reminders and theme changes
  useEffect(() => {
    localStorage.setItem("scb.reminders", JSON.stringify(reminders));
  }, [reminders]);

  useEffect(() => {
    localStorage.setItem("scb.theme", theme);
  }, [theme]);

  // Scroll to bottom of chat list
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, typingText, isTyping]);

  const connectWebSocket = () => {
    try {
      const socket = new WebSocket(WS_URL);
      
      socket.onopen = () => {
        setIsWsOnline(true);
      };

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === "connection_established") {
          setIsWsOnline(true);
          return;
        }

        if (data.type === "message_start") {
          setIsTyping(true);
          setTypingText('');
          return;
        }

        if (data.type === "token") {
          setTypingText(prev => prev + data.char);
          return;
        }

        if (data.type === "chat_message") {
          setIsTyping(false);
          setTypingText('');
          
          const newBotMsg = {
            id: data.message_id || Date.now(),
            text: data.reply,
            who: "bot",
            meta: `(${data.tag} - ${(data.confidence * 100).toFixed(0)}% confidence - ${(data.uncertainty * 100).toFixed(0)}% uncertainty)`,
            tag: data.tag,
            feedbackSubmitted: false
          };
          
          setMessages(prev => [...prev, newBotMsg]);
          speakText(data.reply);
          return;
        }

        if (data.type === "error") {
          setIsTyping(false);
          setMessages(prev => [...prev, {
            id: Date.now(),
            text: data.error || "An error occurred.",
            who: "bot"
          }]);
        }
      };

      socket.onclose = () => {
        setIsWsOnline(false);
        console.log("WebSocket closed. Reconnecting in 3 seconds...");
        setTimeout(connectWebSocket, 3000);
      };

      socketRef.current = socket;
    } catch (e) {
      console.error("WS error:", e);
      setIsWsOnline(false);
    }
  };

  // Speak bot voice
  const speakText = (text) => {
    if (ttsEnabled && window.speechSynthesis) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "en-IN";
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utterance);
    }
  };

  // Toggle TTS setting
  const handleToggleTts = () => {
    const nextVal = !ttsEnabled;
    setTtsEnabled(nextVal);
    localStorage.setItem("ttsOn", String(nextVal));
    showToast("Voice Mode Changed", `Text-to-Speech is now ${nextVal ? 'ON' : 'OFF'}`);
  };

  // Toggle Themes
  const handleToggleTheme = () => {
    const nextTheme = theme === 'slate' ? 'cyberpunk' : 'slate';
    setTheme(nextTheme);
    showToast("Theme Preference Saved", `Display mode set to ${nextTheme === 'slate' ? 'Slate Ocean' : 'Neon Cyberpunk'}`);
  };

  // Notification Toast Helper
  const showToast = (title, message) => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, title, message }]);
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 4500);
  };

  // Trigger Speech Dictation
  const handleToggleListening = () => {
    if (!recognitionRef.current) {
      alert("Speech recognition is not supported in your browser.");
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
    } else {
      recognitionRef.current.start();
    }
  };

  // REST API response fallback if WS is offline
  const sendMessageRest = async (text) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/chat/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, lang: "en" })
      });
      
      const data = await res.json();
      if (res.ok) {
        const metaStr = `(${data.tag} - ${(data.confidence * 100).toFixed(0)}% confidence - ${(data.uncertainty * 100).toFixed(0)}% uncertainty)`;
        setMessages(prev => [...prev, {
          id: data.message_id || Date.now(),
          text: data.reply,
          who: "bot",
          meta: metaStr,
          tag: data.tag,
          feedbackSubmitted: false
        }]);
        speakText(data.reply);
      } else {
        setMessages(prev => [...prev, { id: Date.now(), text: data.error || "Server error.", who: "bot" }]);
      }
    } catch {
      setMessages(prev => [...prev, { id: Date.now(), text: "Network error occurred.", who: "bot" }]);
    }
    setIsTyping(false);
  };

  const handleSendMessage = (overrideText = "") => {
    const query = (overrideText || inputVal).trim();
    if (!query) return;

    setMessages(prev => [...prev, { id: Date.now(), text: query, who: "user" }]);
    setInputVal('');
    setIsTyping(true);

    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ text: query, lang: "en" }));
    } else {
      sendMessageRest(query);
    }
  };

  // User feedback POST rating
  const submitFeedback = async (messageId, rating, predictedIntent) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/feedback/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message_id: messageId,
          rating,
          corrected_intent: predictedIntent,
          feedback_text: rating >= 3 ? "Helpful answer" : "Needs improvement"
        })
      });

      if (res.ok) {
        setMessages(prev => prev.map(m => m.id === messageId ? { ...m, feedbackSubmitted: true, rating } : m));
        showToast("Feedback Saved", rating >= 3 ? "Thanks for rating positive!" : "Query marked for active learning retraining.");
      }
    } catch (e) {
      console.error(e);
    }
  };

  // Save alarm reminders
  const handleSaveReminder = () => {
    if (!remTitle.trim() || !remWhen) {
      alert("Please fill in both fields.");
      return;
    }

    const id = Date.now().toString(36) + Math.random().toString(36).substr(2);
    const newRem = { id, title: remTitle, when: remWhen, done: false };
    
    setReminders(prev => [...prev, newRem]);
    scheduleReminderAlarm(newRem);
    
    setRemTitle('');
    setRemWhen('');
    showToast("Alarm Set Successfully", `Reminder scheduled: "${newRem.title}"`);
  };

  const scheduleReminderAlarm = (rem) => {
    const alarmTime = new Date(rem.when).getTime();
    const now = Date.now();
    if (alarmTime > now) {
      setTimeout(() => {
        if ("Notification" in window && Notification.permission === "granted") {
          new Notification("Reminder Alert", { 
            body: `Alert: ${rem.title}`,
            icon: "/favicon.ico"
          });
        }
        
        showToast("Reminder Due", rem.title);
        speakText(`Reminder alert: ${rem.title}`);
        
        setReminders(prev => prev.map(r => r.id === rem.id ? { ...r, done: true } : r));
      }, alarmTime - now);
    }
  };


  return (
    <div className={`app-layout ${theme}`}>
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div>
          <div className="sidebar-brand">
            <div className="brand-logo">
              <Brain size={22} />
            </div>
            <div className="brand-text">
              <h2>Sahyadri AI</h2>
              <span>Smart Campus Bot</span>
            </div>
          </div>
          <div className="sidebar-menu">
            <button className="menu-item active">
              <MessageSquare size={18} />
              Chat Assistant
            </button>
          </div>
        </div>
        
        <div className="sidebar-footer">
          <div className="user-avatar">SA</div>
          <div className="user-info">
            <h4>Sahyadri Admin</h4>
            <p>Developer Mode</p>
          </div>
        </div>
      </aside>

      {/* Main Container */}
      <main className="main-content">
        <nav className="top-nav">
          <div className="top-nav-title">
            Campus Assistant AI
          </div>
          
          <div className="top-nav-actions">
            {/* Real-time Connection status */}
            <div className="connection-badge">
              <span className={`connection-dot ${isWsOnline ? 'online' : ''}`}></span>
              {isWsOnline ? "Live Network" : "API Fallback"}
            </div>

            {/* Theme selector */}
            <button className="chip-btn" onClick={handleToggleTheme} title="Change Theme Style">
              <Palette size={14} />
              {theme === 'slate' ? 'Cyberpunk' : 'Slate Mode'}
            </button>
            <button 
              className={`chip-btn ${ttsEnabled ? 'active' : ''}`} 
              onClick={handleToggleTts}
            >
              {ttsEnabled ? <Volume2 size={14} /> : <VolumeX size={14} />}
              TTS: {ttsEnabled ? 'On' : 'Off'}
            </button>
            <button className="chip-btn" onClick={() => setShowRemindersModal(true)}>
              <Bell size={14} />
              Reminders
            </button>
          </div>
        </nav>
        <div className="chat-container">
            <div className="messages-list">
              {messages.map(msg => (
                <div key={msg.id} className={`msg-wrapper ${msg.who}`}>
                  <div className="msg-bubble">
                    {msg.text}
                  </div>
                  {msg.meta && <div className="msg-meta">{msg.meta}</div>}
                  
                  {msg.who === 'bot' && msg.meta && !msg.feedbackSubmitted && (
                    <div className="msg-feedback">
                      <button 
                        className="feedback-btn" 
                        onClick={() => submitFeedback(msg.id, 4, msg.tag)}
                      >
                        <ThumbsUp size={12} /> Good
                      </button>
                      <button 
                        className="feedback-btn" 
                        onClick={() => submitFeedback(msg.id, 1, msg.tag)}
                      >
                        <ThumbsDown size={12} /> Improve
                      </button>
                    </div>
                  )}

                  {msg.feedbackSubmitted && (
                    <div className="feedback-done">
                      {msg.rating >= 3 ? "Thanks for the feedback!" : "Saved for improvements."}
                    </div>
                  )}
                </div>
              ))}
              
              {isTyping && (
                <div className="msg-wrapper bot">
                  <div className="msg-bubble">
                    {typingText ? typingText : (
                      <div className="typing-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Clickable Quick Action Recommendation Chips */}
              {messages.length === 1 && !isTyping && (
                <div className="chips-container">
                  <h4>Try Asking These Questions:</h4>
                  <div className="chips-grid">
                    {RECOMMENDED_CHIPS.map((c, idx) => (
                      <button 
                        key={idx} 
                        className="chip"
                        onClick={() => handleSendMessage(c.query)}
                      >
                        {c.label}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>

            <div className="chat-composer">
              {/* STT visual equalizer speech-wave */}
              {isListening ? (
                <div className="speech-wave">
                  <span></span>
                  <span></span>
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              ) : (
                <button 
                  className="action-icon-btn"
                  onClick={handleToggleListening}
                  title="Speech to Text dictation"
                >
                  <Mic size={18} />
                </button>
              )}

              <input 
                type="text" 
                className="composer-input"
                placeholder={isListening ? "Listening..." : "Ask about admissions, placements, timings, internal exams..."}
                value={inputVal}
                onChange={e => setInputVal(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSendMessage()}
              />
              <button 
                className="action-icon-btn send-btn"
                onClick={() => handleSendMessage()}
              >
                <Send size={16} />
              </button>
            </div>
          </div>
      </main>

      {/* Reminders modal card */}
      {showRemindersModal && (
        <div className="modal-overlay">
          <div className="modal-card">
            <div className="modal-header">
              <h3><Bell size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }} /> Schedule Reminder</h3>
              <button className="close-modal-btn" onClick={() => setShowRemindersModal(false)}>
                <X size={20} />
              </button>
            </div>
            
            <div className="modal-form">
              <div className="form-group">
                <label>Reminder Subject</label>
                <input 
                  type="text" 
                  className="form-input" 
                  placeholder="e.g. Submitting fee documents, internal exams"
                  value={remTitle}
                  onChange={e => setRemTitle(e.target.value)}
                />
              </div>
              <div className="form-group">
                <label>Alert Time</label>
                <input 
                  type="datetime-local" 
                  className="form-input"
                  value={remWhen}
                  onChange={e => setRemWhen(e.target.value)}
                />
              </div>
              <button className="chip-btn send-btn" onClick={handleSaveReminder} style={{ alignSelf: 'flex-end', padding: '10px 24px', border: 'none', borderRadius: '8px' }}>
                Save Alarm
              </button>
            </div>

            <div className="reminders-list-container">
              <h4>All Reminders</h4>
              <div className="reminders-list">
                {reminders.map(rem => (
                  <div key={rem.id} className={`reminder-item ${rem.done ? 'done' : ''}`}>
                    <div className="reminder-details">
                      <h5>{rem.title}</h5>
                      <p>{new Date(rem.when).toLocaleString()}</p>
                    </div>
                    <span className={`reminder-status ${rem.done ? 'completed' : 'pending'}`}>
                      {rem.done ? 'Fired' : 'Pending'}
                    </span>
                  </div>
                ))}
                {reminders.length === 0 && (
                  <p style={{ textAlign: 'center', fontSize: '12px', color: 'var(--text-muted)', padding: '10px' }}>No reminders scheduled.</p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Slide-in notification overlays */}
      <div className="toast-container">
        {toasts.map(t => (
          <div key={t.id} className="toast">
            <div className="toast-icon">
              <Bell size={18} />
            </div>
            <div>
              <div className="toast-title">{t.title}</div>
              <div className="toast-message">{t.message}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}



