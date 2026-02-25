(function(){
  const msgs = document.getElementById("messages");
  const input = document.getElementById("input");
  const send = document.getElementById("send");
  const mic = document.getElementById("mic");
  const ttsToggle = document.getElementById("ttsToggle");
  const bell = document.getElementById("bell");

  // English-only defaults
  const UI_LANG = 'en';
  const VOICE_LANG = 'en-IN';

  // restore TTS preference
  const ttsOn = localStorage.getItem("ttsOn") !== "false";
  ttsToggle.setAttribute("aria-pressed", String(!!ttsOn));
  ttsToggle.textContent = ttsOn ? '🔊 TTS: On' : '🔈 TTS: Off';

  function addMessage(text, who="bot", meta="") {
    const li = document.createElement("li");
    li.className = `msg ${who}`;
    li.innerHTML = `<div>${text}</div>${meta?`<div class="meta">${meta}</div>`:""}`;
    msgs.appendChild(li);
    msgs.scrollTop = msgs.scrollHeight;
  }
  async function ask(text){
    addMessage(text, "user");
    input.value = "";
    try {
      const res = await fetch("/api/chat/", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ text, lang: UI_LANG, voiceLang: VOICE_LANG })
      });
      const data = await res.json();
      if (data.error) {
        addMessage(data.error, "bot");
        return;
      }
      const meta = `(${data.tag} • ${(data.confidence*100).toFixed(0)}%)`;
      addMessage(data.reply, "bot", meta);

      if (ttsToggle.getAttribute("aria-pressed")==="true" && window.speechKit?.hasTTS) {
        window.speechKit.speak(data.reply, { lang: VOICE_LANG });
      }
    } catch (e) {
      addMessage("Network error. Please try again.", "bot");
    }
  }

  // events
  send.addEventListener("click", () => {
    const text = input.value.trim();
    if (text) ask(text);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") send.click();
  });

  // Track if we're currently listening
  let isListening = false;
  let stopListening = null;

  mic.addEventListener("click", () => {
    if (!window.speechKit?.hasSTT) {
      return alert('Speech recognition not supported in your browser');
    }

    if (isListening && stopListening) {
      stopListening();
      isListening = false;
      mic.classList.remove('active');
      return;
    }

    // Clear previous input when starting new recording
    input.value = '';
    mic.classList.add('active');
    isListening = true;

    stopListening = window.speechKit.listen({
      lang: VOICE_LANG,
      onStart: () => {
        input.placeholder = 'Listening...';
      },
      onResult: (text, isInterim) => {
        input.value = text;
        if (!isInterim && text.trim()) {
          mic.classList.remove('active');
          isListening = false;
          ask(text); // Auto-send when we get final result
        }
      },
      onEnd: () => {
        input.placeholder = 'Type a message…';
        mic.classList.remove('active');
        isListening = false;
      }
    });
  });

  // Sync UI and voice languages
  ttsToggle.addEventListener("click", () => {
    const isOn = ttsToggle.getAttribute("aria-pressed")==="true";
    ttsToggle.setAttribute("aria-pressed", String(!isOn));
    ttsToggle.textContent = !isOn ? '🔊 TTS: On' : '🔈 TTS: Off';
    localStorage.setItem("ttsOn", String(!isOn));
  });

  // reminders modal
  const modal = document.getElementById("reminderModal");
  const closeRem = document.getElementById("closeRem");
  const saveRem = document.getElementById("saveRem");
  const remTitle = document.getElementById("remTitle");
  const remWhen = document.getElementById("remWhen");
  const remList = document.getElementById("remList");

  function refreshReminders(){
    const list = window.reminders.upcoming();
    remList.innerHTML = "";
    list.forEach(r => {
      const li = document.createElement("li");
      li.textContent = `${r.title} — ${new Date(r.when).toLocaleString()}`;
      remList.appendChild(li);
    });
  }

  bell.addEventListener("click", async () => {
    const ok = await window.reminders.ensurePermission();
    if (!ok) alert("Enable notifications to get reminders.");
    refreshReminders();
    modal.classList.remove("hidden");
  });
  closeRem.addEventListener("click", () => modal.classList.add("hidden"));

  saveRem.addEventListener("click", () => {
    const title = remTitle.value.trim();
    const when = remWhen.value;
    if (!title || !when) return alert("Enter title and time.");
    window.reminders.upsert({ id: crypto.randomUUID(), title, when, done: false });
    remTitle.value = ""; remWhen.value = "";
    refreshReminders();
  });

  // greet
  addMessage("Hello! I’m your Smart Campus Bot. Ask me anything about the college.");
})();
