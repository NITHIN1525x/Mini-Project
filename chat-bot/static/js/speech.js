(function(){
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const hasSTT = !!SR;
  const hasTTS = !!window.speechSynthesis;

  // Enhanced speech recognition with continuous listening
  function listen({ lang="en-IN", onResult, onEnd, onStart } = {}) {
    if (!hasSTT) return () => {};
    const rec = new SR();
    rec.lang = lang;
    rec.interimResults = true;  // Get interim results for more responsive UI
    rec.continuous = false;     // Changed to false to prevent multiple captures
    rec.maxAlternatives = 1;

    let finalTranscript = '';
    let lastInterimTranscript = '';
    
    rec.onresult = e => {
      const result = e.results[e.results.length - 1];
      const transcript = result[0].transcript;
      
      if (result.isFinal) {
        finalTranscript = transcript.trim();
        onResult?.(finalTranscript, false);
        // Auto-stop after getting final result
        try { rec.stop(); } catch {}
      } else {
        // Only update if interim result changed
        if (transcript !== lastInterimTranscript) {
          lastInterimTranscript = transcript;
          onResult?.(transcript, true);
        }
      }
    };

    rec.onstart = () => {
      finalTranscript = '';
      onStart?.();
    };

    rec.onend = () => onEnd?.();
    rec.start();
    return () => { try { rec.abort(); } catch {} };
  }

  // Text-to-speech function
  function speak(text, { lang="en-IN", rate=1 } = {}) {
    if (!hasTTS || !text) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang; 
    u.rate = rate;
    window.speechSynthesis.cancel(); // stop any previous
    window.speechSynthesis.speak(u);
  }

  // Export functions
  window.speechKit = { 
    hasSTT, 
    hasTTS, 
    listen, 
    speak,
    // Helper to stop any ongoing recognition
    stop: () => {
      window.speechSynthesis.cancel();
    }
  };
})();
