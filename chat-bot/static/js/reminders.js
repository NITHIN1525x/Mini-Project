(function(){
  const KEY = "scb.reminders";

  function load() {
    try { return JSON.parse(localStorage.getItem(KEY) || "[]"); } catch { return []; }
  }
  function save(list) {
    localStorage.setItem(KEY, JSON.stringify(list));
  }

  async function ensurePermission() {
    if (!("Notification" in window)) return false;
    if (Notification.permission === "granted") return true;
    if (Notification.permission !== "denied") {
      const perm = await Notification.requestPermission();
      return perm === "granted";
    }
    return false;
  }

  function showToast(title, message) {
    const toastContainer = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `
      <div class="toast-icon">
        <i class="fas fa-bell"></i>
      </div>
      <div class="toast-content">
        <div class="toast-title">${title}</div>
        <div class="toast-message">${message}</div>
      </div>
    `;

    toastContainer.appendChild(toast);

    // Auto remove after 5 seconds
    setTimeout(() => {
      toast.style.animation = 'slideOut 0.3s ease-out forwards';
      setTimeout(() => toast.remove(), 300);
    }, 5000);
  }

  function notify(title, body) {
    if (!("Notification" in window)) return;
    // Show desktop notification
    new Notification(title || "Reminder", { 
      body: body || "",
      icon: "/static/images/favicon.ico",
      badge: "/static/images/favicon.ico"
    });
    
    // Show toast notification
    showToast(title || "Reminder", body || "");
  }

  function upsert(rem) {
    const list = load();
    const id = Date.now().toString(36) + Math.random().toString(36).substr(2);
    list.push({ ...rem, id, done: false });
    save(list);
    scheduleReminder({ ...rem, id });
  }

  function scheduleReminder(rem) {
    const when = new Date(rem.when).getTime();
    const now = Date.now();
    if (when > now) {
      setTimeout(() => {
        notify(rem.title, `⏰ Time for: ${rem.title}`);
        const list = load();
        const reminder = list.find(r => r.id === rem.id);
        if (reminder) {
          reminder.done = true;
          save(list);
          updateReminderList();
          try { 
            if (window.speechKit?.hasTTS) {
              window.speechKit.speak(`Reminder: ${rem.title}`); 
            }
          } catch {}
        }
      }, when - now);
    }
  }

  function getAllReminders() {
    return load().sort((a, b) => new Date(b.when) - new Date(a.when));
  }

  function upcoming() {
    const now = Date.now();
    return load()
      .filter(r => !r.done && new Date(r.when).getTime() >= now)
      .sort((a,b) => new Date(a.when) - new Date(b.when));
  }

  function updateReminderList() {
    const remList = document.getElementById('remList');
    if (!remList) return;
    
    const allReminders = getAllReminders();
    remList.innerHTML = allReminders.map(rem => `
      <li class="reminder-item ${rem.done ? 'done' : ''}" data-id="${rem.id}">
        <div class="reminder-content">
          <div class="reminder-title">${rem.title}</div>
          <div class="reminder-time">${new Date(rem.when).toLocaleString()}</div>
          <div class="reminder-status">${rem.done ? '✓ Completed' : '⏳ Pending'}</div>
        </div>
      </li>
    `).join('');

    // Reschedule upcoming reminders (in case of page refresh)
    allReminders.filter(r => !r.done).forEach(scheduleReminder);
  }

  // Initial setup
  document.addEventListener('DOMContentLoaded', () => {
    ensurePermission();
    updateReminderList();
  });

  window.reminders = { 
    load, 
    save, 
    upsert, 
    upcoming, 
    getAllReminders, 
    ensurePermission, 
    updateReminderList 
  };
})();

