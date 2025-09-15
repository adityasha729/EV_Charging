// chatbot.js - Handles all chatbot-related UI and logic for the EV Charging Game

// Highlight stations based on chatbot colors
function highlightStations(colors) {
    for (let i = 1; i <= 3; i++) {
        const el = document.getElementById('kiosk-' + i);
        if (!el) continue;
        el.classList.remove('station-green', 'station-yellow', 'station-red', 'station-gray');
        if (colors && colors[i]) {
            el.classList.add('station-' + colors[i]);
        }
    }
}

// Chatbot panel logic
function toggleChatbotPanel() {
    const panel = document.getElementById('chatbot-panel');
    panel.style.display = (panel.style.display === 'block') ? 'none' : 'block';
    if (panel.style.display === 'block') {
        document.getElementById('chatbot-input').focus();
    }
}

// Chatbot message handling
const chatbotChatbox = document.getElementById('chatbot-chatbox');
const chatbotInput = document.getElementById('chatbot-input');
const chatbotSendBtn = document.getElementById('chatbot-send-btn');
const chatbotClearBtn = document.getElementById('chatbot-clear-btn');
const chatbotModelSelect = document.getElementById('chatbot-model-select');

function appendChatbotMessage(sender, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'chatbot-message ' + sender;
    msgDiv.textContent = (sender === 'user' ? 'You: ' : 'Bot: ') + text;
    chatbotChatbox.appendChild(msgDiv);
    chatbotChatbox.scrollTop = chatbotChatbox.scrollHeight;
}

chatbotSendBtn.onclick = function() {
    const message = chatbotInput.value.trim();
    if (!message) return;
    appendChatbotMessage('user', message);
    chatbotInput.value = '';
    const model = chatbotModelSelect ? chatbotModelSelect.value : 'gemini';
    fetch('/send_message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bot: model, message })
    })
    .then(res => res.json())
    .then(data => {
        if (data.reply) {
            appendChatbotMessage('bot', data.reply);
        } else if (data.error) {
            appendChatbotMessage('bot', 'Error: ' + data.error);
        }
        if (data.colors) {
            highlightStations(data.colors);
        }
    })
    .catch(() => appendChatbotMessage('bot', 'Error connecting to server.'));
};

chatbotInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') chatbotSendBtn.click();
});

chatbotClearBtn.onclick = function() {
    chatbotChatbox.innerHTML = '';
};

// Add highlight color styles (if not already present)
(function addHighlightStyles() {
    if (document.getElementById('chatbot-highlight-style')) return;
    const style = document.createElement('style');
    style.id = 'chatbot-highlight-style';
    style.innerHTML = `
    .station-green { box-shadow: 0 0 0 4px #4CAF50, 0 8px 32px rgba(0,0,0,0.1) !important; border-color: #4CAF50 !important; }
    .station-yellow { box-shadow: 0 0 0 4px #FFD600, 0 8px 32px rgba(0,0,0,0.1) !important; border-color: #FFD600 !important; }
    .station-red { box-shadow: 0 0 0 4px #F44336, 0 8px 32px rgba(0,0,0,0.1) !important; border-color: #F44336 !important; }
    .station-gray { box-shadow: 0 0 0 4px #BDBDBD, 0 8px 32px rgba(0,0,0,0.1) !important; border-color: #BDBDBD !important; }
    `;
    document.head.appendChild(style);
})();
