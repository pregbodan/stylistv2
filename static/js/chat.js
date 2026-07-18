// HARDDX — chat client logic (multi-turn: clarify -> diagnose -> guided steps)

const chatFeed = document.getElementById('chatFeed');
const composerForm = document.getElementById('composerForm');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const attachBtn = document.getElementById('attachBtn');
const imageInput = document.getElementById('imageInput');
const imagePreviewBar = document.getElementById('imagePreviewBar');
const imagePreviewThumb = document.getElementById('imagePreviewThumb');
const imagePreviewName = document.getElementById('imagePreviewName');
const imagePreviewRemove = document.getElementById('imagePreviewRemove');
const diagnosisTemplate = document.getElementById('diagnosisTemplate');
const stepTemplate = document.getElementById('stepTemplate');
const suggestionList = document.getElementById('suggestionList');
const historyList = document.getElementById('historyList');
const historyEmpty = document.getElementById('historyEmpty');

const settingsBtn = document.getElementById('settingsBtn');
const settingsOverlay = document.getElementById('settingsOverlay');
const settingsCancel = document.getElementById('settingsCancel');
const settingsSave = document.getElementById('settingsSave');
const apiKeyInput = document.getElementById('apiKeyInput');

const RESTART_PATTERNS = /\b(new\s+(issue|problem)|different\s+(issue|problem)|another\s+(issue|problem)|something\s+else|start\s+over)\b/i;
const HISTORY_STORAGE_KEY = 'harddx_session_history_v1';
const INITIAL_BOT_MESSAGE = "Describe the fault you're experiencing, or attach a photo - what happens, any sounds, lights, or error messages - and I'll ask a couple of quick questions, then walk you through the fix one step at a time.";

let selectedImageFile = null;
let activeStepForm = null; // tracks the most recent unanswered step-reply form
const conversationHistory = [];

imagePreviewBar.hidden = true;

function loadSavedHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return;
    parsed.forEach(item => {
      if (!item || typeof item.role !== 'string' || typeof item.content !== 'string') return;
      conversationHistory.push({ role: item.role, content: item.content });
    });
  } catch (err) {
    // Ignore bad local storage data and start fresh.
  }
}

function persistHistory() {
  try {
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(conversationHistory.slice(-16)));
  } catch (err) {
    // Ignore storage quota or privacy mode failures.
  }
}

function renderHistory() {
  if (!historyList) return;
  historyList.innerHTML = '';

  if (conversationHistory.length === 0) {
    const empty = document.createElement('li');
    empty.className = 'history-empty';
    empty.id = 'historyEmpty';
    empty.textContent = 'No messages yet.';
    historyList.appendChild(empty);
    return;
  }

  conversationHistory.slice(-8).forEach(item => {
    const li = document.createElement('li');
    li.className = `history-item history-${item.role}`;
    const role = document.createElement('span');
    role.className = 'history-role';
    role.textContent = item.role === 'assistant' ? 'HARDDX' : 'YOU';
    const content = document.createElement('span');
    content.className = 'history-content';
    content.textContent = item.content;
    li.appendChild(role);
    li.appendChild(content);
    historyList.appendChild(li);
  });
}

// ---------------- Settings modal ----------------

function openSettings() {
  apiKeyInput.value = getGeminiApiKey();
  settingsOverlay.hidden = false;
  apiKeyInput.focus();
}
function closeSettings() { settingsOverlay.hidden = true; }

function saveSettings() {
  setGeminiApiKey(apiKeyInput.value);
  closeSettings();
}

if (settingsBtn) settingsBtn.addEventListener('click', openSettings);
if (settingsCancel) settingsCancel.addEventListener('click', closeSettings);
if (settingsSave) settingsSave.addEventListener('click', saveSettings);
if (settingsOverlay) {
  settingsOverlay.addEventListener('click', (e) => {
    if (e.target === settingsOverlay) closeSettings();
  });
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && settingsOverlay && !settingsOverlay.hidden) {
    closeSettings();
  }
});

// ---------------- Image attach ----------------

attachBtn.addEventListener('click', () => imageInput.click());

imageInput.addEventListener('change', () => {
  const file = imageInput.files[0];
  if (!file) return;
  selectedImageFile = file;
  imagePreviewThumb.src = URL.createObjectURL(file);
  imagePreviewName.textContent = file.name;
  imagePreviewBar.hidden = false;
});

imagePreviewRemove.addEventListener('click', () => {
  selectedImageFile = null;
  imageInput.value = '';
  imagePreviewBar.hidden = true;
});

// ---------------- Feed helpers ----------------

function scrollToBottom() {
  chatFeed.scrollTop = chatFeed.scrollHeight;
}

function pushHistory(role, content) {
  const text = (content || '').trim();
  if (!text) return;
  conversationHistory.push({ role, content: text });
  if (conversationHistory.length > 16) {
    conversationHistory.splice(0, conversationHistory.length - 16);
  }
  persistHistory();
  renderHistory();
}

function snapshotHistory() {
  return conversationHistory.slice(-12);
}

function addUserMessage(text, imageFile) {
  const wrap = document.createElement('div');
  wrap.className = 'msg msg-user';
  const meta = document.createElement('div');
  meta.className = 'msg-meta';
  meta.textContent = 'YOU';
  wrap.appendChild(meta);

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';

  if (imageFile) {
    const img = document.createElement('img');
    img.className = 'msg-image-thumb';
    img.src = URL.createObjectURL(imageFile);
    bubble.appendChild(img);
  }
  if (text) {
    const p = document.createElement('div');
    p.textContent = text;
    bubble.appendChild(p);
  }

  wrap.appendChild(bubble);
  chatFeed.appendChild(wrap);
  scrollToBottom();
}

function addBotText(text) {
  const wrap = document.createElement('div');
  wrap.className = 'msg msg-bot';
  wrap.innerHTML = `<div class="msg-meta">HARDDX</div><div class="msg-bubble"></div>`;
  wrap.querySelector('.msg-bubble').textContent = text;
  chatFeed.appendChild(wrap);
  scrollToBottom();
  pushHistory('assistant', text);
}

function summarizeResponse(data) {
  switch (data.type) {
    case 'clarifying_question':
      return data.question ? `Clarifying question: ${data.question}` : '';
    case 'diagnosis':
      return [
        `Diagnosis: ${data.fault_label}`,
        data.severity ? `Severity: ${data.severity}` : '',
        Array.isArray(data.causes) && data.causes.length ? `Likely causes: ${data.causes.join('; ')}` : '',
        data.step ? `First step: ${data.step}` : '',
      ].filter(Boolean).join(' | ');
    case 'next_step':
      return [
        data.acknowledged_negative ? 'Try the next step.' : '',
        data.step ? `Next step: ${data.step}` : '',
      ].filter(Boolean).join(' ');
    default:
      return '';
  }
}

function addTypingIndicator() {
  const wrap = document.createElement('div');
  wrap.className = 'msg msg-bot msg-typing';
  wrap.id = 'typingIndicator';
  wrap.innerHTML = `
    <div class="msg-meta">HARDDX</div>
    <div class="msg-bubble">
      <span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>
    </div>
  `;
  chatFeed.appendChild(wrap);
  scrollToBottom();
}

function removeTypingIndicator() {
  const el = document.getElementById('typingIndicator');
  if (el) el.remove();
}

function disablePreviousStepForm() {
  if (activeStepForm) {
    const card = activeStepForm.closest('.step-card');
    if (card) card.classList.add('step-answered');
    activeStepForm = null;
  }
}

// ---------------- Rendering each response type ----------------

function addDiagnosisCard(data) {
  const node = diagnosisTemplate.content.cloneNode(true);
  const card = node.querySelector('.diagnosis-card');
  card.classList.add('severity-' + (data.severity || 'medium'));
  node.querySelector('.diagnosis-label').textContent = data.fault_label;
  node.querySelector('.diagnosis-confidence').textContent =
    'confidence ' + Math.round(data.confidence * 100) + '%';
  node.querySelector('.diagnosis-severity').textContent =
    (data.severity || 'unknown') + ' severity · method: ' + data.method;

  const causeList = node.querySelector('.cause-list');
  (data.causes || []).forEach(c => {
    const li = document.createElement('li');
    li.textContent = c;
    causeList.appendChild(li);
  });

  const wrap = document.createElement('div');
  wrap.className = 'msg msg-bot';
  wrap.style.maxWidth = '90%';
  const meta = document.createElement('div');
  meta.className = 'msg-meta';
  meta.textContent = 'HARDDX · DIAGNOSIS';
  wrap.appendChild(meta);
  wrap.appendChild(card);
  chatFeed.appendChild(wrap);
  scrollToBottom();
}

function addStepCard(stepText, checkPrompt, stepNumber, stepTotal, conversationId) {
  disablePreviousStepForm();

  const node = stepTemplate.content.cloneNode(true);
  node.querySelector('.step-number').textContent = `STEP ${stepNumber}`;
  node.querySelector('.step-progress').textContent = `${stepNumber} / ${stepTotal}`;
  node.querySelector('.step-text').textContent = stepText;
  node.querySelector('.step-check').textContent = checkPrompt || '';

  const form = node.querySelector('.step-reply-form');
  const input = node.querySelector('.step-reply-input');

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    sendMessage(text, null);
  });

  const wrap = document.createElement('div');
  wrap.className = 'msg msg-bot';
  wrap.style.maxWidth = '90%';
  const meta = document.createElement('div');
  meta.className = 'msg-meta';
  meta.textContent = 'HARDDX · GUIDED STEP';
  wrap.appendChild(meta);
  wrap.appendChild(node);
  chatFeed.appendChild(wrap);

  activeStepForm = wrap.querySelector('.step-reply-form');
  scrollToBottom();
  activeStepForm.querySelector('.step-reply-input').focus();
}

function handleResponse(data) {
  switch (data.type) {
    case 'conversational':
    case 'no_match':
    case 'resolved':
    case 'escalate':
      disablePreviousStepForm();
      addBotText(data.reply);
      break;

    case 'clarifying_question': {
      const prefix = data.question_total > 1
        ? `(${data.question_number}/${data.question_total}) `
        : '';
      addBotText(prefix + data.question);
      break;
    }

    case 'diagnosis':
      addDiagnosisCard(data);
      addStepCard(data.step, data.step_check, data.step_number, data.step_total, data.conversation_id);
      break;

    case 'next_step': {
      if (data.acknowledged_negative) {
        addBotText("Got it — let's try the next step.");
      }
      addStepCard(data.step, data.step_check, data.step_number, data.step_total);
      break;
    }

    case 'error':
    default:
      addBotText(data.reply || 'Something went wrong. Please try again.');
      break;
  }
}

// ---------------- Sending ----------------

async function sendMessage(text, imageFile) {
  if (RESTART_PATTERNS.test(text)) {
    conversationHistory.length = 0;
    persistHistory();
    renderHistory();
  }
  const history = snapshotHistory();
  addUserMessage(text, imageFile);
  pushHistory('user', imageFile ? (text ? `${text} [image attached]` : '[image attached]') : text);
  userInput.value = '';
  sendBtn.disabled = true;

  let imageFinding = null;

  if (imageFile) {
    imagePreviewBar.hidden = true;
    selectedImageFile = null;
    imageInput.value = '';

    if (!getGeminiApiKey()) {
      removeTypingIndicator();
      addBotText(
        "I don't have a Gemini API key set up yet for image analysis. " +
        "Click \"⚙ VISION KEY\" at the top to add your free Google AI Studio key, " +
        "then resend the photo. In the meantime, describing the issue in words works great too."
      );
      sendBtn.disabled = false;
      return;
    }

    addTypingIndicator();
    try {
      imageFinding = await analyzeImageWithGemini(imageFile);
    } catch (err) {
      removeTypingIndicator();
      if (err.message === 'NO_API_KEY') {
        addBotText('Please add your Gemini API key via "⚙ VISION KEY" first.');
        sendBtn.disabled = false;
        return;
      }
      if (err.message === 'INVALID_API_KEY') {
        addBotText('That Gemini API key looks invalid or unauthorized. Please check it in "⚙ VISION KEY".');
        sendBtn.disabled = false;
        return;
      }
      addBotText('Primary image analysis unavailable — trying backup service...');
      addTypingIndicator();
      try {
        imageFinding = await analyzeImageWithPuter(imageFile);
        removeTypingIndicator();
        if (imageFinding && imageFinding.description) {
          addBotText('I can see: ' + imageFinding.description);
          addTypingIndicator();
        }
      } catch (puterErr) {
        removeTypingIndicator();
        imageFinding = null;
        addBotText('Image analysis is temporarily unavailable — I\'ll diagnose this from your description using my local models.');
      }
    }
  }

  addTypingIndicator();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, image_finding: imageFinding, history }),
    });
    const data = await res.json();
    removeTypingIndicator();
    handleResponse(data);
    if (data.type === 'diagnosis' || data.type === 'next_step') {
      const summary = summarizeResponse(data);
      if (summary) pushHistory('assistant', summary);
    } else if (data.type === 'resolved' || data.type === 'escalate') {
      conversationHistory.length = 0;
      persistHistory();
      renderHistory();
    }
  } catch (err) {
    removeTypingIndicator();
    addBotText('Connection error — please check that the server is running and try again.');
  } finally {
    sendBtn.disabled = false;
    userInput.focus();
  }
}

composerForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = userInput.value.trim();
  const file = selectedImageFile;
  if (!text && !file) return;
  sendMessage(text, file);
});

suggestionList.addEventListener('click', (e) => {
  const li = e.target.closest('li');
  if (!li) return;
  sendMessage(li.textContent.trim(), null);
});

userInput.focus();
loadSavedHistory();
if (conversationHistory.length === 0) {
  conversationHistory.push({ role: 'assistant', content: INITIAL_BOT_MESSAGE });
  persistHistory();
}
renderHistory();
