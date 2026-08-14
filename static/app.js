const messages = document.querySelector('#messages');
const form = document.querySelector('#chat-form');
const question = document.querySelector('#question');
const send = document.querySelector('#send');
const statusText = document.querySelector('#status');

function createSessionId() {
  if (typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }

  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, byte => byte.toString(16).padStart(2, '0'));
  return `${hex.slice(0, 4).join('')}-${hex.slice(4, 6).join('')}-${hex.slice(6, 8).join('')}-${hex.slice(8, 10).join('')}-${hex.slice(10).join('')}`;
}

let sessionId = sessionStorage.getItem('localRagSessionId');
if (!sessionId) {
  sessionId = createSessionId();
  sessionStorage.setItem('localRagSessionId', sessionId);
}

function appendMessage(role, content, citations = [], answerHtml = null, itemQuestion = null) {
  const item = document.createElement('article');
  item.className = `message ${role}`;
  if (itemQuestion) {
    const heading = document.createElement('strong');
    heading.className = 'item-question';
    heading.textContent = itemQuestion;
    item.appendChild(heading);
  }
  const answer = document.createElement('div');
  if (role === 'assistant' && answerHtml !== null) {
    answer.innerHTML = answerHtml;
  } else {
    answer.textContent = content;
  }
  item.appendChild(answer);
  if (citations.length) {
    const list = document.createElement('div');
    list.className = 'citations';
    citations.forEach((citation, index) => {
      const line = document.createElement('div');
      line.textContent = `[${index + 1}] ${citation.document_name}，第 ${citation.page_start}-${citation.page_end} 頁`;
      list.appendChild(line);
      if (citation.crop_url) {
        const image = document.createElement('img');
        image.src = citation.crop_url;
        image.alt = `引用圖片 ${index + 1}`;
        list.appendChild(image);
      }
    });
    item.appendChild(list);
  }
  messages.appendChild(item);
  item.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

async function loadHistory() {
  const response = await fetch(`/api/history/${encodeURIComponent(sessionId)}`);
  if (!response.ok) return;
  const body = await response.json();
  body.messages.forEach(item => {
    if (item.role === 'assistant' && Array.isArray(item.items)) {
      item.items.forEach(responseItem => appendMessage(
        'assistant',
        responseItem.answer,
        responseItem.citations || [],
        responseItem.answer_html || null,
        responseItem.question
      ));
    } else {
      appendMessage(item.role, item.content, item.citations || []);
    }
  });
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  const text = question.value.trim();
  if (!text) return;
  appendMessage('user', text);
  question.value = '';
  send.disabled = true;
  statusText.textContent = '正在查詢文件…';
  try {
    const response = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, question: text })
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.message || '查詢失敗');
    body.items.forEach(item => appendMessage(
      'assistant',
      item.answer,
      item.citations || [],
      item.answer_html || null,
      item.question
    ));
    statusText.textContent = '';
  } catch (error) {
    statusText.textContent = error.message;
  } finally { send.disabled = false; question.focus(); }
});

document.querySelector('#clear').addEventListener('click', async () => {
  const response = await fetch(`/api/history/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
  if (!response.ok) {
    statusText.textContent = '清除對話失敗，請稍後重試。';
    return;
  }
  messages.replaceChildren();
  sessionId = createSessionId();
  sessionStorage.setItem('localRagSessionId', sessionId);
  statusText.textContent = '';
});

document.querySelector('#end-chat').addEventListener('click', async () => {
  const response = await fetch(`/api/session/${encodeURIComponent(sessionId)}/end`, { method: 'POST' });
  if (!response.ok) {
    statusText.textContent = '結束對話失敗，請稍後重試。';
    return;
  }
  messages.replaceChildren();
  sessionId = createSessionId();
  sessionStorage.setItem('localRagSessionId', sessionId);
  statusText.textContent = '對話已結束，診斷紀錄已整理。';
});

loadHistory();
