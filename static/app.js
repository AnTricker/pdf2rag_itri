const messages = document.querySelector('#messages');
const form = document.querySelector('#chat-form');
const question = document.querySelector('#question');
const send = document.querySelector('#send');
const attachments = document.querySelector('#attachments');
const preview = document.querySelector('#attachment-preview');
const statusText = document.querySelector('#status-text');
let selectedFiles = [];
let activeStream = null;
let activeLoadingMessage = null;
let isBusy = false;
const maxFiles = Number(form.dataset.maxFiles);
const maxFileBytes = Number(form.dataset.maxFileBytes);
const maxTotalBytes = Number(form.dataset.maxTotalBytes);
const allowedImageTypes = new Set(['image/jpeg', 'image/png', 'image/webp']);

function updateSendState() {
  send.disabled = isBusy || (!question.value.trim() && selectedFiles.length === 0);
}

function setStatus(text = '') {
  statusText.textContent = text;
}

function setBusy(busy) {
  isBusy = busy;
  form.classList.toggle('busy', busy);
  question.disabled = busy;
  attachments.disabled = busy;
  messages.setAttribute('aria-busy', String(busy));
  updateSendState();
}

function appendMessage(role, content, citations = [], answerHtml = null, itemQuestion = null, beforeNode = null) {
  const article = document.createElement('article');
  article.className = `message ${role}`;
  if (itemQuestion) {
    const heading = document.createElement('strong');
    heading.className = 'item-question';
    heading.textContent = itemQuestion;
    article.appendChild(heading);
  }
  const answer = document.createElement('div');
  if (role === 'assistant' && answerHtml !== null) answer.innerHTML = answerHtml;
  else answer.textContent = content;
  article.appendChild(answer);
  if (citations.length) {
    const list = document.createElement('div');
    list.className = 'citations';
    citations.forEach((citation, index) => {
      const line = document.createElement('div');
      line.textContent = citation.source_type === 'attachment'
        ? `[附件] ${citation.name}`
        : `[${index + 1}] ${citation.document_name}，第 ${citation.page_start}-${citation.page_end} 頁`;
      list.appendChild(line);
      if (citation.crop_url) {
        const image = document.createElement('img');
        image.src = citation.crop_url;
        image.alt = `引用圖片 ${index + 1}`;
        list.appendChild(image);
      }
    });
    article.appendChild(list);
  }
  messages.insertBefore(article, beforeNode);
  article.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return article;
}

function createLoadingMessage(text) {
  const article = document.createElement('article');
  article.className = 'message assistant loading';
  article.setAttribute('role', 'status');
  const dots = document.createElement('span');
  dots.className = 'loading-dots';
  dots.setAttribute('aria-hidden', 'true');
  dots.innerHTML = '<span></span><span></span><span></span>';
  const label = document.createElement('span');
  label.className = 'loading-label';
  label.textContent = text;
  article.append(dots, label);
  messages.appendChild(article);
  article.scrollIntoView({ behavior: 'smooth', block: 'end' });
  activeLoadingMessage = article;
  return article;
}

function updateLoadingMessage(text) {
  const article = activeLoadingMessage?.isConnected
    ? activeLoadingMessage
    : createLoadingMessage(text);
  article.querySelector('.loading-label').textContent = text;
}

function failLoadingMessage(text) {
  const article = activeLoadingMessage?.isConnected
    ? activeLoadingMessage
    : createLoadingMessage(text);
  article.classList.remove('loading');
  article.classList.add('error');
  article.replaceChildren(document.createTextNode(text));
  activeLoadingMessage = null;
}

function renderResult(result, beforeNode = null) {
  result.items.forEach(item => appendMessage(
    'assistant', item.answer, item.citations || [],
    item.answer_html || null, item.question, beforeNode
  ));
}

function renderAttachments() {
  preview.replaceChildren();
  selectedFiles.forEach((file, index) => {
    const card = document.createElement('div');
    card.className = 'attachment-card';
    const image = document.createElement('img');
    const url = URL.createObjectURL(file);
    image.src = url;
    image.onload = () => URL.revokeObjectURL(url);
    image.alt = file.name;
    const label = document.createElement('span');
    label.textContent = file.name;
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.textContent = '移除';
    remove.addEventListener('click', () => {
      selectedFiles.splice(index, 1);
      renderAttachments();
      updateSendState();
    });
    card.append(image, label, remove);
    preview.appendChild(card);
  });
}

function addFiles(incoming) {
  const combined = [...selectedFiles, ...incoming];
  const total = combined.reduce((sum, file) => sum + file.size, 0);
  if (combined.length > maxFiles) {
    setStatus(`附件最多 ${maxFiles} 張。`);
    return false;
  }
  if (combined.some(file => !allowedImageTypes.has(file.type))) {
    setStatus('附件僅支援 JPEG、PNG 或 WebP。');
    return false;
  }
  if (combined.some(file => file.size > maxFileBytes)) {
    setStatus('單張圖片超過大小限制。');
    return false;
  }
  if (total > maxTotalBytes) {
    setStatus('附件合計大小超過限制。');
    return false;
  }
  selectedFiles = combined;
  setStatus('');
  renderAttachments();
  updateSendState();
  return true;
}

attachments.addEventListener('change', () => {
  addFiles(Array.from(attachments.files || []));
  attachments.value = '';
});

question.addEventListener('input', updateSendState);

question.addEventListener('paste', event => {
  if (isBusy) return;
  const clipboardImages = Array.from(event.clipboardData?.items || [])
    .filter(item => item.kind === 'file' && allowedImageTypes.has(item.type))
    .map(item => item.getAsFile())
    .filter(Boolean);
  if (!clipboardImages.length) return;
  event.preventDefault();
  const stamp = Date.now();
  const files = clipboardImages.map((file, index) => {
    const extension = file.type === 'image/png' ? 'png' : file.type === 'image/webp' ? 'webp' : 'jpg';
    return new File([file], `clipboard-${stamp}-${index + 1}.${extension}`, { type: file.type });
  });
  addFiles(files);
});

question.addEventListener('keydown', event => {
  if (event.key !== 'Enter' || event.shiftKey || event.isComposing || event.keyCode === 229) return;
  event.preventDefault();
  if (!send.disabled) form.requestSubmit();
});

async function loadHistory() {
  const response = await fetch('/api/history');
  if (!response.ok) return;
  const body = await response.json();
  body.messages.forEach(item => {
    if (item.role === 'assistant' && Array.isArray(item.items)) {
      item.items.forEach(entry => appendMessage('assistant', entry.answer,
        entry.citations || [], entry.answer_html || null, entry.question));
    } else appendMessage(item.role, item.content, item.citations || []);
  });
}

function watchJob(jobId) {
  if (activeStream) activeStream.close();
  setBusy(true);
  updateLoadingMessage('等待模型');
  const stream = new EventSource(`/api/chat/jobs/${encodeURIComponent(jobId)}/events`);
  activeStream = stream;
  ['queued', 'started', 'planning', 'retrieval', 'vision', 'answering'].forEach(name => {
    stream.addEventListener(name, event => {
      const body = JSON.parse(event.data);
      updateLoadingMessage(body.message || '處理中…');
    });
  });
  stream.addEventListener('completed', event => {
    const loadingMessage = activeLoadingMessage;
    renderResult(JSON.parse(event.data).result, loadingMessage);
    loadingMessage?.remove();
    activeLoadingMessage = null;
    stream.close();
    activeStream = null;
    setBusy(false);
    question.focus();
  });
  ['failed', 'cancelled'].forEach(name => stream.addEventListener(name, event => {
    const body = JSON.parse(event.data);
    stream.close();
    activeStream = null;
    failLoadingMessage(body.message || '工作未完成');
    setBusy(false);
  }));
  stream.onerror = () => {
    if (activeStream === stream) updateLoadingMessage('進度連線中斷，正在重新連線…');
  };
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  if (isBusy) return;
  const enteredText = question.value.trim();
  if (!enteredText && selectedFiles.length === 0) {
    updateSendState();
    return;
  }
  const text = enteredText || '請分析此圖片';
  const names = selectedFiles.map(file => file.name);
  appendMessage('user', names.length ? `${text}\n附件：${names.join('、')}` : text);
  const body = new FormData();
  body.append('question', text);
  selectedFiles.forEach(file => body.append('attachments', file));
  question.value = '';
  selectedFiles = [];
  renderAttachments();
  setStatus('');
  setBusy(true);
  createLoadingMessage('正在建立工作…');
  try {
    const response = await fetch('/api/chat/jobs', { method: 'POST', body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.message || '無法建立工作');
    watchJob(payload.job_id);
  } catch (error) {
    failLoadingMessage(error.message);
    setBusy(false);
  }
});

document.querySelector('#clear').addEventListener('click', async () => {
  if (activeStream) activeStream.close();
  activeStream = null;
  const response = await fetch('/api/history', { method: 'DELETE' });
  if (!response.ok) {
    statusText.textContent = '清除對話失敗，請稍後重試。';
    return;
  }
  messages.replaceChildren();
  activeLoadingMessage = null;
  setStatus('');
  setBusy(false);
});

document.querySelector('#end-chat').addEventListener('click', async () => {
  if (activeStream) activeStream.close();
  activeStream = null;
  const response = await fetch('/api/session/end', { method: 'POST' });
  if (!response.ok) {
    statusText.textContent = '結束對話失敗，請稍後重試。';
    return;
  }
  messages.replaceChildren();
  activeLoadingMessage = null;
  setBusy(false);
  setStatus('對話已結束，診斷紀錄已整理。');
});

async function restoreActiveJob() {
  const response = await fetch('/api/chat/jobs/active');
  if (!response.ok) return;
  const body = await response.json();
  if (body.jobs.length) watchJob(body.jobs[0].job_id);
}

loadHistory().then(restoreActiveJob);
updateSendState();
