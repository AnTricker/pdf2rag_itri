const messages = document.querySelector('#messages');
const form = document.querySelector('#chat-form');
const question = document.querySelector('#question');
const send = document.querySelector('#send');
const attachments = document.querySelector('#attachments');
const preview = document.querySelector('#attachment-preview');
const statusText = document.querySelector('#status-text');
const exportReport = document.querySelector('#export-report');
const suggestedQuestions = JSON.parse(
  document.querySelector('#suggested-questions-data').textContent
);
const ttsOptions = JSON.parse(
  document.querySelector('#tts-options-data').textContent
);
const ttsLanguages = new Map(
  ttsOptions.languages.map(language => [language.value, language])
);
let selectedFiles = [];
let activeStream = null;
let activeLoadingMessage = null;
let modelStartTimer = null;
let isBusy = false;
let completedQaCount = 0;
let chatToken = null;
const ttsCache = new Map();
const ttsTailByQa = new Map();
const chatTokenStorageKey = 'local_rag_chat_session';
const maxFiles = Number(form.dataset.maxFiles);
const maxFileBytes = Number(form.dataset.maxFileBytes);
const maxTotalBytes = Number(form.dataset.maxTotalBytes);
const allowedImageTypes = new Set(['image/jpeg', 'image/png', 'image/webp']);

async function startChatSession(previousChatToken = null) {
  const response = await fetch('/api/chat/session/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ previous_chat_token: previousChatToken }),
  });
  if (!response.ok) throw new Error('無法建立對話 session');
  const body = await response.json();
  chatToken = body.chat_token;
  sessionStorage.setItem(chatTokenStorageKey, chatToken);
  return body;
}

async function apiFetch(url, options = {}) {
  const headers = new Headers(options.headers || {});
  if (chatToken) headers.set('X-Chat-Session', chatToken);
  const response = await fetch(url, { ...options, headers });
  if (response.status === 409) {
    const body = await response.clone().json().catch(() => ({}));
    if (body.error_code === 'access_session_changed') {
      clearTtsCache();
      sessionStorage.removeItem(chatTokenStorageKey);
      chatToken = null;
      messages.replaceChildren();
      completedQaCount = 0;
      await startChatSession();
      renderWelcome();
      setStatus('網路環境已改變，已建立新對話。');
    }
  }
  return response;
}

function updateSendState() {
  send.disabled = isBusy || (!question.value.trim() && selectedFiles.length === 0);
}

function updateReportState() {
  exportReport.disabled = isBusy || completedQaCount === 0;
}

function setStatus(text = '') {
  statusText.textContent = text;
}

function setBusy(busy) {
  isBusy = busy;
  form.classList.toggle('busy', busy);
  question.disabled = busy;
  attachments.disabled = busy;
  document.querySelectorAll('.tts-actions select, .tts-actions button')
    .forEach(control => { control.disabled = busy; });
  messages.setAttribute('aria-busy', String(busy));
  updateSendState();
  updateReportState();
}

function clearTtsCache() {
  ttsCache.forEach(entry => URL.revokeObjectURL(entry.url));
  ttsCache.clear();
  ttsTailByQa.clear();
}

function insertTtsBlock(sourceArticle, qaId, block) {
  const anchor = ttsTailByQa.get(qaId) || sourceArticle;
  anchor.insertAdjacentElement('afterend', block);
  ttsTailByQa.set(qaId, block);
  block.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function createTtsLoading(sourceArticle, qaId) {
  const article = document.createElement('article');
  article.className = 'message assistant loading tts-response';
  article.setAttribute('role', 'status');
  const dots = document.createElement('span');
  dots.className = 'loading-dots';
  dots.setAttribute('aria-hidden', 'true');
  dots.innerHTML = '<span></span><span></span><span></span>';
  const label = document.createElement('span');
  label.className = 'loading-label';
  label.textContent = '正在產生語音…';
  article.append(dots, label);
  insertTtsBlock(sourceArticle, qaId, article);
  return article;
}

function openCitationImage(url, alt) {
  let dialog = document.querySelector('.citation-lightbox');
  if (!dialog) {
    dialog = document.createElement('dialog');
    dialog.className = 'citation-lightbox';
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'citation-lightbox-close';
    close.textContent = '關閉';
    close.addEventListener('click', () => dialog.close());
    const image = document.createElement('img');
    dialog.append(close, image);
    dialog.addEventListener('click', event => {
      if (event.target === dialog) dialog.close();
    });
    document.body.appendChild(dialog);
  }
  const image = dialog.querySelector('img');
  image.src = url;
  image.alt = alt;
  dialog.showModal();
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
  answer.className = 'answer-content';
  if (role === 'assistant' && answerHtml !== null) answer.innerHTML = answerHtml;
  else answer.textContent = content;
  article.appendChild(answer);
  if (citations.length) {
    const list = document.createElement('div');
    list.className = 'citations';
    const title = document.createElement('strong');
    title.className = 'citations-title';
    title.textContent = '參考資料';
    list.appendChild(title);
    const grid = document.createElement('div');
    grid.className = 'citation-grid';
    citations.forEach((citation, index) => {
      const card = document.createElement('article');
      card.className = 'citation-card';
      const heading = document.createElement('strong');
      if (citation.source_type === 'attachment') {
        heading.textContent = `附件｜${citation.name}`;
        card.appendChild(heading);
        grid.appendChild(card);
        return;
      }
      const pages = citation.page_start === citation.page_end
        ? `第 ${citation.page_start} 頁`
        : `第 ${citation.page_start}–${citation.page_end} 頁`;
      heading.textContent = `[${index + 1}] ${citation.document_name}`;
      const location = document.createElement('div');
      location.className = 'citation-location';
      location.textContent = pages;
      card.append(heading, location);
      if ((citation.section_path || []).length) {
        const section = document.createElement('div');
        section.className = 'citation-section';
        section.textContent = citation.section_path.join(' / ');
        card.appendChild(section);
      }
      if ((citation.content_types || []).length) {
        const tags = document.createElement('div');
        tags.className = 'citation-tags';
        citation.content_types.forEach(value => {
          const tag = document.createElement('span');
          tag.textContent = value;
          tags.appendChild(tag);
        });
        card.appendChild(tags);
      }
      const identifiers = [
        citation.source_kb_id ? `KB ${citation.source_kb_id}` : '',
        citation.source_record_id ? `Record ${citation.source_record_id}` : '',
        citation.source_image_id ? `Image ${citation.source_image_id}` : '',
      ].filter(Boolean);
      if (identifiers.length) {
        const source = document.createElement('code');
        source.className = 'citation-source-id';
        source.textContent = identifiers.join(' · ');
        card.appendChild(source);
      }
      if (citation.crop_url) {
        const preview = document.createElement('button');
        preview.type = 'button';
        preview.className = 'citation-preview';
        preview.setAttribute('aria-label', `放大引用圖片 ${index + 1}`);
        const image = document.createElement('img');
        image.src = citation.crop_url;
        image.alt = `引用圖片 ${index + 1}`;
        preview.appendChild(image);
        preview.addEventListener('click', () => {
          openCitationImage(citation.crop_url, image.alt);
        });
        card.appendChild(preview);
      }
      grid.appendChild(card);
    });
    list.appendChild(grid);
    article.appendChild(list);
  }
  messages.insertBefore(article, beforeNode);
  article.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return article;
}

function answerPresentation(item) {
  if (item.route === 'security_request') {
    return { className: 'route-security', label: '安全限制' };
  }
  if (item.route === 'out_of_scope') {
    return { className: 'route-out-of-scope', label: '超出文件範圍' };
  }
  if (item.insufficient_context) {
    return { className: 'route-insufficient', label: '文件資訊不足' };
  }
  return { className: '', label: '' };
}

async function postQaAction(qaId, payload) {
  const response = await apiFetch(`/api/chat/qa/${encodeURIComponent(qaId)}/actions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.message || '無法記錄操作');
  return body;
}

function updateFeedbackButtons(container, feedback) {
  container.querySelector('[data-feedback="1"]').classList.toggle('active', feedback === 1);
  container.querySelector('[data-feedback="-1"]').classList.toggle('active', feedback === -1);
  container.querySelector('[data-feedback="1"]').setAttribute('aria-pressed', String(feedback === 1));
  container.querySelector('[data-feedback="-1"]').setAttribute('aria-pressed', String(feedback === -1));
}

function appendTtsActions(article, item, actions) {
  const controls = document.createElement('div');
  controls.className = 'tts-actions';

  const languageLabel = document.createElement('label');
  languageLabel.textContent = '語言';
  const languageSelect = document.createElement('select');
  languageSelect.setAttribute('aria-label', '語音語言');
  ttsOptions.languages.forEach(language => {
    const option = document.createElement('option');
    option.value = language.value;
    option.textContent = language.label;
    option.selected = language.value === ttsOptions.default_language;
    languageSelect.appendChild(option);
  });
  languageLabel.appendChild(languageSelect);

  const voiceLabel = document.createElement('label');
  voiceLabel.textContent = '聲音';
  const voiceSelect = document.createElement('select');
  voiceSelect.setAttribute('aria-label', '語音聲音');
  voiceLabel.appendChild(voiceSelect);

  const updateVoices = () => {
    const language = ttsLanguages.get(languageSelect.value);
    voiceSelect.replaceChildren();
    language.voices.forEach(voice => {
      const option = document.createElement('option');
      option.value = voice.value;
      option.textContent = voice.label;
      option.selected = voice.value === language.default_voice;
      voiceSelect.appendChild(option);
    });
  };
  languageSelect.addEventListener('change', updateVoices);
  updateVoices();

  const playButton = document.createElement('button');
  playButton.type = 'button';
  playButton.textContent = '生成語音';
  playButton.addEventListener('click', async () => {
    if (isBusy) return;
    const language = ttsLanguages.get(languageSelect.value);
    const voice = language.voices.length ? voiceSelect.value : language.default_voice;
    const voiceText = voiceSelect.selectedOptions[0]?.textContent || voice;
    const cacheKey = `${item.qa_id}:${language.value}:${voice}`;
    const existing = ttsCache.get(cacheKey);
    if (existing) {
      playButton.textContent = '已生成';
      existing.article.scrollIntoView({ behavior: 'smooth', block: 'end' });
      setTimeout(() => { playButton.textContent = '生成語音'; }, 1200);
      return;
    }

    const loading = createTtsLoading(article, item.qa_id);
    setStatus('');
    setBusy(true);
    try {
      const response = await apiFetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: article.querySelector('.answer-content').innerText.trim(),
          lang_type: language.value,
          voice: language.voices.length ? voice : null,
        }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.message || '語音合成失敗，請稍後重試');
      }
      const audioUrl = URL.createObjectURL(await response.blob());
      const label = document.createElement('strong');
      label.className = 'tts-response-label';
      label.textContent = `${language.label} | ${voiceText}`;
      const audio = document.createElement('audio');
      audio.controls = true;
      audio.preload = 'metadata';
      audio.src = audioUrl;
      loading.className = 'message assistant tts-response';
      loading.removeAttribute('role');
      loading.replaceChildren(label, audio);
      ttsCache.set(cacheKey, { url: audioUrl, article: loading });
    } catch (error) {
      loading.classList.remove('loading');
      loading.classList.add('error');
      loading.replaceChildren(document.createTextNode(error.message || '語音合成失敗'));
    } finally {
      setBusy(false);
      question.focus();
    }
  });

  controls.append(languageLabel, voiceLabel, playButton);
  actions.appendChild(controls);
  controls.querySelectorAll('select, button').forEach(control => {
    control.disabled = isBusy;
  });
}

function appendAnswer(item, beforeNode = null) {
  const article = appendMessage(
    'assistant', item.answer, item.citations || [],
    item.answer_html || null, item.question, beforeNode
  );
  const presentation = answerPresentation(item);
  if (presentation.className) article.classList.add(presentation.className);
  if (presentation.label) {
    const label = document.createElement('span');
    label.className = 'answer-label';
    label.textContent = presentation.label;
    article.insertBefore(label, article.querySelector('.answer-content'));
  }
  if (!item.qa_id) return article;

  const actions = document.createElement('div');
  actions.className = 'answer-actions';
  const copyButton = document.createElement('button');
  copyButton.type = 'button';
  copyButton.textContent = '複製';
  copyButton.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(item.answer || '');
      await postQaAction(item.qa_id, { action: 'copy' });
      copyButton.textContent = '已複製';
      setTimeout(() => { copyButton.textContent = '複製'; }, 1200);
    } catch (error) {
      setStatus(error.message || '複製失敗');
    }
  });
  const likeButton = document.createElement('button');
  likeButton.type = 'button';
  likeButton.dataset.feedback = '1';
  likeButton.textContent = '👍';
  const dislikeButton = document.createElement('button');
  dislikeButton.type = 'button';
  dislikeButton.dataset.feedback = '-1';
  dislikeButton.textContent = '👎';
  [likeButton, dislikeButton].forEach(button => {
    button.addEventListener('click', async () => {
      const selected = Number(button.dataset.feedback);
      const next = item.feedback === selected ? 0 : selected;
      try {
        const result = await postQaAction(item.qa_id, {
          action: 'feedback', feedback: next,
        });
        item.feedback = result.feedback;
        updateFeedbackButtons(actions, item.feedback);
      } catch (error) {
        setStatus(error.message || '無法更新 Feedback');
      }
    });
  });
  actions.append(copyButton, likeButton, dislikeButton);
  appendTtsActions(article, item, actions);
  article.appendChild(actions);
  updateFeedbackButtons(actions, item.feedback || 0);
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

function stopModelStartTimer() {
  if (modelStartTimer !== null) clearInterval(modelStartTimer);
  modelStartTimer = null;
}

function startModelStartTimer(message) {
  stopModelStartTimer();
  const startedAt = Date.now();
  const update = () => {
    const seconds = Math.floor((Date.now() - startedAt) / 1000);
    updateLoadingMessage(`${message}（已等待 ${seconds} 秒）`);
  };
  update();
  modelStartTimer = setInterval(update, 1000);
}

function renderResult(result, beforeNode = null) {
  result.items.forEach(item => appendAnswer(item, beforeNode));
  completedQaCount += result.items.length;
  updateReportState();
}

function removeWelcome() {
  document.querySelector('.welcome-panel')?.remove();
}

function renderWelcome() {
  if (document.querySelector('.welcome-panel') || completedQaCount > 0) return;
  const panel = document.createElement('section');
  panel.className = 'welcome-panel';
  const title = document.createElement('h2');
  title.textContent = '此為 CMP 化學機械研磨機操作規範，歡迎使用';
  const description = document.createElement('p');
  description.textContent = '可直接輸入問題，或選擇下列提示問題開始。';
  const choices = document.createElement('div');
  choices.className = 'suggested-questions';
  suggestedQuestions.forEach(text => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'suggested-question';
    button.textContent = text;
    button.addEventListener('click', async () => {
      question.value = text;
      updateSendState();
      question.focus();
      try {
        const response = await apiFetch('/api/chat/suggestions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: text }),
        });
        if (!response.ok) throw new Error('無法記錄提示問題');
      } catch (error) {
        setStatus(error.message);
      }
    });
    choices.appendChild(button);
  });
  panel.append(title, description, choices);
  messages.appendChild(panel);
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
  const response = await apiFetch('/api/history');
  if (!response.ok) return 0;
  const body = await response.json();
  body.messages.forEach(item => {
    if (item.role === 'assistant' && Array.isArray(item.items)) {
      item.items.forEach(entry => appendAnswer(entry));
    } else appendMessage(item.role, item.content, item.citations || []);
  });
  completedQaCount = body.messages
    .filter(item => item.role === 'assistant' && Array.isArray(item.items))
    .reduce((sum, item) => sum + item.items.length, 0);
  updateReportState();
  return completedQaCount;
}

function watchJob(jobId, streamToken) {
  if (activeStream) activeStream.close();
  stopModelStartTimer();
  setBusy(true);
  updateLoadingMessage('等待模型');
  const stream = new EventSource(
    `/api/chat/jobs/${encodeURIComponent(jobId)}/events?token=${encodeURIComponent(streamToken)}`
  );
  activeStream = stream;
  stream.addEventListener('model_starting', event => {
    const body = JSON.parse(event.data);
    startModelStartTimer(body.message || '正在啟動模型，首次回應需要較長時間');
  });
  stream.addEventListener('model_ready', event => {
    const body = JSON.parse(event.data);
    stopModelStartTimer();
    updateLoadingMessage(body.message || '模型已啟動，正在分析問題');
  });
  ['queued', 'started', 'planning', 'retrieval', 'vision', 'answering'].forEach(name => {
    stream.addEventListener(name, event => {
      const body = JSON.parse(event.data);
      if (name === 'planning') stopModelStartTimer();
      updateLoadingMessage(body.message || '處理中…');
    });
  });
  stream.addEventListener('completed', event => {
    stopModelStartTimer();
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
    stopModelStartTimer();
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
  removeWelcome();
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
    const response = await apiFetch('/api/chat/jobs', { method: 'POST', body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.message || '無法建立工作');
    watchJob(payload.job_id, payload.stream_token);
  } catch (error) {
    failLoadingMessage(error.message);
    setBusy(false);
  }
});

document.querySelector('#clear').addEventListener('click', async () => {
  if (activeStream) activeStream.close();
  activeStream = null;
  stopModelStartTimer();
  const response = await apiFetch('/api/history', { method: 'DELETE' });
  if (!response.ok) {
    statusText.textContent = '清除對話失敗，請稍後重試。';
    return;
  }
  const payload = await response.json();
  chatToken = payload.chat_token;
  sessionStorage.setItem(chatTokenStorageKey, chatToken);
  clearTtsCache();
  messages.replaceChildren();
  activeLoadingMessage = null;
  completedQaCount = 0;
  setStatus('');
  setBusy(false);
  renderWelcome();
});

exportReport.addEventListener('click', async () => {
  if (exportReport.disabled) return;
  exportReport.disabled = true;
  setStatus('');
  try {
    const response = await apiFetch('/api/history/report', { method: 'POST' });
    if (!response.ok) {
      const body = await response.json();
      throw new Error(body.message || '無法輸出對話報告');
    }
    const blob = await response.blob();
    const disposition = response.headers.get('Content-Disposition') || '';
    const match = disposition.match(/filename="([^"]+)"/);
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = match?.[1] || 'cmp_chat_report.md';
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(link.href);
  } catch (error) {
    setStatus(error.message);
  } finally {
    updateReportState();
  }
});

document.querySelector('#end-chat').addEventListener('click', async () => {
  if (activeStream) activeStream.close();
  activeStream = null;
  stopModelStartTimer();
  setBusy(true);
  const response = await apiFetch('/api/session/end', { method: 'POST' });
  if (!response.ok) {
    statusText.textContent = '結束對話失敗，請稍後重試。';
    setBusy(false);
    return;
  }
  clearTtsCache();
  sessionStorage.removeItem(chatTokenStorageKey);
  chatToken = null;
  window.close();
  setTimeout(() => {
    document.querySelector('main').innerHTML = [
      '<section class="ended-panel">',
      '<h1>對話已結束</h1>',
      '<p>瀏覽器未允許自動關閉此頁面，現在可以安全地關閉分頁。</p>',
      '</section>',
    ].join('');
  }, 150);
});

async function bootstrap() {
  const previousChatToken = sessionStorage.getItem(chatTokenStorageKey);
  await startChatSession(previousChatToken);
  const count = await loadHistory();
  if (!count) renderWelcome();
}

bootstrap().catch(error => setStatus(error.message));
window.addEventListener('beforeunload', clearTtsCache);
updateSendState();
updateReportState();
