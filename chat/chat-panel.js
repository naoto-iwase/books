// AI Chat Panel functionality
(function() {
  let models = [];
  let messages = [];
  let currentContent = '';
  let isOpen = false;
  let currentLanguage = 'ja';
  let sessions = [];
  let currentSessionId = null;

  // i18n definitions
  const i18n = {
    ja: {
      title: "💬 AIチャット",
      closeBtn: "✕ 閉じる",
      openBtn: "💬 AIチャット",
      inputPlaceholder: "質問を入力... (Shift+Enter で改行)",
      setupMessage: "でAPIキーを取得して、上記に設定してください",
      infoTitle: "💡 このチャットについて",
      infoBullets: [
        "現在表示中のページの内容に基づいて質問に答えます",
        "APIキーはブラウザ（localStorage）にのみ保存され、外部に送信されません",
        "OpenRouterで70以上のモデルから選択可能",
        "使用料金は選択したモデルに依存します（🆓は無料）"
      ],
      resizeHint: "⬅️ をドラッグしてパネルの幅を調整できます",
      ready: "準備完了です。このページの内容について質問してください。",
      error: "エラー:",
      apiKeyRequired: "API Keyを入力してください",
      clearChat: "クリア",
      thinking: "考え中",
      newChat: "新規",
      newSession: "新しいチャット",
      sessionHistory: "履歴",
      saveSettings: "設定を保存",
      deleteApiKey: "API Keyを削除",
      validating: "確認中...",
      invalidApiKey: "API Keyが無効です。正しいキーを入力してください。",
      siteStructure: "**サイト全体の構成 (Recent 5 books per language):**",
      contentLoadError: "このページの内容を読み込めませんでした。一般的な質問には答えられます。",
      deleteConfirm: "API Keyを削除しますか？チャット履歴も全て削除されます。"
    },
    en: {
      title: "💬 AI Chat",
      closeBtn: "✕ Close",
      openBtn: "💬 AI Chat",
      inputPlaceholder: "Ask a question... (Shift+Enter for new line)",
      setupMessage: "to get your API key and save it above",
      infoTitle: "💡 About This Chat",
      infoBullets: [
        "Answers questions based on current page content",
        "API key stored locally in browser only, never sent externally",
        "Choose from 70+ models on OpenRouter",
        "Usage costs depend on selected model (🆓 = free)"
      ],
      resizeHint: "⬅️ Drag to resize panel width",
      ready: "Ready! Ask questions about this page.",
      error: "Error:",
      apiKeyRequired: "API Key is required",
      clearChat: "Clear",
      thinking: "Thinking",
      newChat: "New",
      newSession: "New Chat",
      sessionHistory: "History",
      saveSettings: "Save Settings",
      deleteApiKey: "Delete API Key",
      validating: "Validating...",
      invalidApiKey: "Invalid API Key. Please enter a valid key.",
      siteStructure: "**Site Structure (Recent 5 books per language):**",
      contentLoadError: "Failed to load page content. General questions can still be answered.",
      deleteConfirm: "Delete API Key? All chat history will also be deleted."
    }
  };

  // Detect language from URL path
  function detectLanguage() {
    const path = window.location.pathname;
    if (path.includes('/ja/')) return 'ja';
    if (path.includes('/en/')) return 'en';
    // Default to English for top page
    return 'en';
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  function init() {
    // Always prioritize auto-detection from URL
    // Manual language selection is temporary and doesn't persist across page navigation
    currentLanguage = detectLanguage();

    createChatPanel();
    loadModels();
    loadPageContent();

    const savedKey = localStorage.getItem('openrouter-api-key');
    if (savedKey) {
      document.getElementById('chat-api-key').value = savedKey;
      document.getElementById('chat-settings').classList.add('hidden');
      document.getElementById('chat-setup-wrapper').style.display = 'none';
      document.getElementById('chat-user-input').disabled = false;
      document.getElementById('chat-send-btn').disabled = false;
      document.getElementById('chat-settings-delete').style.display = 'block';

      const savedModel = localStorage.getItem('openrouter-model');
      if (savedModel) {
        setTimeout(() => {
          const select = document.getElementById('chat-model-select');
          if (select) {
            select.value = savedModel;
            updateCurrentModelDisplay();
          }
        }, 500);
      }

    }

    // Set initial language
    updateLanguage(currentLanguage);

    // Load sessions
    loadSessions();

    // Show ready message if no history and API key is set
    if (savedKey && messages.length === 0) {
      addChatMessage('assistant', i18n[currentLanguage].ready);
    }
  }

  // Session management functions
  function loadSessions() {
    const saved = localStorage.getItem('chat-sessions');
    if (saved) {
      try {
        sessions = JSON.parse(saved);
      } catch (e) {
        console.error('Failed to load sessions:', e);
        sessions = [];
      }
    }

    const savedSessionId = localStorage.getItem('current-session-id');
    if (savedSessionId && sessions.find(s => s.id === savedSessionId)) {
      currentSessionId = savedSessionId;
    } else if (sessions.length > 0) {
      currentSessionId = sessions[0].id;
    } else {
      // Create initial session
      const newSession = createSession();
      sessions.push(newSession);
      currentSessionId = newSession.id;
      saveSessions();
      return;
    }

    loadCurrentSession();
  }

  function saveSessions() {
    localStorage.setItem('chat-sessions', JSON.stringify(sessions));
    if (currentSessionId) {
      localStorage.setItem('current-session-id', currentSessionId);
    }
  }

  function getPageDisplay() {
    // Extract path including language: /ja/molmo2/dense-video-captioning.html → ja/molmo2/dense-video-captioning
    const path = window.location.pathname;
    const match = path.match(/\/((?:ja|en)\/.+?)(?:\.html)?$/);

    if (match) {
      return match[1].replace(/\/index$/, ''); // Remove trailing /index if present
    }

    // Fallback to page title
    return document.title.replace(/ [–-] Naoto's Books/g, '').replace(/^Books [–-] /, '');
  }

  function createSession() {
    return {
      id: 'session-' + Date.now() + '-' + Math.random().toString(36).substring(2, 11),
      url: window.location.pathname,
      title: i18n[currentLanguage].newSession,
      page: getPageDisplay(),
      updated: Date.now(),
      messages: []
    };
  }

  function loadCurrentSession() {
    const session = sessions.find(s => s.id === currentSessionId);
    if (!session) return;

    messages = session.messages || [];

    // Clear and reload messages
    const messagesDiv = document.getElementById('chat-messages');
    if (messagesDiv) {
      messagesDiv.innerHTML = '';
      messages.forEach(msg => {
        addChatMessage(msg.role, msg.content);
      });
    }

    updateSessionDisplay();
  }

  function saveCurrentSession() {
    const session = sessions.find(s => s.id === currentSessionId);
    if (!session) return;

    session.messages = messages;
    session.url = window.location.pathname;
    session.page = getPageDisplay();
    session.updated = Date.now();

    // Update title from first user message if still using default title
    if (session.title === i18n.ja.newSession || session.title === i18n.en.newSession) {
      const firstUserMsg = messages.find(m => m.role === 'user');
      if (firstUserMsg) {
        session.title = firstUserMsg.content.substring(0, 30).trim();
      }
    }

    saveSessions();
    updateSessionDisplay();
  }

  function updateSessionDisplay() {
    const session = sessions.find(s => s.id === currentSessionId);
    if (!session) return;

    const nameEl = document.getElementById('chat-current-session-name');
    if (nameEl) {
      const title = session.title || i18n[currentLanguage].newSession;
      nameEl.textContent = title.substring(0, 20);
    }

    renderSessionList();
  }

  function renderSessionList() {
    const listEl = document.getElementById('chat-session-list');
    if (!listEl) return;

    listEl.innerHTML = '';

    sessions.sort((a, b) => b.updated - a.updated).forEach(session => {
      const item = document.createElement('div');
      item.className = 'chat-session-item' + (session.id === currentSessionId ? ' active' : '');

      const date = new Date(session.updated);
      const dateStr = `${date.getMonth() + 1}/${date.getDate()} ${date.getHours()}:${String(date.getMinutes()).padStart(2, '0')}`;
      const title = (session.title || i18n[currentLanguage].newSession).substring(0, 40);
      const page = session.page ? session.page.substring(0, 60) : '';

      item.innerHTML = `
        <div class="chat-session-info" onclick="window.switchSession('${session.id}')">
          <div class="chat-session-date">${dateStr}</div>
          <div class="chat-session-title">${title}</div>
          ${page ? `<div class="chat-session-page">@ ${page}</div>` : ''}
        </div>
        <button class="chat-session-delete" onclick="window.deleteSession(event, '${session.id}')" title="Delete">×</button>
      `;

      listEl.appendChild(item);
    });
  }

  function createChatPanel() {
    const panel = document.createElement('div');
    panel.id = 'chat-panel';
    panel.innerHTML = `
      <div class="chat-resize-handle" id="chat-resize-handle"></div>
      <div class="chat-header">
        <div class="chat-header-top">
          <span class="chat-icon">💬</span>
          <span id="chat-current-model" class="chat-current-model"></span>
          <div class="chat-session-dropdown-wrapper">
            <button class="chat-session-dropdown-btn" id="chat-session-dropdown-btn" onclick="window.toggleSessionDropdown()">
              <span id="chat-current-session-name">New Chat</span> ▼
            </button>
            <div class="chat-session-dropdown hidden" id="chat-session-dropdown">
              <div class="chat-session-list" id="chat-session-list"></div>
            </div>
          </div>
          <div class="chat-header-buttons">
            <button class="chat-new-btn" id="chat-new-btn" onclick="window.createNewSession()" title="New chat">📝</button>
            <button class="chat-settings-toggle" onclick="window.toggleChatSettings()" title="Settings">⚙️</button>
          </div>
        </div>
        <div class="chat-settings" id="chat-settings">
          <label>Model:</label>
          <select id="chat-model-select" disabled>
            <option>Loading models...</option>
          </select>
          <label>API Key:</label>
          <input type="password" id="chat-api-key" placeholder="sk-or-v1-...">
          <div class="chat-settings-buttons">
            <button class="chat-settings-save" id="chat-settings-save-btn" onclick="window.saveChatSettings()">Save Settings</button>
            <button class="chat-settings-delete" id="chat-settings-delete" onclick="window.deleteChatSettings()" style="display: none;">Delete API Key</button>
          </div>
        </div>
      </div>
      <div class="chat-setup-wrapper" id="chat-setup-wrapper">
        <p class="chat-setup-text">
          <a href="https://openrouter.ai/keys" target="_blank">OpenRouter</a> <span id="chat-setup-msg"></span>
        </p>
        <div class="chat-info">
          <strong id="chat-info-title"></strong><br>
          <span id="chat-info-bullets"></span>
        </div>
        <div class="chat-resize-hint" id="chat-resize-hint"></div>
      </div>
      <div class="chat-messages" id="chat-messages">
      </div>
      <div class="chat-input-area">
        <div class="chat-input-group">
          <textarea id="chat-user-input" placeholder="" disabled rows="1"></textarea>
          <button id="chat-send-btn" onclick="window.sendChatMessage()"></button>
        </div>
      </div>
    `;
    document.body.appendChild(panel);

    // Create toggle button
    const toggleBtn = document.createElement('button');
    toggleBtn.className = 'chat-toggle-btn';
    toggleBtn.textContent = '💬 AI Chat';
    toggleBtn.onclick = toggleChat;
    document.body.appendChild(toggleBtn);

    // Enter key to send
    const userInput = document.getElementById('chat-user-input');
    userInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        window.sendChatMessage();
      }
    });

    // Auto-resize textarea based on content
    userInput.addEventListener('input', function() {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 150) + 'px';
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
      const dropdown = document.getElementById('chat-session-dropdown');
      const dropdownBtn = document.getElementById('chat-session-dropdown-btn');
      if (dropdown && dropdownBtn && !dropdown.contains(e.target) && !dropdownBtn.contains(e.target)) {
        dropdown.classList.add('hidden');
      }
    });

    // Setup resize handle
    setupResizeHandle();

    // Restore saved width
    const savedWidth = localStorage.getItem('chat-panel-width');
    if (savedWidth) {
      panel.style.width = savedWidth + 'px';
    }
  }

  function setupResizeHandle() {
    const handle = document.getElementById('chat-resize-handle');
    const panel = document.getElementById('chat-panel');
    let isResizing = false;
    let startX = 0;
    let startWidth = 0;

    handle.addEventListener('mousedown', (e) => {
      isResizing = true;
      startX = e.clientX;
      startWidth = panel.offsetWidth;
      panel.classList.add('resizing');
      e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
      if (!isResizing) return;

      const deltaX = startX - e.clientX; // Inverted because panel grows left
      const newWidth = Math.min(Math.max(300, startWidth + deltaX), window.innerWidth * 0.9);
      panel.style.width = newWidth + 'px';
    });

    document.addEventListener('mouseup', () => {
      if (isResizing) {
        isResizing = false;
        panel.classList.remove('resizing');
        // Save width to localStorage
        localStorage.setItem('chat-panel-width', panel.offsetWidth);
      }
    });
  }

  function toggleChat() {
    isOpen = !isOpen;
    const panel = document.getElementById('chat-panel');
    const btn = document.querySelector('.chat-toggle-btn');
    const t = i18n[currentLanguage];

    if (isOpen) {
      panel.classList.add('open');
      btn.classList.add('open');
      btn.textContent = t.closeBtn;
    } else {
      panel.classList.remove('open');
      btn.classList.remove('open');
      btn.textContent = t.openBtn;
    }
  }

  function updateCurrentModelDisplay() {
    const select = document.getElementById('chat-model-select');
    const display = document.getElementById('chat-current-model');
    if (!select || !display) return;

    const selectedOption = select.options[select.selectedIndex];
    if (selectedOption) {
      display.textContent = selectedOption.textContent;
    }
  }

  async function loadModels() {
    try {
      const response = await fetch('https://openrouter.ai/api/v1/models');
      const data = await response.json();
      models = data.data;

      const select = document.getElementById('chat-model-select');
      select.innerHTML = '';

      const priorityProviders = ['OpenAI', 'Anthropic', 'Google', 'Qwen', 'DeepSeek', 'NVIDIA', 'Meta', 'Mistral'];

      models.sort((a, b) => {
        // Free models first
        const aFree = a.id.includes(':free') ? 0 : 1;
        const bFree = b.id.includes(':free') ? 0 : 1;
        if (aFree !== bFree) return aFree - bFree;

        // Priority providers
        const getPriority = (name) => {
          const provider = name.match(/^([^:]+):/)?.[1].trim() || '';
          const index = priorityProviders.indexOf(provider);
          return index === -1 ? 999 : index;
        };
        const diff = getPriority(a.name) - getPriority(b.name);
        if (diff !== 0) return diff;

        // Alphabetical
        return a.name.localeCompare(b.name);
      });

      models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        const isFree = model.id.includes(':free');
        const modelName = model.name.replace(/\s*\(free\)\s*/gi, '').trim();
        const pricing = !isFree && model.pricing?.prompt
          ? ` ($${(parseFloat(model.pricing.prompt) * 1000000).toFixed(2)}/1M)`
          : '';
        const free = isFree ? ' 🆓' : '';
        option.textContent = `${modelName}${free}${pricing}`;
        select.appendChild(option);
      });

      select.disabled = false;

      // Update current model display when selection changes
      select.addEventListener('change', updateCurrentModelDisplay);
      updateCurrentModelDisplay();
    } catch (error) {
      console.error('Failed to load models:', error);
    }
  }

  async function loadPageContent() {
    try {
      const path = window.location.pathname;
      const basePath = path.startsWith('/books/') ? '/books' : '';

      // Convert URL to .qmd path: /books/ja/olmo-3/index.html → ja/olmo-3/index.qmd
      let qmdPath = path
        .replace(basePath, '')
        .replace(/^\//, '')
        .replace('.html', '.qmd');

      if (!qmdPath.endsWith('.qmd')) {
        qmdPath += (qmdPath.endsWith('/') ? '' : '/') + 'index.qmd';
      }

      // Handle root: empty path → index.qmd
      if (qmdPath === 'index.qmd' || qmdPath === '/index.qmd') {
        qmdPath = 'index.qmd';
      }

      const response = await fetch(`${basePath}/${qmdPath}`);
      if (!response.ok) throw new Error(`Failed to fetch ${qmdPath}`);

      currentContent = await response.text();

      // Include search.json for root page to provide site-wide navigation context
      if (qmdPath === 'index.qmd') {
        try {
          const searchRes = await fetch(`${basePath}/search.json`);
          if (searchRes.ok) {
            const searchData = await searchRes.json();

            // Filter for index pages only (ja/*/index.html, en/*/index.html)
            const jaIndexPages = searchData
              .filter(item => item.href && /^ja\/[^\/]+\/index\.html$/.test(item.href))
              .slice(0, 5)
              .map(item => ({ title: item.title, href: item.href, categories: item.categories }));

            const enIndexPages = searchData
              .filter(item => item.href && /^en\/[^\/]+\/index\.html$/.test(item.href))
              .slice(0, 5)
              .map(item => ({ title: item.title, href: item.href, categories: item.categories }));

            const summary = {
              ja: jaIndexPages,
              en: enIndexPages
            };

            currentContent += '\n\n' + i18n[currentLanguage].siteStructure + '\n' + JSON.stringify(summary, null, 2);
          }
        } catch (e) {
          // search.json is optional, continue without it
        }
      }
    } catch (error) {
      console.error('Failed to load content:', error);
      currentContent = i18n[currentLanguage].contentLoadError;
    }
  }

  function updateLanguage(lang) {
    currentLanguage = lang;
    const t = i18n[lang];
    const $ = (id, prop, val) => {
      const el = document.getElementById(id);
      if (el) el[prop] = val;
    };

    $('chat-user-input', 'placeholder', t.inputPlaceholder);
    $('chat-send-btn', 'textContent', 'Enter');
    $('chat-setup-msg', 'textContent', t.setupMessage);
    $('chat-info-title', 'textContent', t.infoTitle);
    $('chat-info-bullets', 'innerHTML', t.infoBullets.map(b => '• ' + b).join('<br>'));
    $('chat-resize-hint', 'textContent', t.resizeHint);
    $('chat-settings-save-btn', 'textContent', t.saveSettings);
    $('chat-settings-delete', 'textContent', t.deleteApiKey);

    const btn = document.querySelector('.chat-toggle-btn');
    if (btn) btn.textContent = isOpen ? t.closeBtn : t.openBtn;
  }

  window.saveChatSettings = async function() {
    const apiKey = document.getElementById('chat-api-key').value.trim();
    const t = i18n[currentLanguage];
    const saveBtn = document.querySelector('.chat-settings-save');

    if (!apiKey) {
      alert(t.apiKeyRequired);
      return;
    }

    // Disable button and show validating state
    const originalText = saveBtn.textContent;
    saveBtn.disabled = true;
    saveBtn.textContent = t.validating;

    try {
      // Test API key with a simple request
      const response = await fetch('https://openrouter.ai/api/v1/auth/key', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${apiKey}`
        }
      });

      if (!response.ok) {
        throw new Error(t.invalidApiKey);
      }

      // Key is valid, save it
      localStorage.setItem('openrouter-api-key', apiKey);
      localStorage.setItem('openrouter-model', document.getElementById('chat-model-select').value);
      updateCurrentModelDisplay();

      // Reset button state before hiding
      saveBtn.disabled = false;
      saveBtn.textContent = originalText;

      // Show delete button
      document.getElementById('chat-settings-delete').style.display = 'block';

      enableChat();
    } catch (error) {
      alert(t.error + ' ' + error.message);
      saveBtn.disabled = false;
      saveBtn.textContent = originalText;
    }
  };

  function enableChat() {
    document.getElementById('chat-setup-wrapper').style.display = 'none';
    document.getElementById('chat-settings').classList.add('hidden');
    document.getElementById('chat-user-input').disabled = false;
    document.getElementById('chat-send-btn').disabled = false;

    if (messages.length === 0) {
      addChatMessage('assistant', i18n[currentLanguage].ready);
    }
  }

  window.deleteChatSettings = function() {
    const t = i18n[currentLanguage];
    if (!confirm(t.deleteConfirm)) return;

    // Clear localStorage
    localStorage.removeItem('openrouter-api-key');
    localStorage.removeItem('openrouter-model');
    localStorage.removeItem('chat-sessions');
    localStorage.removeItem('current-session-id');

    // Reset UI to initial state
    document.getElementById('chat-api-key').value = '';
    document.getElementById('chat-setup-wrapper').style.display = 'block';
    document.getElementById('chat-settings').classList.remove('hidden');
    document.getElementById('chat-settings-delete').style.display = 'none';
    document.getElementById('chat-user-input').disabled = true;
    document.getElementById('chat-send-btn').disabled = true;

    // Clear messages
    messages = [];
    document.getElementById('chat-messages').innerHTML = '';

    // Reset sessions
    sessions = [];
    currentSessionId = null;
    const newSession = createSession();
    sessions.push(newSession);
    currentSessionId = newSession.id;
    saveSessions();
    updateSessionDisplay();
  };

  window.toggleChatSettings = function() {
    document.getElementById('chat-settings').classList.toggle('hidden');
  };

  window.toggleSessionDropdown = function() {
    const dropdown = document.getElementById('chat-session-dropdown');
    dropdown.classList.toggle('hidden');
  };

  window.createNewSession = function() {
    const newSession = createSession();
    sessions.push(newSession);
    currentSessionId = newSession.id;
    messages = [];

    saveSessions();

    const messagesDiv = document.getElementById('chat-messages');
    if (messagesDiv) {
      messagesDiv.innerHTML = '';
      addChatMessage('assistant', i18n[currentLanguage].ready);
    }

    updateSessionDisplay();

    // Close dropdown
    document.getElementById('chat-session-dropdown')?.classList.add('hidden');
  };

  window.switchSession = function(sessionId) {
    if (sessionId === currentSessionId) {
      document.getElementById('chat-session-dropdown')?.classList.add('hidden');
      return;
    }

    currentSessionId = sessionId;
    loadCurrentSession();
    saveSessions();

    // Close dropdown
    document.getElementById('chat-session-dropdown')?.classList.add('hidden');
  };

  window.deleteSession = function(event, sessionId) {
    event.stopPropagation();
    sessions = sessions.filter(s => s.id !== sessionId);

    if (sessionId === currentSessionId) {
      if (sessions.length > 0) {
        currentSessionId = sessions[0].id;
        loadCurrentSession();
      } else {
        window.createNewSession();
      }
    }

    saveSessions();
    renderSessionList();
  };

  function generateSessionTitle(firstUserMessage) {
    // Generate title from first user message (simple truncation)
    const session = sessions.find(s => s.id === currentSessionId);
    if (!session) return;

    // Remove extra whitespace and take first 30 characters
    const title = firstUserMessage.trim().replace(/\s+/g, ' ').substring(0, 30);
    session.title = title;
    saveSessions();
    updateSessionDisplay();
  }

  window.sendChatMessage = async function() {
    const input = document.getElementById('chat-user-input');
    const userMessage = input.value.trim();

    if (!userMessage) return;

    messages.push({ role: 'user', content: userMessage });
    addChatMessage('user', userMessage);
    saveCurrentSession();
    input.value = '';
    input.style.height = 'auto';

    document.getElementById('chat-send-btn').disabled = true;
    document.getElementById('chat-user-input').disabled = true;

    const loadingId = addChatMessage('assistant', i18n[currentLanguage].thinking, true);
    const contentDiv = document.getElementById(loadingId)?.querySelector('.chat-message-content');
    if (contentDiv) {
      contentDiv.classList.add('chat-loading-dots');
      contentDiv.textContent = i18n[currentLanguage].thinking;
    }

    try {
      const apiKey = localStorage.getItem('openrouter-api-key');
      const model = document.getElementById('chat-model-select').value;

      // System prompt in appropriate language
      const systemPrompt = currentLanguage === 'ja'
        ? `あなたは技術書の内容に基づいて質問に答えるアシスタントです。

**サイト情報:**
- サイト名: Naoto's Books
- サイトURL: https://naoto-iwase.github.io/books
- 著者: Naoto Iwase
- 内容: 機械学習・深層学習に関する技術的なまとめ集

**現在のページの内容:**

${currentContent}

**回答時の注意:**
- 上記の内容に基づいて、正確かつ分かりやすく回答してください
- 専門用語は適切に説明し、必要に応じて数式や図の説明も含めてください
- リンクを提示する際は以下の形式を使用してください：
  - 言語別の一覧ページ: https://naoto-iwase.github.io/books/#{lang}（例: https://naoto-iwase.github.io/books/#ja）
  - 個別ページ: https://naoto-iwase.github.io/books/{lang}/{book}/{page}.html（例: https://naoto-iwase.github.io/books/ja/olmo-3/03-midtraining.html）`
        : `You are an assistant that answers questions based on technical documentation.

**Site Information:**
- Site: Naoto's Books
- Site URL: https://naoto-iwase.github.io/books
- Author: Naoto Iwase
- Content: Technical summaries on machine learning and deep learning

**Current Page Content:**

${currentContent}

**Response Guidelines:**
- Provide accurate and clear answers based on the above content
- Explain technical terms appropriately and include explanations of formulas and figures when necessary
- When providing links, use the following formats:
  - Book listing by language: https://naoto-iwase.github.io/books/#{lang} (e.g., https://naoto-iwase.github.io/books/#en)
  - Individual pages: https://naoto-iwase.github.io/books/{lang}/{book}/{page}.html (e.g., https://naoto-iwase.github.io/books/en/pdlt/04-neural-tangent-kernel.html)`;

      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': window.location.href,
          'X-Title': 'Books Chat'
        },
        body: JSON.stringify({
          model: model,
          stream: true,
          messages: [
            {
              role: 'system',
              content: systemPrompt
            },
            ...messages
          ]
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage = 'API request failed';
        try {
          const errorJson = JSON.parse(errorText);
          errorMessage = errorJson.error?.message || errorMessage;
        } catch (e) {
          errorMessage = errorText || errorMessage;
        }
        throw new Error(errorMessage);
      }

      // Stream response - keep loading message until first content arrives
      let assistantMessage = '';
      let firstContentReceived = false;

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(line => line.trim());

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          const data = line.substring(6);
          if (data === '[DONE]') break;

          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices[0]?.delta?.content;
            if (content) {
              assistantMessage += content;

              // Replace loading message with actual content on first chunk
              if (!firstContentReceived) {
                firstContentReceived = true;
              }

              updateChatMessage(loadingId, assistantMessage);
            }
          } catch (e) {
            // Skip invalid JSON
          }
        }
      }

      messages.push({ role: 'assistant', content: assistantMessage });
      saveCurrentSession();

      // Generate session title if not yet generated
      const session = sessions.find(s => s.id === currentSessionId);
      if (session && session.title === i18n[currentLanguage].newSession) {
        generateSessionTitle(messages[0].content);
      }

    } catch (error) {
      console.error('Error:', error);
      const loadingEl = document.getElementById(loadingId);
      if (loadingEl) loadingEl.remove();
      addChatMessage('error', `${i18n[currentLanguage].error} ${error.message}`);
    } finally {
      document.getElementById('chat-send-btn').disabled = false;
      document.getElementById('chat-user-input').disabled = false;
      document.getElementById('chat-user-input').focus();
    }
  };

  function addChatMessage(role, content) {
    const messagesDiv = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    const messageId = 'msg-' + Date.now() + '-' + Math.random().toString(36).substring(2, 11);
    messageDiv.id = messageId;
    messageDiv.className = `chat-message ${role}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'chat-message-content';

    if (role === 'error') {
      contentDiv.className = 'chat-error';
      contentDiv.textContent = content;
    } else if (role === 'assistant') {
      contentDiv.innerHTML = marked.parse(content || '');
      contentDiv.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
      });
    } else {
      contentDiv.textContent = content;
    }

    messageDiv.appendChild(contentDiv);
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    return messageId;
  }

  function updateChatMessage(messageId, content) {
    const contentDiv = document.getElementById(messageId)?.querySelector('.chat-message-content');
    if (!contentDiv) return;

    contentDiv.classList.remove('chat-loading-dots');
    contentDiv.innerHTML = marked.parse(content);
    contentDiv.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));

    const messagesDiv = document.getElementById('chat-messages');
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }

  // Expose toggle function globally
  window.toggleChat = toggleChat;
})();
