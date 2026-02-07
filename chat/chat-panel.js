/**
 * AI Chat Panel - OpenRouter Integration
 *
 * Design Principles:
 * - KISS: Keep it simple - favor clarity over cleverness
 * - YAGNI: Build only what's needed, when it's needed
 * - Vanilla JS: No frameworks - easy to understand and maintain
 * - Beginner-friendly: Clear naming, consistent patterns, minimal abstractions
 *
 * Features:
 * - Multi-language support (Japanese/English)
 * - Session management with localStorage persistence
 * - Streaming responses with markdown rendering
 * - Math rendering with KaTeX (supports $...$ and \[...\] notation)
 * - Tool calling support (site search)
 * - Chat export to Markdown
 * - Responsive design with resizable panel
 * - Dark mode support (Quarto theme integration)
 *
 * External Dependencies:
 * - marked.js: Markdown parsing
 * - marked-katex-extension: Math rendering integration
 * - KaTeX: LaTeX math rendering
 * - highlight.js: Code syntax highlighting
 * - OpenRouter API: LLM inference
 */
(function() {
  // ============================================================
  // DEFINITIONS - Constants, State, Static Data
  // ============================================================

  const CONFIG = {
    // Language
    DEFAULT_LANGUAGE: 'en',

    // Text length limits
    SESSION_TITLE_MAX_LENGTH: 30,
    SESSION_TITLE_DISPLAY_LENGTH: 20,
    SESSION_LIST_TITLE_MAX: 40,
    SESSION_PAGE_DISPLAY_MAX: 60,

    // Panel dimensions
    PANEL_MIN_WIDTH: 300,
    PANEL_MAX_WIDTH_RATIO: 0.9,

    // UI limits
    TEXTAREA_MAX_HEIGHT: 150,
    SESSION_LIST_MAX_HEIGHT: 300,

    // Timing
    MODEL_LOAD_DELAY: 500,

    // Model sorting
    DEFAULT_PRIORITY_INDEX: 999,
    PRICING_DISPLAY_MULTIPLIER: 1000000,
  };

  // State
  let models = [];
  let messages = [];
  let currentContent = '';
  let isOpen = false;
  let currentLanguage = CONFIG.DEFAULT_LANGUAGE;
  let sessions = [];
  let currentSessionId = null;
  let searchIndex = null;

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
      removeApiKey: "API Keyを削除",
      deleteAllSessions: "全履歴を削除",
      validating: "確認中...",
      invalidApiKey: "API Keyが無効です。正しいキーを入力してください。",
      siteStructure: "**サイト全体の構成 (Recent 5 books per language):**",
      contentLoadError: "このページの内容を読み込めませんでした。一般的な質問には答えられます。",
      removeApiKeyConfirm: "API Keyを削除しますか？チャット履歴は保持されます。",
      deleteAllSessionsConfirm: "全てのチャット履歴を削除しますか？",
      securityWarning: "⚠️ **セキュリティ上の注意**: API KeyはブラウザのlocalStorageに平文で保存されます。共有端末では使用後に必ず削除してください。",
      searching: "検索中...",
      searchResults: "検索結果",
      exportChat: "エクスポート",
      noSearchResults: "該当する記事が見つかりませんでした。",
      noResponse: "回答を生成できませんでした。別のモデルをお試しください。"
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
      removeApiKey: "Remove API Key",
      deleteAllSessions: "Delete All History",
      validating: "Validating...",
      invalidApiKey: "Invalid API Key. Please enter a valid key.",
      siteStructure: "**Site Structure (Recent 5 books per language):**",
      contentLoadError: "Failed to load page content. General questions can still be answered.",
      removeApiKeyConfirm: "Remove API Key? Chat history will be preserved.",
      deleteAllSessionsConfirm: "Delete all chat history?",
      securityWarning: "⚠️ **Security Notice**: API Key is stored in plain text in browser's localStorage. Always remove it after use on shared devices.",
      searching: "Searching...",
      searchResults: "Search Results",
      exportChat: "Export",
      noSearchResults: "No matching articles found.",
      noResponse: "Failed to generate a response. Please try a different model."
    }
  };

  // Tool definition for site search
  const SEARCH_TOOL = {
    type: 'function',
    function: {
      name: 'search_site',
      description: 'Search across all pages on this technical books site. Use this to find related content, other chapters, or specific topics mentioned elsewhere on the site.',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'Search query in Japanese or English'
          }
        },
        required: ['query']
      }
    }
  };

  // ============================================================
  // UTILITIES - Pure Helper Functions
  // ============================================================

  function detectLanguage() {
    const path = window.location.pathname;
    if (path.includes('/ja/')) return 'ja';
    if (path.includes('/en/')) return 'en';
    return CONFIG.DEFAULT_LANGUAGE;
  }

  // Convert \[...\] and \(...\) to $$...$$ and $...$ for marked-katex-extension
  // Use negative lookbehind to avoid converting \\[ (LaTeX line break with spacing)
  function normalizeLatexDelimiters(text) {
    return text
      .replace(/(?<!\\)\\\[/g, '$$')
      .replace(/(?<!\\)\\\]/g, '$$')
      .replace(/(?<!\\)\\\(/g, '$')
      .replace(/(?<!\\)\\\)/g, '$');
  }

  function formatDateTime(date) {
    const pad = n => String(n).padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
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

  // ============================================================
  // API / EXTERNAL - Data Fetching & External Communication
  // ============================================================

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
          return index === -1 ? CONFIG.DEFAULT_PRIORITY_INDEX : index;
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
          ? ` ($${(parseFloat(model.pricing.prompt) * CONFIG.PRICING_DISPLAY_MULTIPLIER).toFixed(2)}/1M)`
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

  async function loadSearchIndex() {
    if (searchIndex) return searchIndex;
    const basePath = window.location.pathname.startsWith('/books/') ? '/books' : '';
    const url = `${basePath}/search.json`;

    try {
      const response = await fetch(url);
      if (!response.ok) return [];
      searchIndex = await response.json();
      return searchIndex;
    } catch (e) {
      return [];
    }
  }

  async function searchSite(query) {
    const index = await loadSearchIndex();
    const queryLower = query.toLowerCase();
    // Allow single-character terms for Japanese (e.g., "木", "AI")
    const queryTerms = queryLower.split(/\s+/).filter(t => t.length > 0);

    const scored = index
      .map(item => {
        let score = 0;
        const titleLower = (item.title || '').toLowerCase();
        const textLower = (item.text || '').toLowerCase();
        const sectionLower = (item.section || '').toLowerCase();

        for (const term of queryTerms) {
          if (titleLower.includes(term)) score += 10;
          if (sectionLower.includes(term)) score += 5;
          if (textLower.includes(term)) score += 1;
        }
        return { ...item, score };
      })
      .filter(item => item.score > 0)
      .sort((a, b) => b.score - a.score);

    // Deduplicate by href (keep highest scoring entry per page)
    const seenHrefs = new Set();
    const results = [];
    for (const item of scored) {
      if (seenHrefs.has(item.href)) continue;
      seenHrefs.add(item.href);
      results.push({
        title: item.title,
        section: item.section,
        href: item.href,
        snippet: (item.text || '').substring(0, 200) + (item.text?.length > 200 ? '...' : '')
      });
      if (results.length >= 5) break;
    }

    return results;
  }

  async function executeToolCall(toolCall) {
    try {
      if (!toolCall?.function?.name || !toolCall?.function?.arguments) {
        return JSON.stringify({ error: 'Invalid tool call structure' });
      }

      const { name, arguments: argsStr } = toolCall.function;
      let args;
      try {
        args = JSON.parse(argsStr);
      } catch (e) {
        return JSON.stringify({ error: 'Failed to parse tool arguments' });
      }

      if (name === 'search_site') {
        const results = await searchSite(args.query);
        if (results.length === 0) {
          return i18n[currentLanguage].noSearchResults;
        }
        return JSON.stringify(results, null, 2);
      }
      return JSON.stringify({ error: 'Unknown tool: ' + name });
    } catch (e) {
      return JSON.stringify({ error: e.message });
    }
  }

  function isToolCallingSupported(modelId) {
    const model = models.find(m => m.id === modelId);
    return model?.supported_parameters?.includes('tools') ?? false;
  }

  // System prompts
  function buildJapaneseSystemPrompt(content) {
    return `あなたは技術書の内容に基づいて質問に答えるアシスタントです。

**サイト情報:**
- サイト名: Naoto's Books
- サイトURL: https://naoto-iwase.github.io/books
- 著者: Naoto Iwase
- 内容: 機械学習・深層学習に関する技術的なまとめ集

**現在のページの内容:**

${content}

**回答時の注意:**
- 上記の内容に基づいて、正確かつ分かりやすく回答してください
- 専門用語は適切に説明し、必要に応じて数式や図の説明も含めてください
- 数式はLaTeX形式で記述してください
- リンクを提示する際は以下の形式を使用してください：
  - 言語別の一覧ページ: https://naoto-iwase.github.io/books/#{lang}（例: https://naoto-iwase.github.io/books/#ja）
  - 個別ページ: https://naoto-iwase.github.io/books/{lang}/{book}/{page}.html（例: https://naoto-iwase.github.io/books/ja/olmo-3/03-midtraining.html）`;
  }

  function buildEnglishSystemPrompt(content) {
    return `You are an assistant that answers questions based on technical documentation.

**Site Information:**
- Site: Naoto's Books
- Site URL: https://naoto-iwase.github.io/books
- Author: Naoto Iwase
- Content: Technical summaries on machine learning and deep learning

**Current Page Content:**

${content}

**Response Guidelines:**
- Provide accurate and clear answers based on the above content
- Explain technical terms appropriately and include explanations of formulas and figures when necessary
- Use LaTeX for math expressions
- When providing links, use the following formats:
  - Book listing by language: https://naoto-iwase.github.io/books/#{lang} (e.g., https://naoto-iwase.github.io/books/#en)
  - Individual pages: https://naoto-iwase.github.io/books/{lang}/{book}/{page}.html (e.g., https://naoto-iwase.github.io/books/en/pdlt/04-neural-tangent-kernel.html)`;
  }

  // ============================================================
  // SESSION MANAGEMENT - Persistence & History
  // ============================================================

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
        session.title = firstUserMsg.content.trim().replace(/\s+/g, ' ').substring(0, CONFIG.SESSION_TITLE_MAX_LENGTH);
      }
    }

    saveSessions();
    updateSessionDisplay();
  }

  function updateSessionDisplay() {
    const session = sessions.find(s => s.id === currentSessionId);
    if (!session) return;

    const currentSessionName = document.getElementById('chat-current-session-name');
    if (currentSessionName) {
      const title = session.title || i18n[currentLanguage].newSession;
      currentSessionName.textContent = title.substring(0, CONFIG.SESSION_TITLE_DISPLAY_LENGTH);
    }

    renderSessionList();
  }

  function renderSessionList() {
    const sessionList = document.getElementById('chat-session-list');
    if (!sessionList) return;

    sessionList.innerHTML = '';

    sessions.sort((a, b) => b.updated - a.updated).forEach(session => {
      const item = document.createElement('div');
      item.className = 'chat-session-item' + (session.id === currentSessionId ? ' active' : '');

      const date = new Date(session.updated);
      const dateStr = `${date.getMonth() + 1}/${date.getDate()} ${date.getHours()}:${String(date.getMinutes()).padStart(2, '0')}`;
      const title = (session.title || i18n[currentLanguage].newSession).substring(0, CONFIG.SESSION_LIST_TITLE_MAX);
      const page = session.page ? session.page.substring(0, CONFIG.SESSION_PAGE_DISPLAY_MAX) : '';

      item.innerHTML = `
        <div class="chat-session-info" onclick="window.switchSession('${session.id}')">
          <div class="chat-session-date">${dateStr}</div>
          <div class="chat-session-title">${title}</div>
          ${page ? `<div class="chat-session-page">@ ${page}</div>` : ''}
        </div>
        <button class="chat-session-delete" onclick="window.deleteSession(event, '${session.id}')" title="Delete">×</button>
      `;

      sessionList.appendChild(item);
    });
  }

  // ============================================================
  // UI CONSTRUCTION - DOM Building
  // ============================================================

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
              <div class="chat-session-delete-all-wrapper">
                <button class="chat-session-delete-all-btn" id="chat-session-delete-all-btn" onclick="window.deleteAllSessions()">🗑️ Delete All History</button>
              </div>
            </div>
          </div>
          <div class="chat-header-buttons">
            <button class="chat-new-btn" id="chat-new-btn" onclick="window.createNewSession()" title="New chat">＋</button>
            <button class="chat-export-btn" id="chat-export-btn" onclick="window.exportChatAsMarkdown()" title="Export">⤓</button>
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
            <button class="chat-settings-remove" id="chat-settings-remove" onclick="window.removeApiKey()" style="display: none;">Remove API Key</button>
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
        </div>
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
        sendChatMessage();
      }
    });

    // Auto-resize textarea based on content
    userInput.addEventListener('input', function() {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, CONFIG.TEXTAREA_MAX_HEIGHT) + 'px';
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
      const newWidth = Math.min(Math.max(CONFIG.PANEL_MIN_WIDTH, startWidth + deltaX), window.innerWidth * CONFIG.PANEL_MAX_WIDTH_RATIO);
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

  // ============================================================
  // UI STATE - Display State Management
  // ============================================================

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

  function updateLanguage(lang) {
    currentLanguage = lang;
    const t = i18n[lang];

    // Update text content for each element
    const updates = {
      'chat-user-input': { placeholder: t.inputPlaceholder },
      'chat-send-btn': { textContent: 'Enter' },
      'chat-setup-msg': { textContent: t.setupMessage },
      'chat-info-title': { textContent: t.infoTitle },
      'chat-info-bullets': { innerHTML: t.infoBullets.map(b => '• ' + b).join('<br>') },
      'chat-resize-hint': { textContent: t.resizeHint },
      'chat-settings-save-btn': { textContent: t.saveSettings },
      'chat-settings-remove': { textContent: t.removeApiKey },
      'chat-session-delete-all-btn': { textContent: '🗑️ ' + t.deleteAllSessions },
      'chat-export-btn': { title: t.exportChat }
    };

    for (const [id, props] of Object.entries(updates)) {
      const el = document.getElementById(id);
      if (el) Object.assign(el, props);
    }

    const btn = document.querySelector('.chat-toggle-btn');
    if (btn) btn.textContent = isOpen ? t.closeBtn : t.openBtn;
  }

  // ============================================================
  // MESSAGES - Display & Update
  // ============================================================

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
      contentDiv.innerHTML = marked.parse(normalizeLatexDelimiters(content || ''));
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
    contentDiv.innerHTML = marked.parse(normalizeLatexDelimiters(content));
    contentDiv.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));

    const messagesDiv = document.getElementById('chat-messages');
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }

  // ============================================================
  // CORE FEATURES - Main Functionality
  // ============================================================

  async function sendChatMessage() {
    const userInput = document.getElementById('chat-user-input');
    const userMessage = userInput.value.trim();

    if (!userMessage) return;

    messages.push({ role: 'user', content: userMessage });
    addChatMessage('user', userMessage);
    saveCurrentSession();
    userInput.value = '';
    userInput.style.height = 'auto';

    const sendBtn = document.getElementById('chat-send-btn');
    sendBtn.disabled = true;
    userInput.disabled = true;

    const loadingId = addChatMessage('assistant', i18n[currentLanguage].thinking, true);
    const contentDiv = document.getElementById(loadingId)?.querySelector('.chat-message-content');
    if (contentDiv) {
      contentDiv.classList.add('chat-loading-dots');
      contentDiv.textContent = i18n[currentLanguage].thinking;
    }

    try {
      const apiKey = localStorage.getItem('openrouter-api-key');
      const model = document.getElementById('chat-model-select').value;
      const supportsTools = isToolCallingSupported(model);

      // Build system prompt in appropriate language
      const systemPrompt = currentLanguage === 'ja'
        ? buildJapaneseSystemPrompt(currentContent)
        : buildEnglishSystemPrompt(currentContent);

      // Conversation messages for API (separate from display messages)
      const apiMessages = [
        { role: 'system', content: systemPrompt },
        ...messages
      ];

      let assistantMessage = '';
      let toolCallLoop = 0;
      const maxToolCalls = 2;  // Reduced to prevent slow loops
      let lastSearchResults = null;  // Store for fallback display
      let lastSearchQuery = null;    // Store query for fallback display
      const executedQueries = new Set();  // Prevent duplicate tool calls

      // Tool calling loop
      while (toolCallLoop < maxToolCalls) {
        toolCallLoop++;

        const requestBody = {
          model: model,
          stream: true,
          messages: apiMessages
        };

        // Add tools only if model supports them AND not on last iteration
        // On last iteration, force model to generate final answer by not providing tools
        const isLastIteration = toolCallLoop >= maxToolCalls;
        if (supportsTools && !isLastIteration) {
          requestBody.tools = [SEARCH_TOOL];
          requestBody.tool_choice = 'auto';
        }

        const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
            'HTTP-Referer': window.location.href,
            'X-Title': 'Books Chat'
          },
          body: JSON.stringify(requestBody)
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

        // Stream response with SSE parsing
        assistantMessage = '';
        let firstContentReceived = false;
        let toolCalls = [];
        let finishReason = null;

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split('\n');
          buffer = parts.pop(); // Keep incomplete last line

          for (const line of parts) {
            const trimmed = line.trim();
            if (!trimmed || !trimmed.startsWith('data: ')) continue;

            const data = trimmed.substring(6).trim();
            if (data === '[DONE]' || data === '') continue;

            // Skip non-JSON data
            if (!data.startsWith('{')) continue;

            try {
              const parsed = JSON.parse(data);
              const choice = parsed.choices?.[0];
              if (!choice) continue;

              if (choice.finish_reason) {
                finishReason = choice.finish_reason;
              }

              const content = choice.delta?.content;
              if (content) {
                assistantMessage += content;
                if (!firstContentReceived) firstContentReceived = true;
                updateChatMessage(loadingId, assistantMessage);
              }

              const deltaToolCalls = choice.delta?.tool_calls;
              if (deltaToolCalls) {
                for (const tc of deltaToolCalls) {
                  const idx = tc.index ?? 0;
                  if (!toolCalls[idx]) {
                    toolCalls[idx] = { id: tc.id || '', function: { name: '', arguments: '' } };
                  }
                  if (tc.id) toolCalls[idx].id = tc.id;
                  if (tc.function?.name) toolCalls[idx].function.name += tc.function.name;
                  if (tc.function?.arguments) toolCalls[idx].function.arguments += tc.function.arguments;
                }
              }
            } catch (e) {
              // Skip invalid JSON (likely chunk boundary issues)
            }
          }
        }

        // Check if we need to execute tools
        const shouldExecuteTools = (finishReason === 'tool_calls' || finishReason === 'function_call')
          && toolCalls.length > 0
          && toolCalls.every(tc => tc.id && tc.function.name && tc.function.arguments);

        if (shouldExecuteTools) {
          updateChatMessage(loadingId, i18n[currentLanguage].searching);

          const assistantToolMsg = {
            role: 'assistant',
            content: assistantMessage || null,
            tool_calls: toolCalls.map(tc => ({
              id: tc.id,
              type: 'function',
              function: tc.function
            }))
          };
          apiMessages.push(assistantToolMsg);

          // Execute each tool and add results
          let allDuplicates = true;
          for (const tc of toolCalls) {
            // Check for duplicate query
            try {
              const args = JSON.parse(tc.function.arguments);
              const queryKey = `${tc.function.name}:${JSON.stringify(args)}`;
              if (executedQueries.has(queryKey)) {
                apiMessages.push({
                  role: 'tool',
                  tool_call_id: tc.id,
                  content: JSON.stringify({ note: 'Already searched, see previous results' })
                });
                continue;
              }
              executedQueries.add(queryKey);
              allDuplicates = false;

              // Save query for fallback display
              if (tc.function.name === 'search_site' && args.query) {
                lastSearchQuery = args.query;
              }
            } catch (e) {
              allDuplicates = false;
            }

            const result = await executeToolCall(tc);
            lastSearchResults = result;

            // Show search results to user in real-time
            try {
              const resultData = JSON.parse(result);
              if (Array.isArray(resultData) && resultData.length > 0) {
                const resultPreview = resultData.map(r =>
                  `- [${r.title}](${r.href})${r.section ? ` (${r.section})` : ''}`
                ).join('\n');
                updateChatMessage(loadingId, `🔍 **${i18n[currentLanguage].searching}**\n\n${resultPreview}\n\n---\n\n`);
              }
            } catch (e) {
              updateChatMessage(loadingId, `🔍 ${result}`);
            }

            apiMessages.push({
              role: 'tool',
              tool_call_id: tc.id,
              content: result
            });
          }

          if (allDuplicates) break;
          continue;
        }

        // No more tool calls, exit loop
        break;
      }

      // If no content was generated but we had search results, show them
      const t = i18n[currentLanguage];
      if (!assistantMessage && lastSearchResults) {
        try {
          const resultData = JSON.parse(lastSearchResults);
          if (Array.isArray(resultData) && resultData.length > 0) {
            assistantMessage = lastSearchQuery
              ? `🔍 **${t.searchResults}**（「${lastSearchQuery}」）\n\n`
              : `🔍 **${t.searchResults}**\n\n`;
            for (const r of resultData) {
              assistantMessage += `### [${r.title}](${r.href})\n`;
              if (r.section) assistantMessage += `**${r.section}**\n`;
              if (r.snippet) assistantMessage += `${r.snippet}\n`;
              assistantMessage += '\n';
            }
          }
        } catch (e) {
          assistantMessage = lastSearchResults;
        }
      }

      // If still no content, show error message
      if (!assistantMessage) {
        assistantMessage = t.noResponse;
      }

      updateChatMessage(loadingId, assistantMessage);
      messages.push({ role: 'assistant', content: assistantMessage });
      saveCurrentSession();

    } catch (error) {
      console.error('Error:', error);
      const loadingEl = document.getElementById(loadingId);
      if (loadingEl) loadingEl.remove();
      addChatMessage('error', `${i18n[currentLanguage].error} ${error.message}`);
    } finally {
      sendBtn.disabled = false;
      userInput.disabled = false;
      userInput.focus();
    }
  }

  async function saveChatSettings() {
    const apiKey = document.getElementById('chat-api-key').value.trim();
    const t = i18n[currentLanguage];
    const saveBtn = document.querySelector('.chat-settings-save');

    if (!apiKey) {
      alert(t.apiKeyRequired);
      return;
    }

    // Check if this is a new API key registration
    const hadApiKey = !!localStorage.getItem('openrouter-api-key');

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

      // Show remove button
      document.getElementById('chat-settings-remove').style.display = 'block';

      // Hide welcome message and enable chat
      document.getElementById('chat-setup-wrapper').classList.add('hidden');
      document.getElementById('chat-settings').classList.add('hidden');
      document.getElementById('chat-user-input').disabled = false;
      document.getElementById('chat-send-btn').disabled = false;

      // Show messages for initial setup only
      if (messages.length === 0) {
        if (!hadApiKey) {
          addChatMessage('assistant', i18n[currentLanguage].securityWarning);
        }
        addChatMessage('assistant', i18n[currentLanguage].ready);
      }
    } catch (error) {
      alert(t.error + ' ' + error.message);
      saveBtn.disabled = false;
      saveBtn.textContent = originalText;
    }
  }

  function removeApiKey() {
    const t = i18n[currentLanguage];
    if (!confirm(t.removeApiKeyConfirm)) return;

    // Clear only API Key and model settings
    localStorage.removeItem('openrouter-api-key');
    localStorage.removeItem('openrouter-model');

    // Reset UI to initial state (but keep chat history)
    document.getElementById('chat-api-key').value = '';
    document.getElementById('chat-settings').classList.remove('hidden');
    document.getElementById('chat-setup-wrapper').classList.remove('hidden');
    document.getElementById('chat-settings-remove').style.display = 'none';
    document.getElementById('chat-user-input').disabled = true;
    document.getElementById('chat-send-btn').disabled = true;

    // Keep messages and sessions intact
  }

  function deleteAllSessions() {
    const t = i18n[currentLanguage];
    if (!confirm(t.deleteAllSessionsConfirm)) return;

    // Clear only chat history (keep API key)
    localStorage.removeItem('chat-sessions');
    localStorage.removeItem('current-session-id');

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

    // Close dropdown
    const dropdown = document.getElementById('chat-session-dropdown');
    if (dropdown) dropdown.classList.add('hidden');

    // Show ready message
    addChatMessage('assistant', i18n[currentLanguage].ready);
  }

  function toggleChatSettings() {
    document.getElementById('chat-settings').classList.toggle('hidden');
  }

  function toggleSessionDropdown() {
    document.getElementById('chat-session-dropdown').classList.toggle('hidden');
  }

  function createNewSession() {
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
    const dropdown = document.getElementById('chat-session-dropdown');
    if (dropdown) dropdown.classList.add('hidden');
  }

  function switchSession(sessionId) {
    const dropdown = document.getElementById('chat-session-dropdown');
    if (sessionId === currentSessionId) {
      if (dropdown) dropdown.classList.add('hidden');
      return;
    }

    currentSessionId = sessionId;
    loadCurrentSession();
    saveSessions();

    // Close dropdown
    if (dropdown) dropdown.classList.add('hidden');
  }

  function deleteSession(event, sessionId) {
    event.stopPropagation();
    sessions = sessions.filter(s => s.id !== sessionId);

    if (sessionId === currentSessionId) {
      if (sessions.length > 0) {
        currentSessionId = sessions[0].id;
        loadCurrentSession();
      } else {
        createNewSession();
      }
    }

    saveSessions();
    renderSessionList();
  }

  function exportChatAsMarkdown() {
    if (messages.length === 0) return;

    const model = document.getElementById('chat-model-select')?.value || 'Unknown';
    const now = new Date();
    const dateDisplay = formatDateTime(now);
    const dateFilename = formatDateTime(now).replace(' ', '-').replace(':', '');

    let md = `# Chat Export\n\n`;
    md += `- **Date**: ${dateDisplay}\n`;
    md += `- **Model**: ${model}\n`;
    md += `- **Page**: ${window.location.href}\n\n---\n\n`;

    for (const msg of messages) {
      if (msg.role === 'user') {
        md += `## User\n\n${msg.content}\n\n`;
      } else if (msg.role === 'assistant') {
        md += `## Assistant\n\n${msg.content}\n\n`;
      }
    }

    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-export-${dateFilename}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // ============================================================
  // INITIALIZATION
  // ============================================================

  function init() {
    // Configure marked with KaTeX extension for math rendering
    if (window.markedKatex) {
      marked.use(markedKatex({
        throwOnError: false,
        nonStandard: true  // Allow $x$ without spaces
      }));
    }

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
      document.getElementById('chat-setup-wrapper').classList.add('hidden');
      document.getElementById('chat-user-input').disabled = false;
      document.getElementById('chat-send-btn').disabled = false;
      document.getElementById('chat-settings-remove').style.display = 'block';

      const savedModel = localStorage.getItem('openrouter-model');
      if (savedModel) {
        setTimeout(() => {
          const select = document.getElementById('chat-model-select');
          if (select) {
            select.value = savedModel;
            updateCurrentModelDisplay();
          }
        }, CONFIG.MODEL_LOAD_DELAY);
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

  // DOM Ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // ============================================================
  // GLOBAL EXPORTS - Expose functions to window
  // ============================================================

  Object.assign(window, {
    toggleChat,
    sendChatMessage,
    saveChatSettings,
    removeApiKey,
    deleteAllSessions,
    toggleChatSettings,
    toggleSessionDropdown,
    createNewSession,
    switchSession,
    deleteSession,
    exportChatAsMarkdown
  });
})();
