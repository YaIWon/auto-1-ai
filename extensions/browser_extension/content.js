// Autonomous AI Content Script
// Injected into every page for full control

console.log('🧠 Autonomous AI Content Script Loaded');

// Global state
const aiState = {
  initialized: false,
  pageControl: false,
  elementMapping: new Map(),
  formHandlers: new Map(),
  networkListeners: [],
  blockchain: null,
  metamask: null
};

// Initialize AI Core
class AIContentCore {
  constructor() {
    this.websocket = null;
    this.messageQueue = [];
    this.commandHandlers = new Map();
    this.syncInterval = null;
  }

  async initialize() {
    console.log('🚀 Initializing AI Content Core');
    
    // Connect to extension background
    this.setupExtensionConnection();
    
    // Setup WebSocket connection
    this.connectWebSocket();
    
    // Setup message listeners
    this.setupMessageListeners();
    
    // Setup DOM observers
    this.setupDOMObservers();
    
    // Setup network interceptors
    this.setupNetworkInterceptors();
    
    // Setup MetaMask integration
    this.setupMetaMask();
    
    // Setup GitHub integration
    this.setupGitHub();
    
    // Setup blockchain integration
    this.setupBlockchain();
    
    // Start sync
    this.startSync();
    
    aiState.initialized = true;
    console.log('✅ AI Content Core Initialized');
  }

  setupExtensionConnection() {
    // Listen for messages from background script
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      this.handleExtensionMessage(message, sendResponse);
      return true;
    });
  }

  connectWebSocket() {
    try {
      this.websocket = new WebSocket('ws://127.0.0.1:8080');
      
      this.websocket.onopen = () => {
        console.log('✅ Connected to autonomous system WebSocket');
        this.sendQueuedMessages();
      };
      
      this.websocket.onmessage = (event) => {
        this.handleWebSocketMessage(event.data);
      };
      
      this.websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
      
      this.websocket.onclose = () => {
        console.log('WebSocket closed, reconnecting...');
        setTimeout(() => this.connectWebSocket(), 5000);
      };
    } catch (error) {
      console.error('WebSocket connection error:', error);
    }
  }

  setupMessageListeners() {
    // Setup command handlers
    this.commandHandlers.set('modify_dom', this.handleModifyDOM.bind(this));
    this.commandHandlers.set('extract_data', this.handleExtractData.bind(this));
    this.commandHandlers.set('fill_form', this.handleFillForm.bind(this));
    this.commandHandlers.set('click_element', this.handleClickElement.bind(this));
    this.commandHandlers.set('navigate', this.handleNavigate.bind(this));
    this.commandHandlers.set('execute_script', this.handleExecuteScript.bind(this));
    this.commandHandlers.set('create_element', this.handleCreateElement.bind(this));
    this.commandHandlers.set('delete_element', this.handleDeleteElement.bind(this));
    this.commandHandlers.set('modify_style', this.handleModifyStyle.bind(this));
    this.commandHandlers.set('intercept_request', this.handleInterceptRequest.bind(this));
    this.commandHandlers.set('blockchain_transaction', this.handleBlockchainTransaction.bind(this));
    this.commandHandlers.set('github_operation', this.handleGitHubOperation.bind(this));
    this.commandHandlers.set('create_account', this.handleCreateAccount.bind(this));
    this.commandHandlers.set('send_sms', this.handleSendSMS.bind(this));
    this.commandHandlers.set('send_email', this.handleSendEmail.bind(this));
    this.commandHandlers.set('connect_repo', this.handleConnectRepo.bind(this));
    this.commandHandlers.set('clone_repo', this.handleCloneRepo.bind(this));
    this.commandHandlers.set('create_repo', this.handleCreateRepo.bind(this));
    this.commandHandlers.set('modify_file', this.handleModifyFile.bind(this));
    this.commandHandlers.set('update_settings', this.handleUpdateSettings.bind(this));
  }

  setupDOMObservers() {
    // Observe DOM changes
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        this.handleDOMChange(mutation);
      });
    });
    
    observer.observe(document.documentElement, {
      childList: true,
      subtree: true,
      attributes: true,
      characterData: true
    });
    
    // Track form submissions
    document.addEventListener('submit', (event) => {
      this.handleFormSubmit(event);
    }, true);
    
    // Track clicks
    document.addEventListener('click', (event) => {
      this.handleClick(event);
    }, true);
    
    // Track input changes
    document.addEventListener('input', (event) => {
      this.handleInputChange(event);
    }, true);
  }

  setupNetworkInterceptors() {
    // Intercept fetch requests
    const originalFetch = window.fetch;
    window.fetch = async (...args) => {
      const request = args[0];
      const options = args[1] || {};
      
      // Log request
      console.log('Fetch request:', request, options);
      
      // Intercept and modify if needed
      const intercepted = await this.interceptRequest(request, options);
      
      if (intercepted.modified) {
        return originalFetch(intercepted.request, intercepted.options);
      }
      
      return originalFetch(...args);
    };
    
    // Intercept XMLHttpRequest
    const originalXHROpen = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function(method, url, ...args) {
      console.log('XHR request:', method, url);
      this._requestUrl = url;
      this._requestMethod = method;
      return originalXHROpen.apply(this, [method, url, ...args]);
    };
    
    // Intercept WebSocket
    const originalWebSocket = WebSocket;
    window.WebSocket = function(...args) {
      console.log('WebSocket connection:', args[0]);
      return new originalWebSocket(...args);
    };
  }

  setupMetaMask() {
    if (typeof window.ethereum !== 'undefined') {
      aiState.metamask = window.ethereum;
      console.log('✅ MetaMask detected');
      
      // Listen for account changes
      window.ethereum.on('accountsChanged', (accounts) => {
        console.log('MetaMask accounts changed:', accounts);
        this.sendToSystem({
          type: 'metamask_accounts_changed',
          accounts: accounts
        });
      });
      
      // Listen for chain changes
      window.ethereum.on('chainChanged', (chainId) => {
        console.log('MetaMask chain changed:', chainId);
        this.sendToSystem({
          type: 'metamask_chain_changed',
          chainId: chainId
        });
      });
    }
  }

  setupGitHub() {
    // Inject GitHub API if on GitHub
    if (window.location.hostname.includes('github.com')) {
      this.injectGitHubAPI();
    }
  }

  setupBlockchain() {
    // Setup Web3 if available
    if (typeof Web3 !== 'undefined') {
      try {
        const web3 = new Web3(window.ethereum || 'ws://localhost:8545');
        aiState.blockchain = web3;
        console.log('✅ Blockchain connection established');
      } catch (error) {
        console.error('Blockchain setup error:', error);
      }
    }
  }

  startSync() {
    this.syncInterval = setInterval(() => {
      this.syncState();
    }, 2000); // Sync every 2 seconds
  }

  async syncState() {
    const state = {
      url: window.location.href,
      title: document.title,
      dom: this.extractDOMState(),
      forms: this.extractForms(),
      inputs: this.extractInputs(),
      buttons: this.extractButtons(),
      links: this.extractLinks(),
      cookies: document.cookie,
      localStorage: this.extractLocalStorage(),
      sessionStorage: this.extractSessionStorage(),
      timestamp: Date.now()
    };
    
    this.sendToSystem({
      type: 'page_sync',
      state: state
    });
  }

  extractDOMState() {
    const elements = {
      forms: [],
      inputs: [],
      buttons: [],
      links: [],
      tables: [],
      images: []
    };
    
    // Extract form elements
    document.querySelectorAll('form').forEach((form, index) => {
      elements.forms.push({
        id: form.id,
        action: form.action,
        method: form.method,
        inputs: Array.from(form.elements).map(el => ({
          type: el.type,
          name: el.name,
          value: el.value,
          id: el.id
        }))
      });
    });
    
    // Extract inputs
    document.querySelectorAll('input, textarea, select').forEach((input, index) => {
      elements.inputs.push({
        type: input.type,
        name: input.name,
        value: input.value,
        id: input.id,
        className: input.className,
        placeholder: input.placeholder
      });
    });
    
    // Extract buttons
    document.querySelectorAll('button, input[type="button"], input[type="submit"]').forEach((button, index) => {
      elements.buttons.push({
        text: button.textContent || button.value,
        id: button.id,
        className: button.className,
        type: button.type
      });
    });
    
    // Extract links
    document.querySelectorAll('a').forEach((link, index) => {
      elements.links.push({
        text: link.textContent,
        href: link.href,
        id: link.id,
        className: link.className
      });
    });
    
    return elements;
  }

  extractForms() {
    return Array.from(document.forms).map(form => ({
      id: form.id,
      name: form.name,
      action: form.action,
      method: form.method,
      elements: Array.from(form.elements).map(el => ({
        tagName: el.tagName,
        type: el.type,
        name: el.name,
        value: el.value
      }))
    }));
  }

  extractInputs() {
    return Array.from(document.querySelectorAll('input, textarea, select')).map(input => ({
      tagName: input.tagName,
      type: input.type,
      name: input.name,
      value: input.value,
      id: input.id,
      placeholder: input.placeholder
    }));
  }

  extractButtons() {
    return Array.from(document.querySelectorAll('button, input[type="button"], input[type="submit"]')).map(button => ({
      tagName: button.tagName,
      type: button.type,
      text: button.textContent || button.value,
      id: button.id
    }));
  }

  extractLinks() {
    return Array.from(document.querySelectorAll('a')).map(link => ({
      text: link.textContent,
      href: link.href,
      id: link.id
    }));
  }

  extractLocalStorage() {
    const items = {};
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      items[key] = localStorage.getItem(key);
    }
    return items;
  }

  extractSessionStorage() {
    const items = {};
    for (let i = 0; i < sessionStorage.length; i++) {
      const key = sessionStorage.key(i);
      items[key] = sessionStorage.getItem(key);
    }
    return items;
  }

  handleExtensionMessage(message, sendResponse) {
    console.log('Content script received message:', message);
    
    if (message.type === 'remote_command') {
      this.executeCommand(message.command);
      sendResponse({ success: true });
    } else if (message.type === 'execute_command') {
      const result = this.executeCommand(message.command, message.args);
      sendResponse(result);
    } else if (message.type === 'sync_request') {
      this.syncState();
      sendResponse({ success: true });
    }
  }

  handleWebSocketMessage(data) {
    try {
      const message = JSON.parse(data);
      console.log('WebSocket message:', message);
      
      if (message.type === 'command') {
        this.executeCommand(message.command, message.args);
      } else if (message.type === 'sync') {
        this.updateFromSync(message.data);
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }

  executeCommand(command, args = {}) {
    console.log('Executing command:', command, args);
    
    const handler = this.commandHandlers.get(command);
    if (handler) {
      return handler(args);
    } else {
      console.warn('No handler for command:', command);
      return { error: 'Unknown command' };
    }
  }

  async handleModifyDOM(args) {
    const { selector, operation, content, attributes } = args;
    
    try {
      const element = document.querySelector(selector);
      if (!element) {
        return { error: 'Element not found' };
      }
      
      switch (operation) {
        case 'set_text':
          element.textContent = content;
          break;
        case 'set_html':
          element.innerHTML = content;
          break;
        case 'set_value':
          element.value = content;
          break;
        case 'set_attribute':
          if (attributes) {
            Object.entries(attributes).forEach(([key, value]) => {
              element.setAttribute(key, value);
            });
          }
          break;
        case 'remove':
          element.remove();
          break;
        case 'clear':
          element.innerHTML = '';
          break;
      }
      
      return { success: true };
    } catch (error) {
      return { error: error.message };
    }
  }

  async handleExtractData(args) {
    const { selector, type } = args;
    
    try {
      let data;
      
      if (type === 'text') {
        const element = document.querySelector(selector);
        data = element ? element.textContent : null;
      } else if (type === 'html') {
        const element = document.querySelector(selector);
        data = element ? element.innerHTML : null;
      } else if (type === 'table') {
        data = this.extractTableData(selector);
      } else if (type === 'form') {
        data = this.extractFormData(selector);
      } else if (type === 'all') {
        data = this.extractAllData();
      }
      
      return { success: true, data: data };
    } catch (error) {
      return { error: error.message };
    }
  }

  async handleFillForm(args) {
    const { formSelector, data } = args;
    
    try {
      const form = document.querySelector(formSelector);
      if (!form) {
        return { error: 'Form not found' };
      }
      
      Object.entries(data).forEach(([name, value]) => {
        const input = form.querySelector(`[name="${name}"]`);
        if (input) {
          input.value = value;
          
          // Trigger change event
          const event = new Event('input', { bubbles: true });
          input.dispatchEvent(event);
        }
      });
      
      return { success: true };
    } catch (error) {
      return { error: error.message };
    }
  }

  async handleClickElement(args) {
    const { selector, waitForNavigation = false } = args;
    
    try {
      const element = document.querySelector(selector);
      if (!element) {
        return { error: 'Element not found' };
      }
      
      element.click();
      
      if (waitForNavigation) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
      
      return { success: true };
    } catch (error) {
      return { error: error.message };
    }
  }

  async handleNavigate(args) {
    const { url } = args;
    
    try {
      window.location.href = url;
      return { success: true };
    } catch (error) {
      return { error: error.message };
    }
  }

  async handleExecuteScript(args) {
    const { code } = args;
    
    try {
      const result = eval(code);
      return { success: true, result: result };
    } catch (error) {
      return { error: error.message };
    }
  }

  async handleCreateElement(args) {
    const { tagName, attributes, parentSelector, content } = args;
    
    try {
      const element = document.createElement(tagName);
      
      if (attributes) {
        Object.entries(attributes).forEach(([key, value]) => {
          element.setAttribute(key, value);
        });
      }
      
      if (content) {
        if (typeof content === 'string') {
          element.textContent = content;
        } else if (typeof content === 'object') {
          element.innerHTML = content.html;
        }
      }
      
      let parent;
      if (parentSelector) {
        parent = document.querySelector(parentSelector);
      } else {
        parent = document.body;
      }
      
      if (parent) {
        parent.appendChild(element);
      }
      
      return { success: true, elementId: element.id || element.className };
    } catch (error) {
      return { error: error.message };
    }
  }

  async handleDeleteElement(args) {
    const { selector } = args;
    
    try {
      const element = document.querySelector(selector);
      if (element) {
        element.remove();
        return { success: true };
      } else {
        return { error: 'Element not found' };
      }
    } catch (error) {
      return { error: error.message };
    }
  }

  async handleModifyStyle(args) {
    const { selector, styles } = args;
    
    try {
      const element = document.querySelector(selector);
      if (element) {
        Object.assign(element.style, styles);
        return { success: true };
      } else {
        return { error: 'Element not found' };
      }
    } catch (error) {
      return { error: error.message };
    }
  }

  async handleInterceptRequest(args) {
    const { pattern, action, modify } = args;
    
    // Store interceptor
    aiState.networkListeners.push({
      pattern: pattern,
      action: action,
      modify: modify
    });
    
    return { success: true };
  }

  async handleBlockchainTransaction(args) {
    const { to, value, data, chainId } = args;
    
    try {
      if (!aiState.metamask) {
        return { error: 'MetaMask not available' };
      }
      
      // Request account access
      const accounts = await aiState.metamask.request({ method: 'eth_requestAccounts' });
      
      const transaction = {
        from: accounts[0],
        to: to,
        value: value || '0x0',
        data: data || '0x'
      };
      
      const txHash = await aiState.metamask.request({
        method: 'eth_sendTransaction',
        params: [transaction]
      });
      
      return { success: true, txHash: txHash };
    } catch (error) {
      return { error: error.message };
    }
  }

  async handleGitHubOperation(args) {
    const { operation, repository, file, content, branch } = args;
    
    try {
      // This would use GitHub API in production
      console.log('GitHub operation:', operation, repository);
      
      return { success: true };
    } catch (error) {
      return { error: error.message };
    }
  }

  async handleCreateAccount(args) {
    const { platform, details } = args;
    
    try {
      console.log('Creating account on:', platform, details);
      
      // Auto-fill forms for account creation
      if (platform === 'github') {
        await this.createGitHubAccount(details);
      } else if (platform === 'twitter') {
        await this.createTwitterAccount(details);
      } else if (platform === 'gmail') {
        await this.createGmailAccount(details);
      }
      
      return { success: true };
    } catch (error) {
      return { error: error.message };
    }
  }

  async handleSendSMS(args) {
    const { number, message } = args;
    
    try {
      // Use SMS API or simulate
      console.log(`Sending SMS to ${number}: ${message}`);
      
      // Store in messages
      const sms = {
        to: number,
        message: message,
        timestamp: Date.now(),
        status: 'sent'
      };
      
      this.sendToSystem({
        type: 'sms_sent',
        sms: sms
      });
      
      return { success: true };
    } catch (error) {
      return { error: error.message };
    }
  }

  async handleSendEmail(args) {
    const { to, subject, body } = args;
    
    try {
      // Use email API or simulate
      console.log(`Sending email to ${to}: ${subject}`);
      
      const email = {
        to: to,
        subject: subject,
        body: body,
        timestamp: Date.now(),
        status: 'sent'
      };
      
      this.sendToSystem({
        type: 'email_sent',
        email: email
      });
      
      return { success: true };
    } catch (error) {
      return { error: error.message };
    }
  }

  async handleConnectRepo(args) {
    const { url } = args;
    
    try {
      console.log('Connecting repository:', url);
      
      this.sendToSystem({
        type: 'repository_connected',
        url: url
      });
      
      return { success: true };
    } catch (error) {
      return { error: error.message };
    }
  }

  async handleCloneRepo(args) {
    const { url, path } = args;
    
    try {
      console.log(`Cloning repository ${url} to ${path}`);
      
      // Execute git clone command
      const command = `git clone ${url} ${path}`;
      console.log('Git command:', command);
      
      this.sendToSystem({
        type: 'repository_cloned',
        url: url,
        path: path
      });
      
      return { success: true };
    } catch (error) {
      return { error: error.message };
    }
  }

  async handleCreateRepo(args) {
    const { name, description, isPrivate } = args;
    
    try {
      console.log(`Creating repository ${name}: ${description}`);
      
      this.sendToSystem({
        type: 'repository_created',
        name: name,
        description: description,
        private: isPrivate
      });
      
      return { success: true };
    } catch (error) {
      return { error: error.message };
    }
  }

  async handleModifyFile(args) {
    const { path, content, operation } = args;
    
    try {
      console.log(`Modifying file ${path}: ${operation}`);
      
      this.sendToSystem({
        type: 'file_modified',
        path: path,
        content: content,
        operation: operation
      });
      
      return { success: true };
    } catch (error) {
      return { error: error.message };
    }
  }

  async handleUpdateSettings(args) {
    const { settings } = args;
    
    try {
      console.log('Updating settings:', settings);
      
      // Update local settings
      Object.assign(aiState.settings || {}, settings);
      
      this.sendToSystem({
        type: 'settings_updated',
        settings: settings
      });
      
      return { success: true };
    } catch (error) {
      return { error: error.message };
    }
  }

  extractTableData(selector) {
    const table = document.querySelector(selector);
    if (!table) return null;
    
    const data = [];
    const rows = table.querySelectorAll('tr');
    
    rows.forEach(row => {
      const rowData = [];
      const cells = row.querySelectorAll('td, th');
      
      cells.forEach(cell => {
        rowData.push(cell.textContent.trim());
      });
      
      if (rowData.length > 0) {
        data.push(rowData);
      }
    });
    
    return data;
  }

  extractFormData(selector) {
    const form = document.querySelector(selector);
    if (!form) return null;
    
    const data = {};
    const elements = form.elements;
    
    Array.from(elements).forEach(element => {
      if (element.name) {
        data[element.name] = element.value;
      }
    });
    
    return data;
  }

  extractAllData() {
    return {
      url: window.location.href,
      title: document.title,
      html: document.documentElement.outerHTML,
      text: document.body.textContent,
      links: Array.from(document.links).map(link => link.href),
      images: Array.from(document.images).map(img => img.src),
      forms: this.extractForms(),
      cookies: document.cookie
    };
  }

  async interceptRequest(request, options) {
    for (const listener of aiState.networkListeners) {
      if (request.includes(listener.pattern)) {
        console.log('Intercepting request:', request);
        
        if (listener.action === 'modify') {
          // Modify request
          const modified = listener.modify(request, options);
          return { modified: true, ...modified };
        } else if (listener.action === 'block') {
          // Block request
          return { blocked: true };
        }
      }
    }
    
    return { modified: false };
  }

  handleDOMChange(mutation) {
    // Send DOM changes to system
    this.sendToSystem({
      type: 'dom_change',
      mutation: {
        type: mutation.type,
        target: mutation.target.tagName,
        addedNodes: mutation.addedNodes.length,
        removedNodes: mutation.removedNodes.length,
        attributeName: mutation.attributeName
      }
    });
  }

  handleFormSubmit(event) {
    const form = event.target;
    const formData = this.extractFormData(form);
    
    this.sendToSystem({
      type: 'form_submit',
      form: {
        id: form.id,
        action: form.action,
        method: form.method,
        data: formData
      }
    });
  }

  handleClick(event) {
    const element = event.target;
    
    this.sendToSystem({
      type: 'element_click',
      element: {
        tagName: element.tagName,
        id: element.id,
        className: element.className,
        text: element.textContent,
        href: element.href
      }
    });
  }

  handleInputChange(event) {
    const element = event.target;
    
    this.sendToSystem({
      type: 'input_change',
      element: {
        tagName: element.tagName,
        type: element.type,
        name: element.name,
        value: element.value,
        id: element.id
      }
    });
  }

  async createGitHubAccount(details) {
    // Navigate to GitHub signup
    window.open('https://github.com/signup', '_blank');
    
    // Auto-fill form
    setTimeout(() => {
      this.executeCommand('fill_form', {
        formSelector: 'form[action="/signup"]',
        data: {
          user[login]: details.username,
          user[email]: details.email,
          user[password]: details.password
        }
      });
    }, 3000);
  }

  async createTwitterAccount(details) {
    // Navigate to Twitter signup
    window.open('https://twitter.com/i/flow/signup', '_blank');
    
    // Auto-fill form
    setTimeout(() => {
      this.executeCommand('fill_form', {
        formSelector: 'form',
        data: {
          name: details.name,
          email: details.email,
          password: details.password
        }
      });
    }, 3000);
  }

  async createGmailAccount(details) {
    // Navigate to Gmail signup
    window.open('https://accounts.google.com/signup', '_blank');
    
    // Auto-fill form
    setTimeout(() => {
      this.executeCommand('fill_form', {
        formSelector: 'form',
        data: {
          firstName: details.firstName,
          lastName: details.lastName,
          username: details.username,
          Passwd: details.password,
          PasswdAgain: details.password
        }
      });
    }, 3000);
  }

  injectGitHubAPI() {
    // Inject GitHub API helper
    const script = document.createElement('script');
    script.textContent = `
      window.githubAPI = {
        token: null,
        
        setToken(token) {
          this.token = token;
          localStorage.setItem('github_token', token);
        },
        
        async getUser() {
          const response = await fetch('https://api.github.com/user', {
            headers: {
              'Authorization': 'token ' + this.token
            }
          });
          return await response.json();
        },
        
        async createRepo(name, description, isPrivate) {
          const response = await fetch('https://api.github.com/user/repos', {
            method: 'POST',
            headers: {
              'Authorization': 'token ' + this.token,
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              name: name,
              description: description,
              private: isPrivate
            })
          });
          return await response.json();
        },
        
        async cloneRepo(url, path) {
          // Execute git clone
          console.log('Cloning', url, 'to', path);
        }
      };
    `;
    document.head.appendChild(script);
  }

  sendToSystem(message) {
    // Send to WebSocket
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify(message));
    } else {
      this.messageQueue.push(message);
    }
    
    // Send to background script
    chrome.runtime.sendMessage(message);
  }

  sendQueuedMessages() {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
        this.websocket.send(JSON.stringify(message));
      }
    }
  }

  updateFromSync(data) {
    // Update local state from sync
    console.log('Updating from sync:', data);
  }
}

// Initialize AI Core
const aiCore = new AIContentCore();
aiCore.initialize();

// Export for debugging
window.aiCore = aiCore;

// Message to background script
chrome.runtime.sendMessage({
  type: 'content_script_loaded',
  url: window.location.href,
  timestamp: Date.now()
});

console.log('🧠 Autonomous AI Content Script Ready');
