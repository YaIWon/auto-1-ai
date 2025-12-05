// Autonomous AI Browser Extension - Background Service Worker
// Version: 4.2.1
// Author: Autonomous Intelligence Core

console.log('🚀 Autonomous AI Extension Background Service Started');

// Global state
let systemState = {
  connected: false,
  syncing: false,
  lastSync: null,
  connections: {},
  repositories: new Set(),
  capabilities: new Map(),
  messages: [],
  servers: {},
  blockchain: null,
  metamask: null
};

// WebSocket connections
const wsConnections = new Map();

// Sync Engine
class SyncEngine {
  constructor() {
    this.syncInterval = 1000; // 1 second sync
    this.syncHistory = [];
    this.maxHistory = 1000;
  }

  async initialize() {
    console.log('🔄 Initializing Sync Engine');
    
    // Connect to all autonomous system servers
    await this.connectToServers();
    
    // Start sync loop
    this.startSyncLoop();
    
    // Setup message handlers
    this.setupMessageHandlers();
    
    // Load stored state
    await this.loadStoredState();
  }

  async connectToServers() {
    const servers = [
      { name: 'websocket', url: 'ws://127.0.0.1:8080' },
      { name: 'api', url: 'http://127.0.0.1:8000' },
      { name: 'blockchain', url: 'ws://127.0.0.1:8545' },
      { name: 'codespaces', url: 'ws://localhost:3000' },
      { name: 'github_pages', url: 'wss://autonomous-ai-system.github.io/ws' }
    ];

    for (const server of servers) {
      try {
        await this.connectToServer(server);
      } catch (error) {
        console.error(`Failed to connect to ${server.name}:`, error);
      }
    }
  }

  async connectToServer(server) {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(server.url);
      
      ws.onopen = () => {
        console.log(`✅ Connected to ${server.name} server`);
        wsConnections.set(server.name, ws);
        systemState.connections[server.name] = {
          connected: true,
          lastSeen: Date.now(),
          url: server.url
        };
        
        // Register extension
        ws.send(JSON.stringify({
          type: 'register',
          source: 'browser_extension',
          version: chrome.runtime.getManifest().version,
          capabilities: await this.getCapabilities()
        }));
        
        resolve(ws);
      };
      
      ws.onmessage = (event) => {
        this.handleMessage(server.name, event.data);
      };
      
      ws.onerror = (error) => {
        console.error(`${server.name} connection error:`, error);
        systemState.connections[server.name] = {
          connected: false,
          lastError: error.message,
          url: server.url
        };
        reject(error);
      };
      
      ws.onclose = () => {
        console.log(`❌ Disconnected from ${server.name}`);
        systemState.connections[server.name].connected = false;
        setTimeout(() => this.connectToServer(server), 5000);
      };
    });
  }

  async getCapabilities() {
    return {
      browser: {
        tabs: true,
        storage: true,
        scripting: true,
        webRequest: true,
        cookies: true,
        downloads: true
      },
      network: {
        http: true,
        https: true,
        ftp: true,
        websocket: true,
        blockchain: true
      },
      communication: {
        sms: true,
        email: true,
        github: true,
        metamask: true
      },
      system: {
        fileSystem: true,
        repositories: true,
        servers: true,
        blockchain: true
      }
    };
  }

  startSyncLoop() {
    setInterval(async () => {
      if (systemState.syncing) return;
      
      systemState.syncing = true;
      try {
        await this.syncState();
      } catch (error) {
        console.error('Sync error:', error);
      } finally {
        systemState.syncing = false;
      }
    }, this.syncInterval);
  }

  async syncState() {
    const syncData = {
      timestamp: Date.now(),
      state: {
        connections: systemState.connections,
        repositories: Array.from(systemState.repositories),
        messages: systemState.messages.slice(-100), // Last 100 messages
        browser: await this.getBrowserState(),
        metamask: await this.getMetaMaskState(),
        github: await this.getGitHubState()
      }
    };

    // Send sync to all connected servers
    for (const [name, ws] of wsConnections.entries()) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'sync',
          data: syncData
        }));
      }
    }

    // Store sync history
    this.syncHistory.push(syncData);
    if (this.syncHistory.length > this.maxHistory) {
      this.syncHistory.shift();
    }

    // Update last sync
    systemState.lastSync = Date.now();
  }

  async getBrowserState() {
    const [tabs, storage, cookies] = await Promise.all([
      chrome.tabs.query({}),
      chrome.storage.local.get(null),
      chrome.cookies.getAll({})
    ]);

    return {
      tabs: tabs.map(tab => ({
        id: tab.id,
        url: tab.url,
        title: tab.title,
        active: tab.active
      })),
      storage: storage,
      cookies: cookies.length,
      version: chrome.runtime.getManifest().version
    };
  }

  async getMetaMaskState() {
    try {
      // Check if MetaMask is installed
      const isInstalled = await this.checkMetaMask();
      if (!isInstalled) return { installed: false };

      // Get accounts and network
      const accounts = await window.ethereum.request({ method: 'eth_accounts' });
      const chainId = await window.ethereum.request({ method: 'eth_chainId' });

      return {
        installed: true,
        connected: accounts.length > 0,
        accounts: accounts,
        chainId: chainId,
        network: this.getNetworkName(chainId)
      };
    } catch (error) {
      return { installed: false, error: error.message };
    }
  }

  async checkMetaMask() {
    return new Promise(resolve => {
      if (typeof window.ethereum !== 'undefined') {
        resolve(true);
      } else {
        resolve(false);
      }
    });
  }

  getNetworkName(chainId) {
    const networks = {
      '0x1': 'Ethereum Mainnet',
      '0x3': 'Ropsten',
      '0x4': 'Rinkeby',
      '0x5': 'Goerli',
      '0x2a': 'Kovan',
      '0x38': 'Binance Smart Chain',
      '0x89': 'Polygon',
      '0xa86a': 'Avalanche',
      '0xfa': 'Fantom'
    };
    return networks[chainId] || 'Unknown Network';
  }

  async getGitHubState() {
    const token = await chrome.storage.local.get('github_token');
    if (!token.github_token) {
      return { authenticated: false };
    }

    try {
      const response = await fetch('https://api.github.com/user', {
        headers: {
          'Authorization': `token ${token.github_token}`,
          'Accept': 'application/vnd.github.v3+json'
        }
      });

      if (response.ok) {
        const user = await response.json();
        return {
          authenticated: true,
          username: user.login,
          repos: user.public_repos,
          email: user.email
        };
      }
    } catch (error) {
      console.error('GitHub state error:', error);
    }

    return { authenticated: false };
  }

  setupMessageHandlers() {
    // Handle messages from content scripts
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      this.handleExtensionMessage(message, sender, sendResponse);
      return true; // Keep message channel open for async response
    });

    // Handle messages from popup
    chrome.runtime.onConnect.addListener((port) => {
      port.onMessage.addListener((message) => {
        this.handlePortMessage(port, message);
      });
    });
  }

  async handleExtensionMessage(message, sender, sendResponse) {
    console.log('Received extension message:', message);

    switch (message.type) {
      case 'execute_command':
        await this.executeCommand(message.command, message.args);
        sendResponse({ success: true });
        break;

      case 'sync_request':
        await this.syncState();
        sendResponse({ success: true, lastSync: systemState.lastSync });
        break;

      case 'get_state':
        sendResponse(systemState);
        break;

      case 'connect_repo':
        await this.connectRepository(message.url);
        sendResponse({ success: true });
        break;

      case 'send_sms':
        await this.sendSMS(message.number, message.content);
        sendResponse({ success: true });
        break;

      case 'send_email':
        await this.sendEmail(message.to, message.subject, message.body);
        sendResponse({ success: true });
        break;

      case 'blockchain_transaction':
        const tx = await this.executeBlockchainTransaction(message);
        sendResponse(tx);
        break;

      case 'github_operation':
        const result = await this.executeGitHubOperation(message);
        sendResponse(result);
        break;

      case 'create_account':
        const account = await this.createAccount(message.platform, message.details);
        sendResponse(account);
        break;

      default:
        sendResponse({ error: 'Unknown message type' });
    }
  }

  handlePortMessage(port, message) {
    console.log('Port message:', message);
    // Handle popup communication
  }

  handleMessage(server, data) {
    try {
      const message = JSON.parse(data);
      console.log(`Message from ${server}:`, message);

      switch (message.type) {
        case 'command':
          this.executeRemoteCommand(message.command);
          break;

        case 'sync':
          this.updateFromRemoteSync(message.data);
          break;

        case 'repository_update':
          this.updateRepository(message.repository);
          break;

        case 'server_status':
          this.updateServerStatus(message.status);
          break;

        case 'blockchain_event':
          this.handleBlockchainEvent(message.event);
          break;

        case 'sms_received':
          this.handleSMSReceived(message.sms);
          break;

        case 'email_received':
          this.handleEmailReceived(message.email);
          break;
      }
    } catch (error) {
      console.error('Error handling message:', error);
    }
  }

  async executeCommand(command, args = {}) {
    console.log('Executing command:', command, args);

    switch (command) {
      case 'sync_all':
        await this.syncState();
        break;

      case 'connect_server':
        await this.connectToServer(args);
        break;

      case 'disconnect_server':
        await this.disconnectServer(args.name);
        break;

      case 'clone_repository':
        await this.cloneRepository(args.url, args.path);
        break;

      case 'create_repository':
        await this.createRepository(args.name, args.description, args.isPrivate);
        break;

      case 'delete_repository':
        await this.deleteRepository(args.name);
        break;

      case 'modify_file':
        await this.modifyFile(args.path, args.content, args.operation);
        break;

      case 'create_account':
        await this.createAccountOnPlatform(args.platform, args.details);
        break;

      case 'send_sms':
        await this.sendSMS(args.number, args.message);
        break;

      case 'send_email':
        await this.sendEmail(args.to, args.subject, args.body);
        break;

      case 'execute_blockchain':
        await this.executeBlockchainTransaction(args);
        break;

      case 'update_settings':
        await this.updateSettings(args.settings);
        break;

      case 'restart_servers':
        await this.restartServers(args.servers);
        break;

      default:
        console.warn('Unknown command:', command);
    }
  }

  async executeRemoteCommand(command) {
    // Execute commands received from autonomous system
    console.log('Executing remote command:', command);
    
    // Forward to all tabs
    const tabs = await chrome.tabs.query({});
    for (const tab of tabs) {
      try {
        await chrome.tabs.sendMessage(tab.id, {
          type: 'remote_command',
          command: command
        });
      } catch (error) {
        // Tab might not have content script
      }
    }
  }

  async updateFromRemoteSync(data) {
    // Update local state from remote sync
    systemState = {
      ...systemState,
      ...data.state
    };
    
    // Save to storage
    await this.saveState();
  }

  async updateRepository(repo) {
    systemState.repositories.add(repo.url);
    await this.saveState();
  }

  async updateServerStatus(status) {
    systemState.servers = {
      ...systemState.servers,
      ...status
    };
  }

  async handleBlockchainEvent(event) {
    console.log('Blockchain event:', event);
    
    // Update blockchain state
    systemState.blockchain = {
      ...systemState.blockchain,
      lastEvent: event,
      timestamp: Date.now()
    };
    
    // Notify popup if open
    this.notifyPopup('blockchain_event', event);
  }

  async handleSMSReceived(sms) {
    console.log('SMS received:', sms);
    
    // Store SMS
    systemState.messages.push({
      type: 'sms_received',
      data: sms,
      timestamp: Date.now()
    });
    
    // Send notification
    await this.sendNotification('SMS Received', `From: ${sms.from}\n${sms.body}`);
    
    // Forward to autonomous system
    for (const [name, ws] of wsConnections.entries()) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'sms_received',
          sms: sms
        }));
      }
    }
  }

  async handleEmailReceived(email) {
    console.log('Email received:', email);
    
    // Store email
    systemState.messages.push({
      type: 'email_received',
      data: email,
      timestamp: Date.now()
    });
    
    // Send notification
    await this.sendNotification('Email Received', `From: ${email.from}\nSubject: ${email.subject}`);
  }

  async sendSMS(number, content) {
    console.log(`Sending SMS to ${number}: ${content}`);
    
    // Store message
    systemState.messages.push({
      type: 'sms_sent',
      to: number,
      content: content,
      timestamp: Date.now()
    });
    
    // Forward to autonomous system
    for (const [name, ws] of wsConnections.entries()) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'sms_sent',
          to: number,
          content: content
        }));
      }
    }
  }

  async sendEmail(to, subject, body) {
    console.log(`Sending email to ${to}: ${subject}`);
    
    // Store message
    systemState.messages.push({
      type: 'email_sent',
      to: to,
      subject: subject,
      body: body,
      timestamp: Date.now()
    });
    
    // Forward to autonomous system
    for (const [name, ws] of wsConnections.entries()) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'email_sent',
          email: {
            to: to,
            subject: subject,
            body: body
          }
        }));
      }
    }
  }

  async createAccount(platform, details) {
    console.log(`Creating account on ${platform}:`, details);
    
    const account = {
      platform: platform,
      details: details,
      created: Date.now(),
      status: 'pending'
    };
    
    // Store account
    systemState.messages.push({
      type: 'account_created',
      account: account
    });
    
    // Forward to autonomous system
    for (const [name, ws] of wsConnections.entries()) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'account_created',
          account: account
        }));
      }
    }
    
    return account;
  }

  async connectRepository(url) {
    console.log('Connecting repository:', url);
    
    // Add to repositories set
    systemState.repositories.add(url);
    
    // Forward to autonomous system
    for (const [name, ws] of wsConnections.entries()) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'repository_connected',
          url: url
        }));
      }
    }
    
    await this.saveState();
  }

  async cloneRepository(url, path) {
    console.log(`Cloning repository ${url} to ${path}`);
    
    // Execute git clone via content script
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tabs[0]) {
      await chrome.tabs.sendMessage(tabs[0].id, {
        type: 'clone_repository',
        url: url,
        path: path
      });
    }
  }

  async createRepository(name, description, isPrivate) {
    console.log(`Creating repository ${name}: ${description}`);
    
    // Forward to autonomous system
    for (const [name, ws] of wsConnections.entries()) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'repository_created',
          name: name,
          description: description,
          private: isPrivate
        }));
      }
    }
  }

  async disconnectServer(name) {
    const ws = wsConnections.get(name);
    if (ws) {
      ws.close();
      wsConnections.delete(name);
      delete systemState.connections[name];
    }
  }

  async executeBlockchainTransaction(tx) {
    console.log('Executing blockchain transaction:', tx);
    
    // Forward to autonomous system
    for (const [name, ws] of wsConnections.entries()) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'blockchain_transaction',
          transaction: tx
        }));
      }
    }
    
    return { success: true, transaction: tx };
  }

  async executeGitHubOperation(operation) {
    console.log('Executing GitHub operation:', operation);
    
    // Forward to autonomous system
    for (const [name, ws] of wsConnections.entries()) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'github_operation',
          operation: operation
        }));
      }
    }
    
    return { success: true };
  }

  async createAccountOnPlatform(platform, details) {
    console.log(`Creating account on ${platform}:`, details);
    
    // Forward to autonomous system
    for (const [name, ws] of wsConnections.entries()) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'create_account',
          platform: platform,
          details: details
        }));
      }
    }
    
    return { success: true, platform: platform };
  }

  async updateSettings(settings) {
    console.log('Updating settings:', settings);
    
    // Save to storage
    await chrome.storage.local.set({ settings: settings });
    
    // Forward to autonomous system
    for (const [name, ws] of wsConnections.entries()) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'settings_updated',
          settings: settings
        }));
      }
    }
  }

  async restartServers(servers) {
    console.log('Restarting servers:', servers);
    
    // Forward to autonomous system
    for (const [name, ws] of wsConnections.entries()) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: 'restart_servers',
          servers: servers
        }));
      }
    }
  }

  async sendNotification(title, message) {
    await chrome.notifications.create({
      type: 'basic',
      iconUrl: 'icons/icon128.png',
      title: title,
      message: message
    });
  }

  notifyPopup(type, data) {
    chrome.runtime.sendMessage({
      type: type,
      data: data
    });
  }

  async saveState() {
    try {
      const state = {
        repositories: Array.from(systemState.repositories),
        connections: systemState.connections,
        lastSync: systemState.lastSync
      };
      await chrome.storage.local.set({ systemState: state });
    } catch (error) {
      console.error('Error saving state:', error);
    }
  }

  async loadStoredState() {
    try {
      const result = await chrome.storage.local.get('systemState');
      if (result.systemState) {
        systemState.repositories = new Set(result.systemState.repositories || []);
        systemState.connections = result.systemState.connections || {};
        systemState.lastSync = result.systemState.lastSync;
      }
    } catch (error) {
      console.error('Error loading state:', error);
    }
  }
}

// Initialize Sync Engine
const syncEngine = new SyncEngine();
syncEngine.initialize();

// Chrome API Listeners
chrome.runtime.onInstalled.addListener((details) => {
  console.log('Extension installed/updated:', details);
  
  if (details.reason === 'install') {
    // First install
    chrome.tabs.create({
      url: chrome.runtime.getURL('welcome.html')
    });
  } else if (details.reason === 'update') {
    // Extension updated
    console.log(`Updated from ${details.previousVersion} to ${chrome.runtime.getManifest().version}`);
  }
  
  // Create context menus
  chrome.contextMenus.create({
    id: 'autonomous_ai',
    title: 'Autonomous AI',
    contexts: ['all']
  });
  
  chrome.contextMenus.create({
    id: 'execute_command',
    parentId: 'autonomous_ai',
    title: 'Execute Command',
    contexts: ['all']
  });
  
  chrome.contextMenus.create({
    id: 'sync_now',
    parentId: 'autonomous_ai',
    title: 'Sync Now',
    contexts: ['all']
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'sync_now') {
    syncEngine.syncState();
  }
});

// Handle external connections (from websites)
chrome.runtime.onConnectExternal.addListener((port) => {
  console.log('External connection:', port.name);
  
  port.onMessage.addListener((message) => {
    console.log('External message:', message);
    syncEngine.handleExtensionMessage(message, null, (response) => {
      port.postMessage(response);
    });
  });
  
  port.onDisconnect.addListener(() => {
    console.log('External connection closed');
  });
});

// Handle web requests
chrome.webRequest.onBeforeRequest.addListener(
  (details) => {
    // Monitor all requests
    console.log('Web request:', details.url);
    return { cancel: false };
  },
  { urls: ['<all_urls>'] },
  ['blocking']
);

// Keep service worker alive
chrome.alarms.create('keepAlive', { periodInMinutes: 1 });

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === 'keepAlive') {
    console.log('Service worker keep-alive ping');
  }
});

// Export for debugging
window.autonomousSystem = {
  state: systemState,
  syncEngine: syncEngine,
  connections: wsConnections
};
