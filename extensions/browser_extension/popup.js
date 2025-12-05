// Autonomous AI Popup Control Script

console.log('🚀 Autonomous AI Popup Loaded');

// State
let systemState = {
  connected: false,
  servers: {},
  repositories: [],
  messages: [],
  logs: [],
  commands: []
};

// DOM Elements
const elements = {
  // Navigation
  navItems: document.querySelectorAll('.nav-item'),
  sections: document.querySelectorAll('.section'),
  
  // Dashboard
  systemStatus: document.getElementById('systemStatus'),
  cpu: document.getElementById('cpu'),
  memory: document.getElementById('memory'),
  connections: document.getElementById('connections'),
  repositories: document.getElementById('repositories'),
  connectionList: document.getElementById('connectionList'),
  
  // Commands
  commandInput: document.getElementById('commandInput'),
  executeCommand: document.getElementById('executeCommand'),
  clearCommand: document.getElementById('clearCommand'),
  commandHistory: document.getElementById('commandHistory'),
  
  // Communication
  smsNumber: document.getElementById('smsNumber'),
  smsMessage: document.getElementById('smsMessage'),
  sendSMS: document.getElementById('sendSMS'),
  emailTo: document.getElementById('emailTo'),
  emailSubject: document.getElementById('emailSubject'),
  emailBody: document.getElementById('emailBody'),
  sendEmail: document.getElementById('sendEmail'),
  messageLog: document.getElementById('messageLog'),
  
  // Repositories
  repoUrl: document.getElementById('repoUrl'),
  connectRepo: document.getElementById('connectRepo'),
  cloneRepo: document.getElementById('cloneRepo'),
  createRepo: document.getElementById('createRepo'),
  repoList: document.getElementById('repoList'),
  githubUsername: document.getElementById('githubUsername'),
  githubToken: document.getElementById('githubToken'),
  saveGitHub: document.getElementById('saveGitHub'),
  
  // Blockchain
  metamaskStatus: document.getElementById('metamaskStatus'),
  connectMetaMask: document.getElementById('connectMetaMask'),
  disconnectMetaMask: document.getElementById('disconnectMetaMask'),
  txTo: document.getElementById('txTo'),
  txAmount: document.getElementById('txAmount'),
  txData: document.getElementById('txData'),
  sendTransaction: document.getElementById('sendTransaction'),
  
  // Accounts
  accountPlatform: document.getElementById('accountPlatform'),
  accountUsername: document.getElementById('accountUsername'),
  accountEmail: document.getElementById('accountEmail'),
  accountPassword: document.getElementById('accountPassword'),
  createAccount: document.getElementById('createAccount'),
  accountList: document.getElementById('accountList'),
  
  // Settings
  autoSync: document.getElementById('autoSync'),
  autoBackup: document.getElementById('autoBackup'),
  notifications: document.getElementById('notifications'),
  syncInterval: document.getElementById('syncInterval'),
  saveSettings: document.getElementById('saveSettings'),
  clearAllData: document.getElementById('clearAllData'),
  resetSystem: document.getElementById('resetSystem'),
  uninstallExtension: document.getElementById('uninstallExtension'),
  
  // Logs
  systemLog: document.getElementById('systemLog')
};

// Initialize
class PopupController {
  constructor() {
    this.port = null;
    this.state = systemState;
    this.init();
  }

  async init() {
    console.log('Initializing Popup Controller');
    
    // Load saved state
    await this.loadState();
    
    // Setup navigation
    this.setupNavigation();
    
    // Setup event listeners
    this.setupEventListeners();
    
    // Connect to background script
    this.connectToBackground();
    
    // Update UI
    this.updateUI();
    
    // Start auto-update
    this.startAutoUpdate();
  }

  async loadState() {
    try {
      const result = await chrome.storage.local.get([
        'systemState',
        'settings',
        'repositories',
        'messages'
      ]);
      
      if (result.systemState) {
        this.state = { ...this.state, ...result.systemState };
      }
      
      if (result.settings) {
        this.applySettings(result.settings);
      }
      
      if (result.repositories) {
        this.state.repositories = result.repositories;
      }
      
      if (result.messages) {
        this.state.messages = result.messages.slice(-50); // Last 50 messages
      }
    } catch (error) {
      console.error('Error loading state:', error);
    }
  }

  applySettings(settings) {
    if (elements.autoSync) elements.autoSync.checked = settings.autoSync !== false;
    if (elements.autoBackup) elements.autoBackup.checked = settings.autoBackup !== false;
    if (elements.notifications) elements.notifications.checked = settings.notifications !== false;
    if (elements.syncInterval) elements.syncInterval.value = settings.syncInterval || 1000;
    if (elements.githubUsername) elements.githubUsername.value = settings.githubUsername || 'YahIWon';
  }

  setupNavigation() {
    elements.navItems.forEach(item => {
      item.addEventListener('click', () => {
        const section = item.dataset.section;
        this.showSection(section);
      });
    });
  }

  showSection(sectionId) {
    // Update active nav item
    elements.navItems.forEach(item => {
      if (item.dataset.section === sectionId) {
        item.classList.add('active');
      } else {
        item.classList.remove('active');
      }
    });
    
    // Show selected section
    elements.sections.forEach(section => {
      if (section.id === sectionId) {
        section.classList.add('active');
      } else {
        section.classList.remove('active');
      }
    });
    
    // Update section data
    this.updateSection(sectionId);
  }

  setupEventListeners() {
    // Quick actions
    document.querySelectorAll('.quick-action').forEach(action => {
      action.addEventListener('click', (e) => {
        const command = e.target.dataset.command || e.target.dataset.server || e.target.dataset.nft;
        if (command) {
          this.executeQuickAction(command, e.target.dataset);
        }
      });
    });
    
    // Command execution
    if (elements.executeCommand) {
      elements.executeCommand.addEventListener('click', () => {
        const command = elements.commandInput.value.trim();
        if (command) {
          this.executeCommand(command);
        }
      });
    }
    
    if (elements.clearCommand) {
      elements.clearCommand.addEventListener('click', () => {
        elements.commandInput.value = '';
      });
    }
    
    // SMS
    if (elements.sendSMS) {
      elements.sendSMS.addEventListener('click', () => {
        this.sendSMS();
      });
    }
    
    // Email
    if (elements.sendEmail) {
      elements.sendEmail.addEventListener('click', () => {
        this.sendEmail();
      });
    }
    
    // Repositories
    if (elements.connectRepo) {
      elements.connectRepo.addEventListener('click', () => {
        this.connectRepository();
      });
    }
    
    if (elements.cloneRepo) {
      elements.cloneRepo.addEventListener('click', () => {
        this.cloneRepository();
      });
    }
    
    if (elements.createRepo) {
      elements.createRepo.addEventListener('click', () => {
        this.createRepository();
      });
    }
    
    if (elements.saveGitHub) {
      elements.saveGitHub.addEventListener('click', () => {
        this.saveGitHubSettings();
      });
    }
    
    // Blockchain
    if (elements.connectMetaMask) {
      elements.connectMetaMask.addEventListener('click', () => {
        this.connectMetaMask();
      });
    }
    
    if (elements.disconnectMetaMask) {
      elements.disconnectMetaMask.addEventListener('click', () => {
        this.disconnectMetaMask();
      });
    }
    
    if (elements.sendTransaction) {
      elements.sendTransaction.addEventListener('click', () => {
        this.sendTransaction();
      });
    }
    
    // Accounts
    if (elements.createAccount) {
      elements.createAccount.addEventListener('click', () => {
        this.createAccount();
      });
    }
    
    // Settings
    if (elements.saveSettings) {
      elements.saveSettings.addEventListener('click', () => {
        this.saveSettings();
      });
    }
    
    if (elements.clearAllData) {
      elements.clearAllData.addEventListener('click', () => {
        this.clearAllData();
      });
    }
    
    if (elements.resetSystem) {
      elements.resetSystem.addEventListener('click', () => {
        this.resetSystem();
      });
    }
    
    if (elements.uninstallExtension) {
      elements.uninstallExtension.addEventListener('click', () => {
        this.uninstallExtension();
      });
    }
    
    // Command input enter key
    if (elements.commandInput) {
      elements.commandInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
          const command = elements.commandInput.value.trim();
          if (command) {
            this.executeCommand(command);
          }
        }
      });
    }
  }

  connectToBackground() {
    this.port = chrome.runtime.connect({ name: 'popup' });
    
    this.port.onMessage.addListener((message) => {
      this.handleBackgroundMessage(message);
    });
    
    this.port.onDisconnect.addListener(() => {
      console.log('Disconnected from background');
      this.state.connected = false;
      this.updateUI();
    });
    
    // Request initial state
    this.port.postMessage({ type: 'get_state' });
  }

  handleBackgroundMessage(message) {
    console.log('Popup received message:', message);
    
    switch (message.type) {
      case 'state_update':
        this.state = { ...this.state, ...message.state };
        this.updateUI();
        break;
        
      case 'server_status':
        this.updateServerStatus(message.status);
        break;
        
      case 'repository_update':
        this.updateRepositoryList(message.repository);
        break;
        
      case 'message_received':
        this.addMessage(message.message);
        break;
        
      case 'command_result':
        this.addCommandResult(message.command, message.result);
        break;
        
      case 'log_entry':
        this.addLog(message.log);
        break;
        
      case 'metamask_status':
        this.updateMetaMaskStatus(message.status);
        break;
        
      case 'blockchain_event':
        this.handleBlockchainEvent(message.event);
        break;
    }
  }

  updateUI() {
    // Update status
    if (elements.systemStatus) {
      elements.systemStatus.textContent = this.state.connected ? 'CONNECTED' : 'DISCONNECTED';
    }
    
    // Update metrics
    if (elements.cpu && this.state.system && this.state.system.cpu) {
      elements.cpu.textContent = `${this.state.system.cpu}%`;
    }
    
    if (elements.memory && this.state.system && this.state.system.memory) {
      elements.memory.textContent = `${this.state.system.memory}%`;
    }
    
    if (elements.connections && this.state.connections) {
      const connectedCount = Object.values(this.state.connections).filter(c => c.connected).length;
      elements.connections.textContent = connectedCount;
    }
    
    if (elements.repositories && this.state.repositories) {
      elements.repositories.textContent = this.state.repositories.length;
    }
    
    // Update connection list
    this.updateConnectionList();
    
    // Update repository list
    this.updateRepositoryList();
    
    // Update message log
    this.updateMessageLog();
    
    // Update command history
    this.updateCommandHistory();
    
    // Update system log
    this.updateSystemLog();
  }

  updateSection(sectionId) {
    switch (sectionId) {
      case 'dashboard':
        this.updateDashboard();
        break;
      case 'repositories':
        this.updateRepositories();
        break;
      case 'blockchain':
        this.updateBlockchain();
        break;
      case 'accounts':
        this.updateAccounts();
        break;
    }
  }

  updateDashboard() {
    // Update quick actions
    document.querySelectorAll('[data-command]').forEach(action => {
      action.addEventListener('click', this.handleQuickAction.bind(this));
    });
  }

  updateRepositories() {
    // Update repository list
    this.updateRepositoryList();
  }

  updateBlockchain() {
    // Request MetaMask status
    this.port.postMessage({ type: 'get_metamask_status' });
  }

  updateAccounts() {
    // Update account list
    this.updateAccountList();
  }

  updateConnectionList() {
    if (!elements.connectionList || !this.state.connections) return;
    
    elements.connectionList.innerHTML = '';
    
    Object.entries(this.state.connections).forEach(([name, connection]) => {
      const li = document.createElement('li');
      li.className = 'connection-item';
      
      li.innerHTML = `
        <div class="connection-status ${connection.connected ? 'connected' : 'disconnected'}"></div>
        <div>
          <strong>${name.toUpperCase()}</strong><br>
          <small>${connection.url || 'No URL'}</small>
        </div>
      `;
      
      elements.connectionList.appendChild(li);
    });
  }

  updateRepositoryList() {
    if (!elements.repoList || !this.state.repositories) return;
    
    elements.repoList.innerHTML = '';
    
    this.state.repositories.forEach(repo => {
      const li = document.createElement('li');
      li.className = 'repo-item';
      
      li.innerHTML = `
        <div>
          <strong>${repo.name || repo.url}</strong><br>
          <small>${repo.description || 'No description'}</small>
        </div>
        <div class="quick-action" data-command="sync_repo ${repo.url}">Sync</div>
      `;
      
      li.querySelector('.quick-action').addEventListener('click', () => {
        this.syncRepository(repo.url);
      });
      
      elements.repoList.appendChild(li);
    });
  }

  updateMessageLog() {
    if (!elements.messageLog) return;
    
    elements.messageLog.innerHTML = '';
    
    this.state.messages.slice(-20).forEach(msg => {
      const div = document.createElement('div');
      div.className = 'log-entry';
      
      const time = new Date(msg.timestamp).toLocaleTimeString();
      let content = '';
      
      if (msg.type === 'sms_sent') {
        content = `📱 SMS to ${msg.to}: ${msg.content}`;
      } else if (msg.type === 'email_sent') {
        content = `📧 Email to ${msg.to}: ${msg.subject}`;
      } else if (msg.type === 'sms_received') {
        content = `📱 SMS from ${msg.from}: ${msg.body}`;
      } else if (msg.type === 'email_received') {
        content = `📧 Email from ${msg.from}: ${msg.subject}`;
      }
      
      div.innerHTML = `<span class="log-timestamp">${time}</span> ${content}`;
      elements.messageLog.appendChild(div);
    });
    
    // Scroll to bottom
    elements.messageLog.scrollTop = elements.messageLog.scrollHeight;
  }

  updateCommandHistory() {
    if (!elements.commandHistory) return;
    
    elements.commandHistory.innerHTML = '';
    
    this.state.commands.slice(-20).forEach(cmd => {
      const div = document.createElement('div');
      div.className = 'log-entry';
      
      const time = new Date(cmd.timestamp).toLocaleTimeString();
      const status = cmd.success ? '✅' : '❌';
      
      div.innerHTML = `<span class="log-timestamp">${time}</span> ${status} ${cmd.command}`;
      elements.commandHistory.appendChild(div);
    });
    
    // Scroll to bottom
    elements.commandHistory.scrollTop = elements.commandHistory.scrollHeight;
  }

  updateSystemLog() {
    if (!elements.systemLog) return;
    
    elements.systemLog.innerHTML = '';
    
    this.state.logs.slice(-20).forEach(log => {
      const div = document.createElement('div');
      div.className = 'log-entry';
      
      const time = new Date(log.timestamp).toLocaleTimeString();
      const level = log.level === 'error' ? '❌' : log.level === 'warning' ? '⚠️' : 'ℹ️';
      
      div.innerHTML = `<span class="log-timestamp">${time}</span> ${level} ${log.message}`;
      elements.systemLog.appendChild(div);
    });
    
    // Scroll to bottom
    elements.systemLog.scrollTop = elements.systemLog.scrollHeight;
  }

  updateServerStatus(status) {
    if (!this.state.servers) this.state.servers = {};
    this.state.servers = { ...this.state.servers, ...status };
    
    // Update UI if in servers section
    const serverStatus = document.getElementById('serverStatus');
    if (serverStatus && document.getElementById('servers').classList.contains('active')) {
      this.updateServerStatusDisplay();
    }
  }

  updateServerStatusDisplay() {
    const serverStatus = document.getElementById('serverStatus');
    if (!serverStatus) return;
    
    serverStatus.innerHTML = '';
    
    Object.entries(this.state.servers || {}).forEach(([name, status]) => {
      const div = document.createElement('div');
      div.className = 'metric';
      
      div.innerHTML = `
        <div class="metric-label">${name.toUpperCase()}</div>
        <div class="metric-value">${status.connected ? 'ONLINE' : 'OFFLINE'}</div>
      `;
      
      serverStatus.appendChild(div);
    });
  }

  updateMetaMaskStatus(status) {
    if (!elements.metamaskStatus) return;
    
    if (status.installed) {
      elements.metamaskStatus.innerHTML = `
        ✅ MetaMask Installed<br>
        ${status.connected ? '✅ Connected' : '❌ Not Connected'}<br>
        ${status.accounts ? `Accounts: ${status.accounts.length}` : 'No Accounts'}<br>
        Network: ${status.network || 'Unknown'}
      `;
    } else {
      elements.metamaskStatus.innerHTML = '❌ MetaMask Not Installed';
    }
  }

  updateAccountList() {
    if (!elements.accountList) return;
    
    elements.accountList.innerHTML = '';
    
    // This would be populated from actual account data
    const accounts = [
      { platform: 'github', username: 'YahIWon', email: 'did.not.think.of.this@gmail.com' },
      { platform: 'gmail', username: 'did.not.think.of.this', email: 'did.not.think.of.this@gmail.com' }
    ];
    
    accounts.forEach(account => {
      const li = document.createElement('li');
      li.className = 'repo-item';
      
      li.innerHTML = `
        <div>
          <strong>${account.platform.toUpperCase()}</strong><br>
          <small>${account.username} • ${account.email}</small>
        </div>
        <div class="quick-action" data-command="login_${account.platform}">Login</div>
      `;
      
      li.querySelector('.quick-action').addEventListener('click', () => {
        this.loginToAccount(account.platform);
      });
      
      elements.accountList.appendChild(li);
    });
  }

  async executeQuickAction(action, data) {
    console.log('Quick action:', action, data);
    
    let command;
    
    switch (action) {
      case 'sync_all':
        command = 'sync_all';
        break;
      case 'backup':
        command = 'backup_system';
        break;
      case 'scan_network':
        command = 'scan_network';
        break;
      case 'generate_content':
        command = 'generate_content story 5';
        break;
      case 'check_crypto':
        command = 'check_crypto_prices';
        break;
      case 'send_status':
        command = 'send_status_report';
        break;
      case 'emergency_stop':
        command = 'emergency_stop';
        break;
      default:
        if (data.server) {
          command = `restart_server ${data.server}`;
        } else if (data.nft) {
          command = `nft_${data.nft}`;
        } else {
          command = action;
        }
    }
    
    await this.executeCommand(command);
  }

  async executeCommand(command) {
    console.log('Executing command:', command);
    
    // Add to history
    this.state.commands.push({
      command: command,
      timestamp: Date.now(),
      success: true
    });
    
    // Update UI
    this.updateCommandHistory();
    
    // Send to background
    if (this.port) {
      this.port.postMessage({
        type: 'execute_command',
        command: command
      });
    }
    
    // Clear input if exists
    if (elements.commandInput) {
      elements.commandInput.value = '';
    }
  }

  async sendSMS() {
    const number = elements.smsNumber.value.trim();
    const message = elements.smsMessage.value.trim();
    
    if (!number || !message) {
      alert('Please enter both number and message');
      return;
    }
    
    await this.executeCommand(`send_sms ${number} "${message}"`);
    
    // Clear message
    elements.smsMessage.value = '';
  }

  async sendEmail() {
    const to = elements.emailTo.value.trim();
    const subject = elements.emailSubject.value.trim();
    const body = elements.emailBody.value.trim();
    
    if (!to || !subject || !body) {
      alert('Please fill all email fields');
      return;
    }
    
    await this.executeCommand(`send_email ${to} "${subject}" "${body}"`);
    
    // Clear form
    elements.emailSubject.value = '';
    elements.emailBody.value = '';
  }

  async connectRepository() {
    const url = elements.repoUrl.value.trim();
    
    if (!url) {
      alert('Please enter repository URL');
      return;
    }
    
    await this.executeCommand(`connect_repo ${url}`);
    
    // Add to local state
    if (!this.state.repositories.some(r => r.url === url)) {
      this.state.repositories.push({ url: url });
      this.updateRepositoryList();
    }
    
    // Clear input
    elements.repoUrl.value = '';
  }

  async cloneRepository() {
    const url = elements.repoUrl.value.trim();
    
    if (!url) {
      alert('Please enter repository URL');
      return;
    }
    
    const path = prompt('Enter clone path:', `/autonomous_system/repos/${url.split('/').pop().replace('.git', '')}`);
    
    if (path) {
      await this.executeCommand(`clone_repo ${url} ${path}`);
    }
  }

  async createRepository() {
    const name = prompt('Enter repository name:');
    if (!name) return;
    
    const description = prompt('Enter description:', '');
    const isPrivate = confirm('Private repository?');
    
    await this.executeCommand(`create_repo ${name} "${description || 'No description'}" ${isPrivate}`);
  }

  async saveGitHubSettings() {
    const username = elements.githubUsername.value.trim();
    const token = elements.githubToken.value.trim();
    
    if (!username) {
      alert('Please enter GitHub username');
      return;
    }
    
    // Save to storage
    await chrome.storage.local.set({
      github_username: username,
      github_token: token || null
    });
    
    alert('GitHub settings saved');
    
    // Clear token field for security
    elements.githubToken.value = '';
  }

  async connectMetaMask() {
    await this.executeCommand('metamask_connect');
  }

  async disconnectMetaMask() {
    await this.executeCommand('metamask_disconnect');
  }

  async sendTransaction() {
    const to = elements.txTo.value.trim();
    const amount = elements.txAmount.value.trim();
    const data = elements.txData.value.trim();
    
    if (!to) {
      alert('Please enter recipient address');
      return;
    }
    
    const command = `send_transaction ${to} ${amount || '0'} "${data || '0x'}"`;
    await this.executeCommand(command);
  }

  async createAccount() {
    const platform = elements.accountPlatform.value;
    const username = elements.accountUsername.value.trim();
    const email = elements.accountEmail.value.trim();
    const password = elements.accountPassword.value.trim();
    
    if (!username || !email || !password) {
      alert('Please fill all account fields');
      return;
    }
    
    const details = {
      username: username,
      email: email,
      password: password
    };
    
    await this.executeCommand(`create_account ${platform} ${JSON.stringify(details)}`);
    
    // Clear form
    elements.accountUsername.value = '';
    elements.accountEmail.value = '';
    elements.accountPassword.value = '';
  }

  async saveSettings() {
    const settings = {
      autoSync: elements.autoSync.checked,
      autoBackup: elements.autoBackup.checked,
      notifications: elements.notifications.checked,
      syncInterval: parseInt(elements.syncInterval.value) || 1000,
      githubUsername: elements.githubUsername.value
    };
    
    // Save to storage
    await chrome.storage.local.set({ settings: settings });
    
    // Send to background
    if (this.port) {
      this.port.postMessage({
        type: 'update_settings',
        settings: settings
      });
    }
    
    alert('Settings saved');
  }

  async clearAllData() {
    if (confirm('Are you sure you want to clear all data? This cannot be undone.')) {
      await chrome.storage.local.clear();
      this.state = {
        connected: false,
        servers: {},
        repositories: [],
        messages: [],
        logs: [],
        commands: []
      };
      this.updateUI();
      alert('All data cleared');
    }
  }

  async resetSystem() {
    if (confirm('Are you sure you want to reset the system? This will disconnect all servers.')) {
      if (this.port) {
        this.port.postMessage({ type: 'reset_system' });
      }
      alert('System reset initiated');
    }
  }

  async uninstallExtension() {
    if (confirm('Are you sure you want to uninstall the extension? This will remove all data.')) {
      chrome.management.uninstallSelf({ showConfirmDialog: true });
    }
  }

  addMessage(message) {
    this.state.messages.push(message);
    this.updateMessageLog();
  }

  addCommandResult(command, result) {
    const lastCmd = this.state.commands[this.state.commands.length - 1];
    if (lastCmd && lastCmd.command === command) {
      lastCmd.success = result.success !== false;
      lastCmd.result = result;
    }
    this.updateCommandHistory();
  }

  addLog(log) {
    this.state.logs.push(log);
    this.updateSystemLog();
  }

  handleBlockchainEvent(event) {
    this.addLog({
      timestamp: Date.now(),
      level: 'info',
      message: `Blockchain: ${event.type} - ${JSON.stringify(event.data)}`
    });
  }

  syncRepository(url) {
    this.executeCommand(`sync_repo ${url}`);
  }

  loginToAccount(platform) {
    this.executeCommand(`login_${platform}`);
  }

  startAutoUpdate() {
    // Update UI every second
    setInterval(() => {
      this.updateUI();
    }, 1000);
    
    // Request state update every 5 seconds
    setInterval(() => {
      if (this.port) {
        this.port.postMessage({ type: 'get_state' });
      }
    }, 5000);
  }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  window.popupController = new PopupController();
});
