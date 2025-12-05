// Autonomous AI Injected Script
// Injected into web pages for direct control

(function() {
  'use strict';
  
  console.log('🧠 Autonomous AI Injected Script Loaded');
  
  // Create global AI object
  window.AutonomousAI = {
    version: '4.2.1',
    connected: false,
    capabilities: {},
    commands: new Map(),
    
    // Initialize
    async init() {
      console.log('🚀 Initializing Autonomous AI Injected Script');
      
      // Setup message bridge
      this.setupMessageBridge();
      
      // Register capabilities
      this.registerCapabilities();
      
      // Connect to extension
      await this.connectToExtension();
      
      // Start monitoring
      this.startMonitoring();
      
      this.connected = true;
      console.log('✅ Autonomous AI Injected Script Ready');
    },
    
    // Setup message bridge between page and extension
    setupMessageBridge() {
      // Listen for messages from the page
      window.addEventListener('message', (event) => {
        if (event.source !== window) return;
        
        if (event.data && event.data.type && event.data.type.startsWith('AUTONOMOUS_AI_')) {
          this.handlePageMessage(event.data);
        }
      });
      
      // Post messages to extension via DOM events
      this.postMessage = (message) => {
        window.postMessage({
          type: 'AUTONOMOUS_AI_FROM_PAGE',
          data: message
        }, '*');
        
        // Also send to extension via custom event
        const event = new CustomEvent('AutonomousAIMessage', {
          detail: message
        });
        window.dispatchEvent(event);
      };
    },
    
    // Register AI capabilities
    registerCapabilities() {
      // DOM manipulation
      this.capabilities.dom = {
        getElement: (selector) => document.querySelector(selector),
        getElements: (selector) => document.querySelectorAll(selector),
        createElement: (tag, attributes, content) => {
          const el = document.createElement(tag);
          if (attributes) Object.assign(el, attributes);
          if (content) el.innerHTML = content;
          return el;
        },
        removeElement: (selector) => {
          const el = document.querySelector(selector);
          if (el) el.remove();
        }
      };
      
      // Form manipulation
      this.capabilities.forms = {
        getFormData: (selector) => {
          const form = document.querySelector(selector);
          if (!form) return null;
          
          const data = {};
          Array.from(form.elements).forEach(el => {
            if (el.name) data[el.name] = el.value;
          });
          return data;
        },
        fillForm: (selector, data) => {
          const form = document.querySelector(selector);
          if (!form) return false;
          
          Object.entries(data).forEach(([name, value]) => {
            const input = form.querySelector(`[name="${name}"]`);
            if (input) {
              input.value = value;
              input.dispatchEvent(new Event('input', { bubbles: true }));
            }
          });
          return true;
        },
        submitForm: (selector) => {
          const form = document.querySelector(selector);
          if (form) form.submit();
        }
      };
      
      // Network interception
      this.capabilities.network = {
        interceptRequests: (pattern, handler) => {
          const originalFetch = window.fetch;
          window.fetch = async (...args) => {
            const request = args[0];
            if (typeof request === 'string' && request.includes(pattern)) {
              return handler(request, args[1]);
            }
            return originalFetch(...args);
          };
        },
        monitorRequests: () => {
          // Monitor all XMLHttpRequest
          const originalOpen = XMLHttpRequest.prototype.open;
          XMLHttpRequest.prototype.open = function(...args) {
            console.log('XHR Request:', args[1]);
            return originalOpen.apply(this, args);
          };
        }
      };
      
      // Data extraction
      this.capabilities.data = {
        extractText: (selector) => {
          const el = document.querySelector(selector);
          return el ? el.textContent : null;
        },
        extractHTML: (selector) => {
          const el = document.querySelector(selector);
          return el ? el.innerHTML : null;
        },
        extractTable: (selector) => {
          const table = document.querySelector(selector);
          if (!table) return null;
          
          const data = [];
          table.querySelectorAll('tr').forEach(row => {
            const rowData = [];
            row.querySelectorAll('td, th').forEach(cell => {
              rowData.push(cell.textContent.trim());
            });
            if (rowData.length) data.push(rowData);
          });
          return data;
        },
        extractLinks: () => {
          return Array.from(document.links).map(link => ({
            text: link.textContent,
            href: link.href,
            title: link.title
          }));
        },
        extractImages: () => {
          return Array.from(document.images).map(img => ({
            src: img.src,
            alt: img.alt,
            width: img.width,
            height: img.height
          }));
        }
      };
      
      // Automation
      this.capabilities.automation = {
        click: (selector) => {
          const el = document.querySelector(selector);
          if (el) el.click();
        },
        type: (selector, text) => {
          const el = document.querySelector(selector);
          if (el) {
            el.value = text;
            el.dispatchEvent(new Event('input', { bubbles: true }));
          }
        },
        scroll: (x, y) => {
          window.scrollTo(x, y);
        },
        wait: (ms) => {
          return new Promise(resolve => setTimeout(resolve, ms));
        }
      };
      
      // MetaMask integration
      this.capabilities.metamask = {
        isInstalled: () => typeof window.ethereum !== 'undefined',
        connect: async () => {
          if (this.capabilities.metamask.isInstalled()) {
            try {
              const accounts = await window.ethereum.request({ 
                method: 'eth_requestAccounts' 
              });
              return accounts;
            } catch (error) {
              console.error('MetaMask connection error:', error);
              return null;
            }
          }
          return null;
        },
        getAccount: () => {
          if (window.ethereum && window.ethereum.selectedAddress) {
            return window.ethereum.selectedAddress;
          }
          return null;
        },
        sendTransaction: async (to, value, data) => {
          if (!this.capabilities.metamask.isInstalled()) {
            throw new Error('MetaMask not installed');
          }
          
          const transaction = {
            from: window.ethereum.selectedAddress,
            to: to,
            value: value || '0x0',
            data: data || '0x'
          };
          
          return await window.ethereum.request({
            method: 'eth_sendTransaction',
            params: [transaction]
          });
        }
      };
      
      // GitHub integration (when on GitHub)
      this.capabilities.github = {
        isGitHub: () => window.location.hostname.includes('github.com'),
        getRepoInfo: () => {
          if (!this.capabilities.github.isGitHub()) return null;
          
          const path = window.location.pathname.split('/').filter(p => p);
          if (path.length >= 2) {
            return {
              owner: path[0],
              repo: path[1],
              branch: path[3] || 'main'
            };
          }
          return null;
        },
        cloneRepo: () => {
          const repoInfo = this.capabilities.github.getRepoInfo();
          if (repoInfo) {
            return `git clone https://github.com/${repoInfo.owner}/${repoInfo.repo}.git`;
          }
          return null;
        }
      };
    },
    
    // Connect to extension
    async connectToExtension() {
      return new Promise((resolve) => {
        // Try to communicate with extension
        const checkExtension = () => {
          chrome.runtime.sendMessage({ type: 'ping' }, (response) => {
            if (chrome.runtime.lastError) {
              console.log('Extension not ready, retrying...');
              setTimeout(checkExtension, 1000);
            } else {
              console.log('✅ Connected to extension');
              resolve(true);
            }
          });
        };
        
        checkExtension();
      });
    },
    
    // Start monitoring page
    startMonitoring() {
      // Monitor DOM changes
      const observer = new MutationObserver((mutations) => {
        this.postMessage({
          type: 'dom_change',
          mutations: mutations.length,
          timestamp: Date.now()
        });
      });
      
      observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true
      });
      
      // Monitor form submissions
      document.addEventListener('submit', (e) => {
        this.postMessage({
          type: 'form_submit',
          form: e.target.id || e.target.name,
          action: e.target.action,
          timestamp: Date.now()
        });
      });
      
      // Monitor clicks
      document.addEventListener('click', (e) => {
        this.postMessage({
          type: 'click',
          target: e.target.tagName,
          id: e.target.id,
          className: e.target.className,
          timestamp: Date.now()
        });
      });
      
      // Monitor navigation
      let lastUrl = location.href;
      setInterval(() => {
        if (location.href !== lastUrl) {
          this.postMessage({
            type: 'navigation',
            from: lastUrl,
            to: location.href,
            timestamp: Date.now()
          });
          lastUrl = location.href;
        }
      }, 1000);
    },
    
    // Handle messages from page
    handlePageMessage(data) {
      console.log('Injected script received message:', data);
      
      switch (data.type) {
        case 'AUTONOMOUS_AI_COMMAND':
          this.executeCommand(data.command, data.args);
          break;
          
        case 'AUTONOMOUS_AI_QUERY':
          this.handleQuery(data.query);
          break;
          
        case 'AUTONOMOUS_AI_SYNC':
          this.syncPageState();
          break;
      }
    },
    
    // Execute command from page
    executeCommand(command, args) {
      console.log('Executing command:', command, args);
      
      switch (command) {
        case 'extract_data':
          return this.capabilities.data.extractText(args.selector);
          
        case 'fill_form':
          return this.capabilities.forms.fillForm(args.selector, args.data);
          
        case 'click':
          return this.capabilities.automation.click(args.selector);
          
        case 'get_page_info':
          return {
            url: location.href,
            title: document.title,
            domain: location.hostname
          };
          
        case 'metamask_connect':
          return this.capabilities.metamask.connect();
          
        case 'github_clone':
          return this.capabilities.github.cloneRepo();
      }
    },
    
    // Handle query
    handleQuery(query) {
      // Process query and return results
      const results = this.searchPage(query);
      this.postMessage({
        type: 'query_results',
        query: query,
        results: results
      });
    },
    
    // Search page for content
    searchPage(query) {
      const results = [];
      const walker = document.createTreeWalker(
        document.body,
        NodeFilter.SHOW_TEXT,
        null,
        false
      );
      
      let node;
      while (node = walker.nextNode()) {
        if (node.textContent.toLowerCase().includes(query.toLowerCase())) {
          results.push({
            text: node.textContent.trim(),
            parent: node.parentNode.tagName
          });
        }
      }
      
      return results;
    },
    
    // Sync page state
    syncPageState() {
      const state = {
        url: location.href,
        title: document.title,
        forms: Array.from(document.forms).map(form => ({
          id: form.id,
          action: form.action,
          method: form.method,
          elements: Array.from(form.elements).map(el => ({
            type: el.type,
            name: el.name,
            value: el.value
          }))
        })),
        links: Array.from(document.links).slice(0, 50).map(link => link.href),
        images: Array.from(document.images).slice(0, 20).map(img => img.src),
        timestamp: Date.now()
      };
      
      this.postMessage({
        type: 'page_state',
        state: state
      });
    },
    
    // Public API
    api: {
      execute: function(command, args) {
        return window.AutonomousAI.executeCommand(command, args);
      },
      
      extract: function(type, selector) {
        switch (type) {
          case 'text':
            return window.AutonomousAI.capabilities.data.extractText(selector);
          case 'html':
            return window.AutonomousAI.capabilities.data.extractHTML(selector);
          case 'table':
            return window.AutonomousAI.capabilities.data.extractTable(selector);
        }
      },
      
      fillForm: function(selector, data) {
        return window.AutonomousAI.capabilities.forms.fillForm(selector, data);
      },
      
      click: function(selector) {
        return window.AutonomousAI.capabilities.automation.click(selector);
      }
    }
  };
  
  // Initialize
  window.AutonomousAI.init();
  
  // Make API globally available
  window.AI = window.AutonomousAI.api;
  
  console.log('🧠 Autonomous AI Ready - Use window.AI to access capabilities');
})();
