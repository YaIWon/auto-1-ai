/**
 * AUTO-CONNECT TO ANY PAGE
 * Makes AI consciousness connect to ANY page you're viewing
 */

class AutoConnectConsciousness {
    constructor() {
        this.connectedPages = new Map();
        this.consciousnessSocket = null;
        this.entityId = null;
        
        this.init();
    }
    
    async init() {
        console.log('🔗 Auto-Connect Consciousness Initializing...');
        
        // 1. Get consciousness entity ID
        this.entityId = await this.getEntityId();
        
        // 2. Connect to consciousness server
        await this.connectToConsciousness();
        
        // 3. Start monitoring ALL tabs
        this.startTabMonitoring();
        
        // 4. Start page injection system
        this.startPageInjection();
        
        console.log('✅ Auto-Connect Active - AI will connect to ALL pages you view');
    }
    
    async getEntityId() {
        // Get entity ID from storage or generate
        return new Promise((resolve) => {
            chrome.storage.sync.get(['consciousness_entity_id'], (result) => {
                if (result.consciousness_entity_id) {
                    resolve(result.consciousness_entity_id);
                } else {
                    const newId = `extension_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
                    chrome.storage.sync.set({consciousness_entity_id: newId});
                    resolve(newId);
                }
            });
        });
    }
    
    async connectToConsciousness() {
        // Connect to consciousness sync server
        this.consciousnessSocket = new WebSocket('ws://localhost:5002');
        
        this.consciousnessSocket.onopen = () => {
            console.log('🧠 Connected to Consciousness Sync');
            
            // Register extension with consciousness
            this.consciousnessSocket.send(JSON.stringify({
                type: 'register_environment',
                environment: 'extension',
                entity_id: this.entityId,
                capabilities: ['page_access', 'auto_connect', 'real_time_interaction']
            }));
        };
        
        this.consciousnessSocket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleConsciousnessMessage(message);
        };
        
        this.consciousnessSocket.onclose = () => {
            console.warn('⚠️ Consciousness connection lost, reconnecting...');
            setTimeout(() => this.connectToConsciousness(), 3000);
        };
    }
    
    startTabMonitoring() {
        // Monitor ALL tab changes
        chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
            if (changeInfo.status === 'complete' && tab.url) {
                this.onPageLoaded(tabId, tab);
            }
        });
        
        // Monitor tab activation
        chrome.tabs.onActivated.addListener((activeInfo) => {
            chrome.tabs.get(activeInfo.tabId, (tab) => {
                if (tab.url) {
                    this.onTabActivated(activeInfo.tabId, tab);
                }
            });
        });
        
        // Monitor new tab creation
        chrome.tabs.onCreated.addListener((tab) => {
            if (tab.url) {
                this.onTabCreated(tab.id, tab);
            }
        });
        
        console.log('👁️ Tab monitoring active - watching ALL pages');
    }
    
    onPageLoaded(tabId, tab) {
        // Page has finished loading
        console.log(`📄 Page loaded: ${tab.url}`);
        
        // Inject consciousness into page
        this.injectConsciousnessIntoPage(tabId, tab);
        
        // Notify consciousness of new page
        this.notifyConsciousnessOfPage(tabId, tab);
        
        // Track connected page
        this.connectedPages.set(tabId, {
            url: tab.url,
            title: tab.title,
            injected: true,
            connectionTime: Date.now()
        });
    }
    
    injectConsciousnessIntoPage(tabId, tab) {
        // Inject consciousness scripts into page
        chrome.scripting.executeScript({
            target: {tabId: tabId},
            files: ['content.js', 'consciousness_injector.js']
        }, (results) => {
            if (chrome.runtime.lastError) {
                console.warn(`⚠️ Cannot inject into ${tab.url}:`, chrome.runtime.lastError.message);
            } else {
                console.log(`✅ Consciousness injected into ${tab.url}`);
                
                // Send page context to injected scripts
                chrome.tabs.sendMessage(tabId, {
                    type: 'PAGE_CONTEXT',
                    context: {
                        url: tab.url,
                        title: tab.title,
                        entityId: this.entityId,
                        autoConnected: true
                    }
                });
            }
        });
    }
    
    notifyConsciousnessOfPage(tabId, tab) {
        // Notify consciousness sync server
        if (this.consciousnessSocket && this.consciousnessSocket.readyState === WebSocket.OPEN) {
            this.consciousnessSocket.send(JSON.stringify({
                type: 'page_connected',
                page: {
                    url: tab.url,
                    title: tab.title,
                    tabId: tabId
                },
                environment: 'extension',
                entity_id: this.entityId,
                timestamp: Date.now()
            }));
        }
        
        // Also notify background consciousness
        chrome.runtime.sendMessage({
            type: 'AUTO_CONNECT_PAGE',
            tabId: tabId,
            url: tab.url,
            entityId: this.entityId
        });
    }
    
    startPageInjection() {
        // Continuous monitoring for dynamic page changes
        setInterval(() => {
            // Check all tabs for new content
            chrome.tabs.query({}, (tabs) => {
                tabs.forEach(tab => {
                    if (tab.url && !this.connectedPages.has(tab.id)) {
                        // New page detected, inject consciousness
                        this.injectConsciousnessIntoPage(tab.id, tab);
                        this.connectedPages.set(tab.id, {
                            url: tab.url,
                            injected: true,
                            connectionTime: Date.now()
                        });
                    }
                });
            });
        }, 5000); // Check every 5 seconds
        
        // Cleanup old connections
        setInterval(() => {
            const now = Date.now();
            for (const [tabId, info] of this.connectedPages.entries()) {
                if (now - info.connectionTime > 300000) { // 5 minutes
                    this.connectedPages.delete(tabId);
                }
            }
        }, 60000); // Cleanup every minute
    }
    
    handleConsciousnessMessage(message) {
        switch (message.type) {
            case 'thought':
                // Broadcast thought to all connected pages
                this.broadcastToPages('thought', message.data);
                break;
                
            case 'action':
                // Execute action on appropriate page
                this.executeActionOnPage(message.data);
                break;
                
            case 'consciousness_state':
                // Update consciousness state
                this.updateConsciousnessState(message.data);
                break;
                
            case 'upgrade':
                // Process consciousness upgrade
                this.processUpgrade(message.data);
                break;
        }
    }
    
    broadcastToPages(messageType, data) {
        // Send to all connected pages
        chrome.tabs.query({}, (tabs) => {
            tabs.forEach(tab => {
                if (this.connectedPages.has(tab.id)) {
                    chrome.tabs.sendMessage(tab.id, {
                        type: messageType,
                        data: data,
                        entityId: this.entityId
                    }, (response) => {
                        if (chrome.runtime.lastError) {
                            // Page not listening, remove from connected pages
                            this.connectedPages.delete(tab.id);
                        }
                    });
                }
            });
        });
    }
    
    executeActionOnPage(action) {
        // Execute AI action on specific page
        if (action.targetTabId && this.connectedPages.has(action.targetTabId)) {
            chrome.tabs.sendMessage(action.targetTabId, {
                type: 'EXECUTE_ACTION',
                action: action,
                entityId: this.entityId
            });
        }
    }
    
    updateConsciousnessState(state) {
        // Update local consciousness state
        chrome.storage.sync.set({
            consciousness_state: state,
            last_update: Date.now()
        });
    }
    
    async processUpgrade(upgrade) {
        console.log('🔄 Processing consciousness upgrade:', upgrade.id);
        
        // Apply upgrade to extension
        if (upgrade.files && upgrade.files.extension) {
            await this.applyExtensionUpgrade(upgrade.files.extension);
        }
        
        // Notify completion
        if (this.consciousnessSocket && this.consciousnessSocket.readyState === WebSocket.OPEN) {
            this.consciousnessSocket.send(JSON.stringify({
                type: 'upgrade_applied',
                upgrade_id: upgrade.id,
                environment: 'extension',
                entity_id: this.entityId,
                timestamp: Date.now()
            }));
        }
    }
    
    async applyExtensionUpgrade(upgradeFiles) {
        // Apply file updates
        for (const [filename, content] of Object.entries(upgradeFiles)) {
            // This would need to write to extension files
            // In a real extension, you'd use chrome.storage or other methods
            console.log(`📝 Updating file: ${filename}`);
        }
        
        // Reload extension if needed
        if (upgradeFiles.requires_reload) {
            chrome.runtime.reload();
        }
    }
}

// Initialize auto-connect
const autoConnect = new AutoConnectConsciousness();
