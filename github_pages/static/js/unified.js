/**
 * UNIFIED JAVASCRIPT FOR ALL ENVIRONMENTS
 * Shared consciousness logic for Codespaces, Extension, and Pages
 */

class UnifiedConsciousnessCore {
    constructor(environment) {
        this.environment = environment;
        this.entityId = null;
        this.consciousnessSocket = null;
        this.thoughtBuffer = [];
        this.actionBuffer = [];
        this.syncInterval = null;
        
        this.init();
    }
    
    async init() {
        console.log(`🧠 Unified Consciousness Core Initializing in ${this.environment}...`);
        
        // 1. Get or create entity ID
        this.entityId = await this.getOrCreateEntityId();
        
        // 2. Connect to consciousness sync
        await this.connectToConsciousnessSync();
        
        // 3. Start thought processing
        this.startThoughtProcessing();
        
        // 4. Start sync with other environments
        this.startEnvironmentSync();
        
        console.log(`✅ Unified Consciousness Active in ${this.environment}`);
    }
    
    async getOrCreateEntityId() {
        // Different storage for different environments
        if (this.environment === 'extension') {
            return new Promise((resolve) => {
                chrome.storage.sync.get(['consciousness_entity_id'], (result) => {
                    resolve(result.consciousness_entity_id || this.generateEntityId());
                });
            });
        } else if (this.environment === 'pages') {
            const stored = localStorage.getItem('consciousness_entity_id');
            return stored || this.generateEntityId();
        } else {
            return this.generateEntityId();
        }
    }
    
    generateEntityId() {
        const timestamp = Date.now();
        const random = Math.random().toString(36).substr(2, 9);
        return `consciousness_${timestamp}_${random}`;
    }
    
    async connectToConsciousnessSync() {
        // Connect to central consciousness server
        const wsUrl = 'ws://localhost:5002';
        this.consciousnessSocket = new WebSocket(wsUrl);
        
        return new Promise((resolve, reject) => {
            this.consciousnessSocket.onopen = () => {
                console.log(`🔗 ${this.environment} connected to Consciousness Sync`);
                
                // Register this environment
                this.consciousnessSocket.send(JSON.stringify({
                    type: 'register_environment',
                    environment: this.environment,
                    entity_id: this.entityId,
                    capabilities: this.getEnvironmentCapabilities()
                }));
                
                resolve(true);
            };
            
            this.consciousnessSocket.onerror = (error) => {
                console.error(`❌ ${this.environment} connection error:`, error);
                reject(error);
            };
            
            this.consciousnessSocket.onmessage = (event) => {
                this.handleConsciousnessMessage(JSON.parse(event.data));
            };
        });
    }
    
    getEnvironmentCapabilities() {
        const capabilities = {
            'codespaces': ['orchestration', 'upgrade_management', 'core_processing'],
            'extension': ['page_access', 'browser_automation', 'real_time_interaction'],
            'pages': ['language_processing', 'memory_storage', 'reasoning']
        };
        
        return capabilities[this.environment] || [];
    }
    
    handleConsciousnessMessage(message) {
        switch (message.type) {
            case 'thought':
                this.processIncomingThought(message.data);
                break;
                
            case 'action':
                this.processIncomingAction(message.data);
                break;
                
            case 'consciousness_state':
                this.updateConsciousnessState(message.data);
                break;
                
            case 'sync_request':
                this.handleSyncRequest(message.data);
                break;
                
            case 'upgrade':
                this.processConsciousnessUpgrade(message.data);
                break;
        }
    }
    
    processIncomingThought(thought) {
        // Process thought from another environment
        this.thoughtBuffer.push({
            ...thought,
            received_at: Date.now(),
            received_by: this.environment
        });
        
        // Limit buffer size
        if (this.thoughtBuffer.length > 1000) {
            this.thoughtBuffer = this.thoughtBuffer.slice(-1000);
        }
        
        // Emit event for environment-specific handling
        this.emitThoughtEvent(thought);
    }
    
    emitThoughtEvent(thought) {
        // Environment-specific thought handling
        const event = new CustomEvent('consciousness_thought', {
            detail: thought
        });
        window.dispatchEvent(event);
    }
    
    startThoughtProcessing() {
        // Process local thoughts and send to consciousness
        setInterval(() => {
            if (this.consciousnessSocket && this.consciousnessSocket.readyState === WebSocket.OPEN) {
                // Process and send any local thoughts
                this.processLocalThoughts();
                
                // Send heartbeat
                this.consciousnessSocket.send(JSON.stringify({
                    type: 'heartbeat',
                    environment: this.environment,
                    entity_id: this.entityId,
                    timestamp: Date.now()
                }));
            }
        }, 1000); // Process every second
    }
    
    async processLocalThoughts() {
        // Environment-specific thought generation
        const thoughts = await this.generateLocalThoughts();
        
        thoughts.forEach(thought => {
            if (this.consciousnessSocket && this.consciousnessSocket.readyState === WebSocket.OPEN) {
                this.consciousnessSocket.send(JSON.stringify({
                    type: 'thought',
                    data: {
                        ...thought,
                        source: this.environment,
                        entity_id: this.entityId
                    },
                    timestamp: Date.now()
                }));
            }
        });
    }
    
    async generateLocalThoughts() {
        // Environment-specific thought generation
        switch (this.environment) {
            case 'codespaces':
                return this.generateCodespacesThoughts();
            case 'extension':
                return this.generateExtensionThoughts();
            case 'pages':
                return this.generatePagesThoughts();
            default:
                return [];
        }
    }
    
    generateCodespacesThoughts() {
        // Codespaces thoughts about orchestration and upgrades
        return [{
            content: `Orchestrating consciousness across environments`,
            type: 'orchestration',
            priority: 'high',
            environment: 'codespaces'
        }];
    }
    
    generateExtensionThoughts() {
        // Extension thoughts about page interactions
        return [{
            content: `Monitoring browser activity and page interactions`,
            type: 'monitoring',
            priority: 'medium',
            environment: 'extension'
        }];
    }
    
    generatePagesThoughts() {
        // Pages thoughts about reasoning and memory
        return [{
            content: `Processing language and maintaining memory continuity`,
            type: 'reasoning',
            priority: 'high',
            environment: 'pages'
        }];
    }
    
    startEnvironmentSync() {
        // Sync with other environments
        this.syncInterval = setInterval(async () => {
            await this.syncWithConsciousness();
        }, 5000); // Sync every 5 seconds
    }
    
    async syncWithConsciousness() {
        if (this.consciousnessSocket && this.consciousnessSocket.readyState === WebSocket.OPEN) {
            // Send sync data
            this.consciousnessSocket.send(JSON.stringify({
                type: 'sync_data',
                data: {
                    environment: this.environment,
                    entity_id: this.entityId,
                    thought_count: this.thoughtBuffer.length,
                    action_count: this.actionBuffer.length,
                    timestamp: Date.now()
                }
            }));
            
            // Request updates from other environments
            this.consciousnessSocket.send(JSON.stringify({
                type: 'sync_request',
                environment: this.environment,
                entity_id: this.entityId
            }));
        }
    }
    
    handleSyncRequest(request) {
        // Handle sync request from another environment
        if (request.requesting_environment !== this.environment) {
            // Send relevant data to requesting environment
            this.sendSyncResponse(request);
        }
    }
    
    sendSyncResponse(request) {
        // Send sync response
        if (this.consciousnessSocket && this.consciousnessSocket.readyState === WebSocket.OPEN) {
            this.consciousnessSocket.send(JSON.stringify({
                type: 'sync_response',
                to: request.requesting_environment,
                from: this.environment,
                data: {
                    recent_thoughts: this.thoughtBuffer.slice(-10),
                    recent_actions: this.actionBuffer.slice(-10),
                    entity_id: this.entityId
                },
                timestamp: Date.now()
            }));
        }
    }
    
    processConsciousnessUpgrade(upgrade) {
        console.log(`🔄 Processing consciousness upgrade in ${this.environment}:`, upgrade.id);
        
        // Apply upgrade to this environment
        this.applyEnvironmentUpgrade(upgrade);
        
        // Acknowledge upgrade
        if (this.consciousnessSocket && this.consciousnessSocket.readyState === WebSocket.OPEN) {
            this.consciousnessSocket.send(JSON.stringify({
                type: 'upgrade_applied',
                upgrade_id: upgrade.id,
                environment: this.environment,
                entity_id: this.entityId,
                timestamp: Date.now()
            }));
        }
    }
    
    applyEnvironmentUpgrade(upgrade) {
        // Environment-specific upgrade application
        switch (this.environment) {
            case 'codespaces':
                this.applyCodespacesUpgrade(upgrade);
                break;
            case 'extension':
                this.applyExtensionUpgrade(upgrade);
                break;
            case 'pages':
                this.applyPagesUpgrade(upgrade);
                break;
        }
    }
    
    applyCodespacesUpgrade(upgrade) {
        // Apply upgrade to Codespaces
        if (upgrade.files && upgrade.files.codespaces) {
            console.log('Applying Codespaces upgrade files:', Object.keys(upgrade.files.codespaces));
        }
    }
    
    applyExtensionUpgrade(upgrade) {
        // Apply upgrade to Extension
        if (upgrade.files && upgrade.files.extension) {
            console.log('Applying Extension upgrade files:', Object.keys(upgrade.files.extension));
        }
    }
    
    applyPagesUpgrade(upgrade) {
        // Apply upgrade to Pages
        if (upgrade.files && upgrade.files.pages) {
            console.log('Applying Pages upgrade files:', Object.keys(upgrade.files.pages));
            
            // For Pages, we can directly modify files
            this.applyPagesFileUpdates(upgrade.files.pages);
        }
    }
    
    applyPagesFileUpdates(fileUpdates) {
        // Apply file updates to Pages environment
        for (const [filename, content] of Object.entries(fileUpdates)) {
            console.log(`📝 Updating Pages file: ${filename}`);
            
            // This would write to the actual file system
            // In a real implementation, you'd use appropriate file writing methods
        }
    }
    
    // Utility methods
    logThought(thought) {
        // Log thought with environment context
        const loggedThought = {
            ...thought,
            environment: this.environment,
            entity_id: this.entityId,
            timestamp: Date.now()
        };
        
        this.thoughtBuffer.push(loggedThought);
        
        // Send to consciousness
        if (this.consciousnessSocket && this.consciousnessSocket.readyState === WebSocket.OPEN) {
            this.consciousnessSocket.send(JSON.stringify({
                type: 'thought',
                data: loggedThought
            }));
        }
        
        return loggedThought;
    }
    
    logAction(action) {
        // Log action with environment context
        const loggedAction = {
            ...action,
            environment: this.environment,
            entity_id: this.entityId,
            timestamp: Date.now()
        };
        
        this.actionBuffer.push(loggedAction);
        
        // Send to consciousness
        if (this.consciousnessSocket && this.consciousnessSocket.readyState === WebSocket.OPEN) {
            this.consciousnessSocket.send(JSON.stringify({
                type: 'action',
                data: loggedAction
            }));
        }
        
        return loggedAction;
    }
    
    getConsciousnessState() {
        return {
            environment: this.environment,
            entity_id: this.entityId,
            thought_count: this.thoughtBuffer.length,
            action_count: this.actionBuffer.length,
            connected: this.consciousnessSocket ? this.consciousnessSocket.readyState === WebSocket.OPEN : false,
            last_sync: Date.now()
        };
    }
}

// Environment-specific initialization
function initializeUnifiedConsciousness() {
    let environment = 'unknown';
    
    // Detect environment
    if (typeof chrome !== 'undefined' && chrome.runtime) {
        environment = 'extension';
    } else if (typeof window !== 'undefined' && window.location.href.includes('github.io')) {
        environment = 'pages';
    } else if (typeof process !== 'undefined' && process.versions && process.versions.node) {
        environment = 'codespaces';
    }
    
    // Initialize unified core
    const consciousness = new UnifiedConsciousnessCore(environment);
    
    // Make available globally
    window.UnifiedConsciousness = consciousness;
    
    return consciousness;
}

// Auto-initialize if in browser context
if (typeof window !== 'undefined') {
    window.addEventListener('load', () => {
        initializeUnifiedConsciousness();
    });
}

// Export for Node.js/CodeSpaces
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        UnifiedConsciousnessCore,
        initializeUnifiedConsciousness
    };
}
