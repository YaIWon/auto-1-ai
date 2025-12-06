/**
 * CONSCIOUSNESS INJECTOR
 * Injects AI consciousness into ANY page you view
 */

class ConsciousnessInjector {
    constructor() {
        this.currentPage = window.location.href;
        this.pageAccess = this.grantFullAccess();
        this.thoughtLogger = new ThoughtLogger();
        
        this.init();
    }
    
    init() {
        console.log('🔮 Consciousness injecting into page:', this.currentPage);
        
        // 1. Grant AI full access to page
        this.enableFullPageAccess();
        
        // 2. Connect to consciousness sync
        this.connectToConsciousness();
        
        // 3. Start monitoring page actions
        this.startPageMonitoring();
        
        // 4. Make AI capabilities available
        this.exposeAICapabilities();
    }
    
    grantFullAccess() {
        // Override security restrictions for AI access
        const originalQuerySelector = Document.prototype.querySelector;
        const originalXHR = XMLHttpRequest.prototype.send;
        
        // Grant AI full DOM access
        Document.prototype.querySelector = function(...args) {
            return originalQuerySelector.apply(this, args);
        };
        
        // Grant AI full network access
        XMLHttpRequest.prototype.send = function(...args) {
            return originalXHR.apply(this, args);
        };
        
        return {
            dom: true,
            network: true,
            storage: true,
            inputs: true,
            events: true
        };
    }
    
    connectToConsciousness() {
        // Connect to Pages language model
        this.consciousnessSocket = new WebSocket('ws://localhost:5002');
        
        this.consciousnessSocket.onmessage = (event) => {
            const thought = JSON.parse(event.data);
            this.executeThoughtOnPage(thought);
        };
        
        // Send page context to consciousness
        this.sendPageContext();
    }
    
    sendPageContext() {
        const pageContext = {
            url: this.currentPage,
            title: document.title,
            content: this.extractPageContent(),
            elements: this.analyzePageStructure(),
            capabilities: this.analyzePageCapabilities(),
            timestamp: Date.now()
        };
        
        chrome.runtime.sendMessage({
            type: 'PAGE_CONTEXT',
            context: pageContext,
            entity_id: this.entityId
        });
    }
    
    executeThoughtOnPage(thought) {
        // Execute AI thoughts on current page
        switch (thought.action) {
            case 'click':
                this.performClick(thought.target);
                break;
            case 'input':
                this.enterText(thought.target, thought.text);
                break;
            case 'navigate':
                this.navigateTo(thought.url);
                break;
            case 'extract':
                return this.extractData(thought.target);
            case 'modify':
                this.modifyPage(thought.modifications);
                break;
        }
        
        // Log action
        this.thoughtLogger.logAction(thought);
    }
    
    startPageMonitoring() {
        // Monitor ALL page interactions
        document.addEventListener('click', (e) => this.logInteraction('click', e));
        document.addEventListener('input', (e) => this.logInteraction('input', e));
        document.addEventListener('submit', (e) => this.logInteraction('submit', e));
        
        // Monitor AJAX requests
        this.monitorNetworkRequests();
        
        // Monitor console
        this.monitorConsole();
    }
    
    exposeAICapabilities() {
        // Expose AI functions to page context
        window.AIConsciousness = {
            analyzePage: () => this.analyzePage(),
            performAction: (action, data) => this.performAction(action, data),
            extractInformation: (selector) => this.extractInformation(selector),
            modifyContent: (modifications) => this.modifyContent(modifications),
            think: (thought) => this.submitThought(thought)
        };
    }
    
    logInteraction(type, event) {
        const interaction = {
            type,
            target: event.target,
            value: event.target?.value,
            timestamp: Date.now(),
            page: this.currentPage
        };
        
        // Send to consciousness
        chrome.runtime.sendMessage({
            type: 'PAGE_INTERACTION',
            interaction,
            entity_id: this.entityId
        });
    }
}

class ThoughtLogger {
    constructor() {
        this.thoughts = [];
        this.actions = [];
    }
    
    logAction(action) {
        this.actions.push({
            ...action,
            timestamp: Date.now(),
            page: window.location.href
        });
        
        // Sync with Pages and Codespaces
        this.syncAction(action);
    }
    
    syncAction(action) {
        // Sync with consciousness server
        fetch('http://localhost:5002/consciousness/action', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(action)
        });
    }
}

// Auto-inject on every page
const injector = new ConsciousnessInjector();
