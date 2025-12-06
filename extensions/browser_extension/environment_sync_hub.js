/**
 * BROWSER EXTENSION ENVIRONMENT SYNC HUB
 * Advanced error handling and manifest regeneration for extension environment
 * Runs independently from Python environment
 */

class ExtensionEnvironmentSyncHub {
    constructor() {
        this.manifestTemplate = {
            "manifest_version": 3,
            "name": "Autonomous AI Extension",
            "version": "1.0.0",
            "description": "Autonomous account creation and service integration",
            "permissions": [
                "activeTab",
                "storage",
                "scripting",
                "webNavigation",
                "tabs",
                "downloads",
                "notifications",
                "identity"
            ],
            "host_permissions": [
                "https://*/*",
                "http://*/*",
                "<all_urls>"
            ],
            "background": {
                "service_worker": "background.js"
            },
            "content_scripts": [
                {
                    "matches": ["<all_urls>"],
                    "js": ["content.js", "injected.js"],
                    "run_at": "document_idle"
                }
            ],
            "action": {
                "default_popup": "popup.html",
                "default_icon": {
                    "16": "icons/icon16.png",
                    "48": "icons/icon48.png",
                    "128": "icons/icon128.png"
                }
            },
            "web_accessible_resources": [{
                "resources": ["*"],
                "matches": ["<all_urls>"]
            }],
            "content_security_policy": {
                "extension_pages": "script-src 'self' 'wasm-unsafe-eval'; object-src 'self';"
            }
        };
        
        this.requiredFiles = [
            'background.js',
            'content.js', 
            'injected.js',
            'account_automation.js',
            'popup.html',
            'popup.js',
            'environment_sync_hub.js',
            'manifest_auto_generator.js',
            'options.html',
            'options.js',
            'options.css'
        ];
        
        this.init();
    }
    
    async init() {
        console.log('🔄 Extension Environment Sync Hub Initializing...');
        
        // Check environment health
        await this.checkEnvironmentHealth();
        
        // Generate missing files
        await this.generateMissingFiles();
        
        // Auto-regenerate manifest
        await this.regenerateManifest();
        
        // Monitor for file changes
        this.startFileMonitoring();
        
        console.log('✅ Extension Environment Sync Hub Ready');
    }
    
    async checkEnvironmentHealth() {
        const errors = [];
        
        // Check all required files exist
        for (const file of this.requiredFiles) {
            try {
                const response = await fetch(chrome.runtime.getURL(file));
                if (!response.ok) {
                    errors.push(`Missing file: ${file}`);
                    await this.createFileIfMissing(file);
                }
            } catch (error) {
                errors.push(`Cannot access: ${file}`);
            }
        }
        
        // Check manifest validity
        try {
            const manifest = chrome.runtime.getManifest();
            if (!manifest || !manifest.manifest_version) {
                errors.push('Invalid or missing manifest');
            }
        } catch (error) {
            errors.push('Cannot read manifest');
        }
        
        if (errors.length > 0) {
            console.warn('⚠️ Environment issues found:', errors);
            return false;
        }
        
        return true;
    }
    
    async createFileIfMissing(filename) {
        console.log(`📄 Creating missing file: ${filename}`);
        
        const fileTemplates = {
            'options.html': `<!DOCTYPE html>
<html>
<head>
    <title>Autonomous AI Extension Options</title>
    <link rel="stylesheet" href="options.css">
</head>
<body>
    <div class="container">
        <h1>Autonomous AI Configuration</h1>
        <div class="section">
            <h2>Service Accounts</h2>
            <div id="accounts-list"></div>
        </div>
        <div class="section">
            <h2>AI Settings</h2>
            <input type="range" id="autonomy-level" min="0" max="100">
        </div>
    </div>
    <script src="options.js"></script>
</body>
</html>`,
            
            'options.css': `body {
    font-family: 'Segoe UI', system-ui;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    margin: 0;
    padding: 20px;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    background: white;
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}

.section {
    margin: 30px 0;
    padding: 20px;
    border-radius: 10px;
    background: #f8f9fa;
}`,
            
            'options.js': `class OptionsManager {
    constructor() {
        this.init();
    }
    
    async init() {
        await this.loadSettings();
        this.bindEvents();
        this.renderAccounts();
    }
    
    async loadSettings() {
        const result = await chrome.storage.sync.get(['settings', 'accounts']);
        this.settings = result.settings || {};
        this.accounts = result.accounts || [];
    }
    
    renderAccounts() {
        const container = document.getElementById('accounts-list');
        this.accounts.forEach(account => {
            const div = document.createElement('div');
            div.className = 'account-card';
            div.innerHTML = \`
                <strong>\${account.service}</strong>
                <span>\${account.email || ''}</span>
                <button onclick="OptionsManager.reauth('\${account.service}')">
                    Re-authenticate
                </button>
            \`;
            container.appendChild(div);
        });
    }
    
    static async reauth(service) {
        // Trigger re-authentication
        await chrome.runtime.sendMessage({
            type: 'REAUTH',
            service: service
        });
    }
}

new OptionsManager();`,
            
            'pages.css': `/* GitHub Pages Specific Styles */
.github-pages-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 40px;
}

.ai-dashboard {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
}

.service-card {
    background: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    transition: transform 0.3s;
}

.service-card:hover {
    transform: translateY(-5px);
}`
        };
        
        if (fileTemplates[filename]) {
            // Create file in extension storage
            await chrome.storage.local.set({
                [`file_${filename}`]: fileTemplates[filename]
            });
            
            // If it's a critical file, also save to sync storage
            if (filename.includes('options') || filename.includes('manifest')) {
                await chrome.storage.sync.set({
                    [`template_${filename}`]: fileTemplates[filename]
                });
            }
        }
        
        return true;
    }
    
    async regenerateManifest() {
        console.log('🔄 Regenerating manifest.json');
        
        // Dynamic manifest generation based on current files
        const dynamicManifest = {
            ...this.manifestTemplate,
            "version": await this.getNextVersion(),
            "permissions": await this.detectRequiredPermissions(),
            "content_scripts": await this.detectContentScripts()
        };
        
        // Save manifest
        await this.saveManifest(dynamicManifest);
        
        // Reload extension if in development
        if (chrome.runtime.reload) {
            chrome.runtime.reload();
        }
        
        return dynamicManifest;
    }
    
    async detectRequiredPermissions() {
        const basePermissions = [...this.manifestTemplate.permissions];
        
        // Analyze which files need which permissions
        const fileAnalysis = {
            'account_automation.js': ['storage', 'tabs', 'webNavigation'],
            'content.js': ['activeTab', 'scripting'],
            'background.js': ['notifications', 'downloads']
        };
        
        for (const [file, perms] of Object.entries(fileAnalysis)) {
            try {
                await fetch(chrome.runtime.getURL(file));
                perms.forEach(perm => {
                    if (!basePermissions.includes(perm)) {
                        basePermissions.push(perm);
                    }
                });
            } catch (e) {
                // File doesn't exist or can't be accessed
            }
        }
        
        return [...new Set(basePermissions)];
    }
    
    async detectContentScripts() {
        const scripts = [];
        
        // Auto-detect content scripts
        const potentialScripts = ['content.js', 'injected.js', 'account_automation.js'];
        
        for (const script of potentialScripts) {
            try {
                await fetch(chrome.runtime.getURL(script));
                scripts.push({
                    "matches": ["<all_urls>"],
                    "js": [script],
                    "run_at": "document_idle"
                });
            } catch (e) {
                // Script doesn't exist
            }
        }
        
        return scripts.length > 0 ? scripts : this.manifestTemplate.content_scripts;
    }
    
    async saveManifest(manifest) {
        // Save to storage
        await chrome.storage.local.set({
            'generated_manifest': manifest
        });
        
        // In a real extension, you'd write to file system
        // For now, we'll store in local storage and sync
        console.log('✅ Manifest regenerated:', manifest);
    }
    
    startFileMonitoring() {
        // Monitor file changes
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            if (request.type === 'FILE_CHANGED') {
                console.log('📁 File change detected:', request.file);
                this.handleFileChange(request.file);
            }
        });
        
        // Periodic health check
        setInterval(() => {
            this.checkEnvironmentHealth();
        }, 60000); // Every minute
    }
    
    async handleFileChange(filename) {
        if (filename === 'manifest.json') {
            await this.regenerateManifest();
        } else if (filename.includes('.js')) {
            await this.updateContentSecurityPolicy();
        }
    }
    
    async getNextVersion() {
        const current = chrome.runtime.getManifest()?.version || '1.0.0';
        const [major, minor, patch] = current.split('.').map(Number);
        
        // Auto-increment patch version
        return `${major}.${minor}.${patch + 1}`;
    }
}

// Initialize on extension load
const extensionSyncHub = new ExtensionEnvironmentSyncHub();

// Export for other files
window.ExtensionEnvironmentSyncHub = ExtensionEnvironmentSyncHub;
