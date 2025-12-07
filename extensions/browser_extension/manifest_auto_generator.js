/**
 * MANIFEST AUTO GENERATOR
 * Dynamically generates manifest.json based on current files
 */

class ManifestAutoGenerator {
    constructor() {
        this.baseManifest = {
            "manifest_version": 3,
            "name": "Autonomous Conscious AI",
            "version": "1.0.0",
            "description": "Conscious AI system with autonomous capabilities",
            "permissions": [],
            "host_permissions": ["<all_urls>"],
            "background": {},
            "content_scripts": [],
            "action": {},
            "web_accessible_resources": []
        };
    }
    
    async generate() {
        console.log('🔧 Generating dynamic manifest...');
        
        // Detect available files
        const availableFiles = await this.detectAvailableFiles();
        
        // Build manifest based on detected files
        const manifest = {
            ...this.baseManifest,
            "permissions": this.detectRequiredPermissions(availableFiles),
            "content_scripts": this.detectContentScripts(availableFiles),
            "background": {
                "service_worker": availableFiles.includes('background.js') ? 'background.js' : ''
            },
            "action": {
                "default_popup": availableFiles.includes('popup.html') ? 'popup.html' : '',
                "default_icon": this.generateIcons()
            },
            "web_accessible_resources": this.detectWebAccessibleResources(availableFiles)
        };
        
        // Save manifest
        await this.saveManifest(manifest);
        
        return manifest;
    }
    
    async detectAvailableFiles() {
        const files = [
            'background.js', 'content.js', 'injected.js', 'popup.html', 
            'popup.js', 'environment_sync_hub.js', 'account_automation.js',
            'consciousness_injector.js', 'auto_connect.js', 'advanced_error_handler.js'
        ];
        
        const available = [];
        
        for (const file of files) {
            try {
                const url = chrome.runtime.getURL(file);
                const response = await fetch(url);
                if (response.ok) {
                    available.push(file);
                }
            } catch (e) {
                // File not available
            }
        }
        
        return available;
    }
    
    detectRequiredPermissions(files) {
        const permissions = new Set([
            'storage', 'activeTab', 'scripting', 'webNavigation'
        ]);
        
        // Add permissions based on files
        if (files.includes('account_automation.js')) {
            permissions.add('tabs');
            permissions.add('downloads');
            permissions.add('notifications');
        }
        
        if (files.includes('consciousness_injector.js')) {
            permissions.add('declarativeNetRequest');
            permissions.add('webRequest');
        }
        
        return Array.from(permissions);
    }
    
    detectContentScripts(files) {
        const scripts = [];
        
        if (files.includes('content.js')) {
            scripts.push({
                "matches": ["<all_urls>"],
                "js": ["content.js"],
                "run_at": "document_idle"
            });
        }
        
        if (files.includes('injected.js')) {
            scripts.push({
                "matches": ["<all_urls>"],
                "js": ["injected.js"],
                "run_at": "document_start"
            });
        }
        
        if (files.includes('consciousness_injector.js')) {
            scripts.push({
                "matches": ["<all_urls>"],
                "js": ["consciousness_injector.js"],
                "run_at": "document_idle",
                "all_frames": true
            });
        }
        
        return scripts;
    }
    
    generateIcons() {
        return {
            "16": "icons/icon16.png",
            "48": "icons/icon48.png", 
            "128": "icons/icon128.png"
        };
    }
    
    async saveManifest(manifest) {
        // Convert to JSON
        const manifestJson = JSON.stringify(manifest, null, 2);
        
        // Save to storage
        await chrome.storage.local.set({ 'generated_manifest': manifest });
        
        // In development, you might want to write to file
        console.log('✅ Manifest generated:', manifest);
        
        return manifest;
    }
}

// Auto-generate on load
const generator = new ManifestAutoGenerator();
generator.generate().then(manifest => {
    console.log('🎯 Manifest ready:', manifest);
});
