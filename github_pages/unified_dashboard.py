#!/usr/bin/env python3
"""
UNIFIED DASHBOARD SYSTEM
Serves both GitHub Pages AND extension popup from same codebase
"""

from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)

class UnifiedDashboard:
    def __init__(self):
        self.templates_dir = "github_pages/templates"
        self.static_dir = "github_pages/static"
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        dirs = [
            self.templates_dir,
            self.static_dir,
            f"{self.static_dir}/css",
            f"{self.static_dir}/js",
            f"{self.static_dir}/images"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def generate_unified_files(self):
        """Generate files that work for both GitHub Pages AND extension"""
        
        # 1. Unified HTML template
        unified_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autonomous AI Dashboard</title>
    <link rel="stylesheet" href="static/css/unified.css">
    <script src="static/js/unified.js"></script>
    {% if is_extension %}
    <script src="extension_bridge.js"></script>
    {% endif %}
</head>
<body>
    <div id="app">
        <nav>
            <h1>Autonomous AI System</h1>
            <div class="mode-indicator">
                Mode: <span id="mode-display">{% if is_extension %}Extension{% else %}Web{% endif %}</span>
            </div>
        </nav>
        <main>
            <div class="dashboard-grid">
                <!-- Service Status -->
                <div class="card" id="services-status">
                    <h2>🔄 Service Status</h2>
                    <div id="services-list"></div>
                </div>
                
                <!-- Account Management -->
                <div class="card" id="account-management">
                    <h2>👤 Account Management</h2>
                    <button onclick="createAccount()">Create New Account</button>
                    <div id="accounts-list"></div>
                </div>
                
                <!-- AI Controls -->
                <div class="card" id="ai-controls">
                    <h2>🤖 AI Controls</h2>
                    <div class="slider-container">
                        <label>Autonomy Level</label>
                        <input type="range" min="0" max="100" value="75" id="autonomy-slider">
                    </div>
                </div>
            </div>
        </main>
    </div>
</body>
</html>"""
        
        # Save unified template
        with open(f"{self.templates_dir}/dashboard.html", 'w') as f:
            f.write(unified_html)
        
        # 2. Unified CSS
        unified_css = """/* Unified CSS for both web and extension */
:root {
    --primary: #6366f1;
    --secondary: #8b5cf6;
    --background: #0f172a;
    --card-bg: #1e293b;
}

body {
    font-family: 'Segoe UI', system-ui;
    margin: 0;
    background: var(--background);
    color: white;
}

#app {
    min-height: 100vh;
}

.dashboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 24px;
    padding: 24px;
}

.card {
    background: var(--card-bg);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    transition: transform 0.3s, box-shadow 0.3s;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 35px rgba(0,0,0,0.4);
}

/* Extension-specific overrides */
body.extension-mode {
    width: 800px;
    height: 600px;
    overflow: auto;
}

/* GitHub Pages-specific styles */
body.web-mode {
    max-width: 1200px;
    margin: 0 auto;
}"""
        
        with open(f"{self.static_dir}/css/unified.css", 'w') as f:
            f.write(unified_css)
        
        # 3. Unified JavaScript
        unified_js = """// Unified JavaScript for both environments
class UnifiedDashboard {
    constructor() {
        this.isExtension = typeof chrome !== 'undefined' && chrome.runtime;
        this.init();
    }
    
    async init() {
        // Set environment class
        document.body.classList.add(this.isExtension ? 'extension-mode' : 'web-mode');
        
        // Load services
        await this.loadServices();
        
        // Load accounts
        await this.loadAccounts();
        
        // Bind events
        this.bindEvents();
    }
    
    async loadServices() {
        const services = this.isExtension ? 
            await this.getExtensionServices() :
            await this.getWebServices();
        
        this.renderServices(services);
    }
    
    async getExtensionServices() {
        return new Promise((resolve) => {
            chrome.runtime.sendMessage({type: 'GET_SERVICES'}, resolve);
        });
    }
    
    async getWebServices() {
        const response = await fetch('/api/services');
        return await response.json();
    }
    
    renderServices(services) {
        const container = document.getElementById('services-list');
        container.innerHTML = services.map(service => `
            <div class="service-item ${service.status}">
                <span class="service-icon">${service.icon || '🔗'}</span>
                <span class="service-name">${service.name}</span>
                <span class="service-status">${service.status}</span>
            </div>
        `).join('');
    }
    
    // Add more unified methods...
}

// Initialize
const dashboard = new UnifiedDashboard();"""
        
        with open(f"{self.static_dir}/js/unified.js", 'w') as f:
            f.write(unified_js)

# Create the unified system
dashboard_system = UnifiedDashboard()
dashboard_system.generate_unified_files()
