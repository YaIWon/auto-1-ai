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
