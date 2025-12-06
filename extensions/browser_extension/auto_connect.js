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
