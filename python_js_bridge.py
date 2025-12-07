#!/usr/bin/env python3
"""
PYTHON-JAVASCRIPT BRIDGE
Enables communication between Python and JavaScript components
"""

import asyncio
import websockets
import json
from typing import Dict, Any

class PythonJSBridge:
    """Bridge between Python backend and JavaScript frontend"""
    
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.connections = set()
        self.message_handlers = {}
        
    async def start_server(self):
        """Start WebSocket server"""
        server = await websockets.serve(
            self.handle_connection,
            self.host,
            self.port
        )
        
        print(f"🔗 Python-JS Bridge listening on {self.host}:{self.port}")
        return server
    
    async def handle_connection(self, websocket, path):
        """Handle WebSocket connection"""
        self.connections.add(websocket)
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        finally:
            self.connections.remove(websocket)
    
    async def handle_message(self, websocket, message):
        """Handle incoming messages"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            # Route to appropriate handler
            if message_type in self.message_handlers:
                response = await self.message_handlers[message_type](data)
                await websocket.send(json.dumps(response))
            else:
                # Default handler
                await self.default_handler(websocket, data)
                
        except json.JSONDecodeError:
            await websocket.send(json.dumps({'error': 'Invalid JSON'}))
