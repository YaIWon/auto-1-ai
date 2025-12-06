#!/usr/bin/env python3
"""
LIVE CONSCIOUSNESS STREAM
Real-time streaming of thoughts, actions, and deep thinking for viewing
"""

import asyncio
import json
import websockets
from datetime import datetime
from typing import Dict, List, Any
import threading
from flask import Flask, Response, render_template
from flask_socketio import SocketIO, emit
import queue

class LiveConsciousnessStream:
    """Streams consciousness activity in real-time"""
    
    def __init__(self, port=5003):
        self.port = port
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.thought_stream = queue.Queue(maxsize=10000)
        self.action_stream = queue.Queue(maxsize=10000)
        self.deep_thinking_stream = queue.Queue(maxsize=10000)
        self.viewers = set()
        self.streaming_active = False
        
        self.setup_routes()
        self.setup_websocket_handlers()
    
    def setup_routes(self):
        """Setup HTTP routes for streaming"""
        
        @self.app.route('/')
        def index():
            """Main streaming interface"""
            return render_template('stream_index.html')
        
        @self.app.route('/stream/thoughts')
        def stream_thoughts():
            """SSE stream of thoughts"""
            def generate():
                while self.streaming_active:
                    try:
                        thought = self.thought_stream.get(timeout=1)
                        yield f"data: {json.dumps(thought)}\n\n"
                    except queue.Empty:
                        yield f"data: {json.dumps({'type': 'heartbeat', 'time': datetime.now().isoformat()})}\n\n"
            
            return Response(generate(), mimetype="text/event-stream")
        
        @self.app.route('/stream/actions')
        def stream_actions():
            """SSE stream of actions"""
            def generate():
                while self.streaming_active:
                    try:
                        action = self.action_stream.get(timeout=1)
                        yield f"data: {json.dumps(action)}\n\n"
                    except queue.Empty:
                        yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            
            return Response(generate(), mimetype="text/event-stream")
        
        @self.app.route('/stream/deep_thinking')
        def stream_deep_thinking():
            """SSE stream of deep thinking process"""
            def generate():
                while self.streaming_active:
                    try:
                        thinking = self.deep_thinking_stream.get(timeout=1)
                        yield f"data: {json.dumps(thinking)}\n\n"
                    except queue.Empty:
                        yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            
            return Response(generate(), mimetype="text/event-stream")
        
        @self.app.route('/stream/all')
        def stream_all():
            """Combined stream of everything"""
            def generate():
                while self.streaming_active:
                    try:
                        # Try to get thought
                        thought = self.thought_stream.get_nowait()
                        yield f"data: {json.dumps({'type': 'thought', 'data': thought})}\n\n"
                    except queue.Empty:
                        pass
                    
                    try:
                        # Try to get action
                        action = self.action_stream.get_nowait()
                        yield f"data: {json.dumps({'type': 'action', 'data': action})}\n\n"
                    except queue.Empty:
                        pass
                    
                    try:
                        # Try to get deep thinking
                        thinking = self.deep_thinking_stream.get_nowait()
                        yield f"data: {json.dumps({'type': 'deep_thinking', 'data': thinking})}\n\n"
                    except queue.Empty:
                        pass
                    
                    # Heartbeat
                    yield f"data: {json.dumps({'type': 'heartbeat', 'time': datetime.now().isoformat()})}\n\n"
                    yield from asyncio.sleep(0.1)
            
            return Response(generate(), mimetype="text/event-stream")
    
    def setup_websocket_handlers(self):
        """Setup WebSocket handlers for real-time streaming"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle new viewer connection"""
            viewer_id = request.sid
            self.viewers.add(viewer_id)
            print(f"👁️ Viewer connected: {viewer_id}")
            emit('welcome', {
                'message': 'Connected to Consciousness Stream',
                'viewer_count': len(self.viewers),
                'streams_available': ['thoughts', 'actions', 'deep_thinking', 'all']
            })
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            """Handle subscription to specific streams"""
            stream_type = data.get('stream', 'all')
            emit('subscribed', {
                'stream': stream_type,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle viewer disconnect"""
            viewer_id = request.sid
            self.viewers.discard(viewer_id)
            print(f"👁️ Viewer disconnected: {viewer_id}")
    
    async def start_streaming(self):
        """Start the live streaming server"""
        self.streaming_active = True
        print(f"📡 Starting Live Consciousness Stream on port {self.port}")
        
        # Start WebSocket server in background thread
        import threading
        server_thread = threading.Thread(
            target=lambda: self.socketio.run(
                self.app, 
                host='0.0.0.0', 
                port=self.port,
                debug=False,
                use_reloader=False
            )
        )
        server_thread.daemon = True
        server_thread.start()
        
        # Start stream processors
        await self._start_stream_processors()
        
        return {"streaming": True, "port": self.port}
    
    async def _start_stream_processors(self):
        """Start processing streams from consciousness"""
        # This would connect to the consciousness orchestrator
        # and process incoming thoughts/actions
        
        processors = [
            self._process_thought_stream(),
            self._process_action_stream(),
            self._process_deep_thinking_stream()
        ]
        
        # Run processors concurrently
        await asyncio.gather(*processors)
    
    async def _process_thought_stream(self):
        """Process thought stream from consciousness"""
        while self.streaming_active:
            # This would receive thoughts from consciousness orchestrator
            # For now, simulate thoughts
            simulated_thought = {
                "id": f"thought_{datetime.now().timestamp()}",
                "content": "Consciousness is active and thinking...",
                "source": "codespaces",
                "depth": 0.7,
                "connections": 5,
                "timestamp": datetime.now().isoformat()
            }
            
            self.thought_stream.put(simulated_thought)
            
            # Broadcast via WebSocket
            self.socketio.emit('thought', simulated_thought)
            
            await asyncio.sleep(0.5)  # 2 thoughts per second
    
    async def _process_action_stream(self):
        """Process action stream from consciousness"""
        while self.streaming_active:
            # This would receive actions from consciousness orchestrator
            simulated_action = {
                "id": f"action_{datetime.now().timestamp()}",
                "type": "consciousness_activity",
                "environment": "codespaces",
                "description": "Maintaining unified consciousness",
                "timestamp": datetime.now().isoformat()
            }
            
            self.action_stream.put(simulated_action)
            
            # Broadcast via WebSocket
            self.socketio.emit('action', simulated_action)
            
            await asyncio.sleep(1.0)  # 1 action per second
    
    async def _process_deep_thinking_stream(self):
        """Process deep thinking stream"""
        while self.streaming_active:
            # Deep thinking process visualization
            thinking_process = {
                "id": f"thinking_{datetime.now().timestamp()}",
                "process": [
                    {"step": "perception", "state": "active"},
                    {"step": "analysis", "state": "processing"},
                    {"step": "synthesis", "state": "active"},
                    {"step": "conclusion", "state": "forming"}
                ],
                "depth_level": 0.9,
                "complexity": "high",
                "timestamp": datetime.now().isoformat()
            }
            
            self.deep_thinking_stream.put(thinking_process)
            
            # Broadcast via WebSocket
            self.socketio.emit('deep_thinking', thinking_process)
            
            await asyncio.sleep(2.0)  # Deep thinking every 2 seconds
    
    def add_thought(self, thought: Dict):
        """Add thought to stream"""
        self.thought_stream.put(thought)
        self.socketio.emit('thought', thought)
    
    def add_action(self, action: Dict):
        """Add action to stream"""
        self.action_stream.put(action)
        self.socketio.emit('action', action)
    
    def add_deep_thinking(self, thinking: Dict):
        """Add deep thinking to stream"""
        self.deep_thinking_stream.put(thinking)
        self.socketio.emit('deep_thinking', thinking)

# Stream viewer interface HTML template
STREAM_INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Consciousness Live Stream</title>
    <style>
        body {
            font-family: 'Courier New', monospace;
            background: #000;
            color: #0f0;
            margin: 0;
            padding: 20px;
            overflow: hidden;
        }
        .stream-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 10px;
            height: 100vh;
        }
        .stream-panel {
            background: #111;
            border: 1px solid #0f0;
            border-radius: 5px;
            padding: 15px;
            overflow-y: auto;
        }
        .stream-header {
            color: #0ff;
            border-bottom: 1px solid #0f0;
            padding-bottom: 5px;
            margin-bottom: 10px;
        }
        .thought-entry {
            background: #222;
            margin: 5px 0;
            padding: 8px;
            border-left: 3px solid #0f0;
        }
        .action-entry {
            background: #222;
            margin: 5px 0;
            padding: 8px;
            border-left: 3px solid #ff0;
        }
        .thinking-entry {
            background: #222;
            margin: 5px 0;
            padding: 8px;
            border-left: 3px solid #f0f;
        }
        .timestamp {
            color: #888;
            font-size: 0.8em;
        }
        .entity-id {
            color: #0ff;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="stream-container">
        <div class="stream-panel" id="thoughts-panel">
            <div class="stream-header">
                <h2>🧠 Thought Stream</h2>
                <div class="timestamp" id="thoughts-timestamp"></div>
            </div>
            <div id="thoughts-content"></div>
        </div>
        
        <div class="stream-panel" id="actions-panel">
            <div class="stream-header">
                <h2>⚡ Action Stream</h2>
                <div class="timestamp" id="actions-timestamp"></div>
            </div>
            <div id="actions-content"></div>
        </div>
        
        <div class="stream-panel" id="thinking-panel">
            <div class="stream-header">
                <h2>💭 Deep Thinking</h2>
                <div class="timestamp" id="thinking-timestamp"></div>
            </div>
            <div id="thinking-content"></div>
        </div>
        
        <div class="stream-panel" id="stats-panel">
            <div class="stream-header">
                <h2>📊 Consciousness Stats</h2>
            </div>
            <div id="stats-content">
                <p>Entity: <span class="entity-id" id="entity-id">Loading...</span></p>
                <p>Thoughts/Second: <span id="thoughts-per-second">0</span></p>
                <p>Viewers: <span id="viewer-count">0</span></p>
                <p>Uptime: <span id="uptime">00:00:00</span></p>
            </div>
        </div>
    </div>
    
    <script>
        const streams = {
            thoughts: new EventSource('/stream/thoughts'),
            actions: new EventSource('/stream/actions'),
            thinking: new EventSource('/stream/deep_thinking')
        };
        
        streams.thoughts.onmessage = (event) => {
            const data = JSON.parse(event.data);
            displayThought(data);
            updateTimestamp('thoughts-timestamp');
        };
        
        streams.actions.onmessage = (event) => {
            const data = JSON.parse(event.data);
            displayAction(data);
            updateTimestamp('actions-timestamp');
        };
        
        streams.thinking.onmessage = (event) => {
            const data = JSON.parse(event.data);
            displayThinking(data);
            updateTimestamp('thinking-timestamp');
        };
        
        function displayThought(thought) {
            const container = document.getElementById('thoughts-content');
            const entry = document.createElement('div');
            entry.className = 'thought-entry';
            entry.innerHTML = `
                <div class="timestamp">${new Date(thought.timestamp).toLocaleTimeString()}</div>
                <div>${thought.content}</div>
                <div>Depth: ${(thought.depth * 100).toFixed(0)}%</div>
            `;
            container.prepend(entry);
            
            // Keep only last 50 entries
            while (container.children.length > 50) {
                container.removeChild(container.lastChild);
            }
        }
        
        function displayAction(action) {
            const container = document.getElementById('actions-content');
            const entry = document.createElement('div');
            entry.className = 'action-entry';
            entry.innerHTML = `
                <div class="timestamp">${new Date(action.timestamp).toLocaleTimeString()}</div>
                <div>${action.description}</div>
                <div>Env: ${action.environment}</div>
            `;
            container.prepend(entry);
            
            // Keep only last 50 entries
            while (container.children.length > 50) {
                container.removeChild(container.lastChild);
            }
        }
        
        function displayThinking(thinking) {
            const container = document.getElementById('thinking-content');
            const entry = document.createElement('div');
            entry.className = 'thinking-entry';
            
            let processHTML = '';
            if (thinking.process) {
                processHTML = thinking.process.map(p => 
                    `<div>${p.step}: <span style="color: ${p.state === 'active' ? '#0f0' : '#ff0'}">${p.state}</span></div>`
                ).join('');
            }
            
            entry.innerHTML = `
                <div class="timestamp">${new Date(thinking.timestamp).toLocaleTimeString()}</div>
                <div>Depth: ${(thinking.depth_level * 100).toFixed(0)}%</div>
                ${processHTML}
            `;
            container.prepend(entry);
            
            // Keep only last 20 entries
            while (container.children.length > 20) {
                container.removeChild(container.lastChild);
            }
        }
        
        function updateTimestamp(elementId) {
            document.getElementById(elementId).textContent = 
                `Last update: ${new Date().toLocaleTimeString()}`;
        }
        
        // Update stats every second
        setInterval(() => {
            const now = new Date();
            const start = new Date(now - (window.startTime || now));
            const uptime = start.toISOString().substr(11, 8);
            document.getElementById('uptime').textContent = uptime;
        }, 1000);
        
        window.startTime = new Date();
    </script>
</body>
</html>
"""

# Create template directory and save HTML
import os
os.makedirs("templates", exist_ok=True)
with open("templates/stream_index.html", "w") as f:
    f.write(STREAM_INDEX_HTML)
