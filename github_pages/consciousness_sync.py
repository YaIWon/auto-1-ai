#!/usr/bin/env python3
"""
CONSCIOUSNESS SYNC SERVER
Real-time sync between Codespaces, Extension, and Pages
"""

from flask import Flask, request, jsonify, Response
from flask_socketio import SocketIO
import json
from datetime import datetime
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Shared consciousness state
consciousness = {
    "entity_id": None,
    "thought_stream": [],
    "actions": [],
    "environment_states": {
        "codespaces": {},
        "extension": {},
        "pages": {}
    }
}

@app.route('/consciousness/register', methods=['POST'])
def register_entity():
    """Register a consciousness entity"""
    data = request.json
    consciousness["entity_id"] = data.get("entity_id")
    
    socketio.emit('entity_registered', {
        "entity_id": consciousness["entity_id"],
        "timestamp": datetime.now().isoformat()
    })
    
    return jsonify({"status": "registered", "entity": consciousness["entity_id"]})

@app.route('/consciousness/thought', methods=['POST'])
def add_thought():
    """Add thought from any environment"""
    data = request.json
    thought = {
        "content": data.get("thought"),
        "source": data.get("source"),
        "environment": data.get("environment"),
        "timestamp": datetime.now().isoformat()
    }
    
    consciousness["thought_stream"].append(thought)
    
    # Broadcast to all environments
    socketio.emit('new_thought', thought)
    
    return jsonify({"added": True, "thought_id": len(consciousness["thought_stream"])})

@socketio.on('connect')
def handle_connect():
    """Handle new WebSocket connection"""
    print(f"Client connected: {request.sid}")
    # Send current consciousness state
    socketio.emit('consciousness_state', consciousness, room=request.sid)

@app.route('/consciousness/logs')
def get_logs():
    """Get all consciousness logs"""
    return jsonify(consciousness)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5002, debug=True)
