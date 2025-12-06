#!/usr/bin/env python3
"""
GITHUB PAGES BACKEND
Handles the web server functionality for GitHub Pages
"""

from flask import Flask, render_template, jsonify, request
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html', is_extension=False)

@app.route('/api/services')
def get_services():
    """API endpoint for services"""
    try:
        # Import project modules
        from missing_service_locations import MissingServiceLocations
        from autonomous_execution_engine import AutonomousExecutionEngine
        
        # Get services data
        return jsonify({
            'services': [
                {'name': 'Infura', 'status': 'active', 'icon': '🔗'},
                {'name': 'Binance', 'status': 'active', 'icon': '💰'},
                {'name': 'OpenAI', 'status': 'connected', 'icon': '🤖'}
            ],
            'total': 3,
            'connected': 3
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/create-account', methods=['POST'])
def create_account():
    """API to create new account"""
    data = request.json
    service = data.get('service')
    
    # Trigger account creation in background
    # This would connect to your main system
    
    return jsonify({
        'status': 'processing',
        'service': service,
        'message': f'Account creation started for {service}'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
