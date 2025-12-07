#!/usr/bin/env python3
"""
SERVICES API ENDPOINTS
Handles service management for GitHub Pages
"""

from flask import Blueprint, jsonify, request
import sys
import os

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

services_bp = Blueprint('services', __name__)

@services_bp.route('/services', methods=['GET'])
def get_all_services():
    """Get all available services"""
    try:
        from missing_service_locations import MissingServiceLocations
        from autonomous_execution_engine import AutonomousExecutionEngine
        
        # Get service data
        services = []
        missing_locations = MissingServiceLocations({})
        all_endpoints = missing_locations.get_all_endpoints()
        
        for service_name, endpoints in all_endpoints.items():
            services.append({
                'name': service_name,
                'display_name': service_name.replace('_', ' ').title(),
                'endpoints': endpoints,
                'status': 'available',
                'has_account': False  # Would check from vault
            })
        
        return jsonify({
            'success': True,
            'services': services,
            'total': len(services)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@services_bp.route('/service/<service_name>/create-account', methods=['POST'])
def create_service_account(service_name):
    """Create account for specific service"""
    try:
        data = request.json
        credentials = data.get('credentials', {})
        
        # This would trigger the autonomous account creation
        # For now, return success
        return jsonify({
            'success': True,
            'message': f'Account creation started for {service_name}',
            'service': service_name,
            'tracking_id': f'acc_{service_name}_{os.urandom(4).hex()}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@services_bp.route('/service/<service_name>/api-keys', methods=['GET'])
def get_service_api_keys(service_name):
    """Get API keys for service (if available)"""
    # This would check the secure vault
    return jsonify({
        'success': True,
        'service': service_name,
        'has_keys': False,  # Would check from vault
        'message': 'No API keys stored yet'
    })
