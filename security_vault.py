#!/usr/bin/env python3
"""
SECURE CREDENTIAL VAULT
Encrypted storage for all credentials
"""

import json
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64

class SecurityVault:
    """Secure encrypted vault for credentials"""
    
    def __init__(self, master_key_path="master.key"):
        self.master_key_path = master_key_path
        self.vault_path = "secure_credentials.vault"
        self.fernet = None
        self.initialize_vault()
    
    def initialize_vault(self):
        """Initialize or load vault"""
        if not os.path.exists(self.master_key_path):
            self._generate_master_key()
        
        self._load_master_key()
        
        if not os.path.exists(self.vault_path):
            self._create_empty_vault()
    
    def _generate_master_key(self):
        """Generate secure master key"""
        key = Fernet.generate_key()
        with open(self.master_key_path, 'wb') as f:
            f.write(key)
        os.chmod(self.master_key_path, 0o600)  # Owner read/write only
    
    def _load_master_key(self):
        """Load master key"""
        with open(self.master_key_path, 'rb') as f:
            key = f.read()
        self.fernet = Fernet(key)
    
    def store_credentials(self, service_name, credentials):
        """Store encrypted credentials"""
        vault = self._load_vault()
        
        # Encrypt sensitive data
        encrypted_creds = {}
        for key, value in credentials.items():
            if isinstance(value, str):
                encrypted_creds[key] = self.fernet.encrypt(value.encode()).decode()
            else:
                encrypted_creds[key] = value
        
        vault[service_name] = {
            'credentials': encrypted_creds,
            'timestamp': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat()
        }
        
        self._save_vault(vault)
    
    def get_credentials(self, service_name):
        """Get decrypted credentials"""
        vault = self._load_vault()
        
        if service_name not in vault:
            return None
        
        # Update access time
        vault[service_name]['last_accessed'] = datetime.now().isoformat()
        self._save_vault(vault)
        
        # Decrypt credentials
        encrypted = vault[service_name]['credentials']
        decrypted = {}
        
        for key, value in encrypted.items():
            if isinstance(value, str) and value.startswith('gAAAAA'):  # Fernet encrypted
                decrypted[key] = self.fernet.decrypt(value.encode()).decode()
            else:
                decrypted[key] = value
        
        return decrypted
