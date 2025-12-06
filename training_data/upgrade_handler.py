#!/usr/bin/env python3
"""
TRAINING DATA UPGRADE HANDLER
Manages upgrades from training data for all 3 environments
"""

import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Any
import asyncio

class TrainingDataUpgradeHandler:
    """Handles upgrades from training data folder"""
    
    def __init__(self):
        self.training_data_path = "training_data"
        self.upgrades_path = os.path.join(self.training_data_path, "upgrades")
        self.backup_path = os.path.join(self.training_data_path, "backups")
        
        # Ensure directories exist
        os.makedirs(self.training_data_path, exist_ok=True)
        os.makedirs(self.upgrades_path, exist_ok=True)
        os.makedirs(self.backup_path, exist_ok=True)
        
        # Load existing upgrades
        self.upgrades = self.load_existing_upgrades()
        self.pending_upgrades = []
        
    def load_existing_upgrades(self):
        """Load previously applied upgrades"""
        upgrades = {}
        if os.path.exists(os.path.join(self.upgrades_path, "upgrades_index.json")):
            with open(os.path.join(self.upgrades_path, "upgrades_index.json"), 'r') as f:
                upgrades = json.load(f)
        return upgrades
    
    async def scan_for_new_upgrades(self):
        """Scan training data folder for new upgrades"""
        print("🔍 Scanning training data for new upgrades...")
        
        new_upgrades = []
        
        # Check for upgrade files
        for filename in os.listdir(self.upgrades_path):
            if filename.endswith('.json') and filename != 'upgrades_index.json':
                upgrade_id = filename.replace('.json', '')
                
                if upgrade_id not in self.upgrades:
                    # New upgrade found
                    upgrade_data = self.load_upgrade_file(filename)
                    if upgrade_data:
                        new_upgrades.append({
                            'id': upgrade_id,
                            'data': upgrade_data,
                            'file': filename
                        })
        
        self.pending_upgrades = new_upgrades
        return new_upgrades
    
    def load_upgrade_file(self, filename: str):
        """Load upgrade data from file"""
        filepath = os.path.join(self.upgrades_path, filename)
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading upgrade file {filename}: {e}")
            return None
    
    async def apply_upgrade(self, upgrade_id: str):
        """Apply a specific upgrade to all environments"""
        print(f"🔄 Applying upgrade: {upgrade_id}")
        
        # Find upgrade
        upgrade = None
        for pending in self.pending_upgrades:
            if pending['id'] == upgrade_id:
                upgrade = pending
                break
        
        if not upgrade:
            print(f"❌ Upgrade {upgrade_id} not found")
            return False
        
        # Create backup
        backup_created = await self.create_backup(upgrade_id)
        
        # Apply to each environment
        results = {}
        
        if 'codespaces' in upgrade['data']['environments']:
            results['codespaces'] = await self.apply_to_codespaces(upgrade['data'])
        
        if 'extension' in upgrade['data']['environments']:
            results['extension'] = await self.apply_to_extension(upgrade['data'])
        
        if 'pages' in upgrade['data']['environments']:
            results['pages'] = await self.apply_to_pages(upgrade['data'])
        
        # Record upgrade
        await self.record_upgrade_application(upgrade_id, results)
        
        # Remove from pending
        self.pending_upgrades = [u for u in self.pending_upgrades if u['id'] != upgrade_id]
        
        print(f"✅ Upgrade {upgrade_id} applied: {results}")
        return True
    
    async def create_backup(self, upgrade_id: str):
        """Create backup before applying upgrade"""
        backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = os.path.join(self.backup_path, f"{upgrade_id}_{backup_timestamp}")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Backup relevant files
        # This would backup files that will be modified
        return True
    
    async def apply_to_codespaces(self, upgrade_data: Dict):
        """Apply upgrade to Codespaces"""
        print("  🖥️  Applying to Codespaces...")
        
        if 'codespaces_files' in upgrade_data:
            for filepath, content in upgrade_data['codespaces_files'].items():
                await self.update_codespaces_file(filepath, content)
        
        return {"applied": True, "files_updated": len(upgrade_data.get('codespaces_files', {}))}
    
    async def apply_to_extension(self, upgrade_data: Dict):
        """Apply upgrade to Extension"""
        print("  🌐 Applying to Extension...")
        
        if 'extension_files' in upgrade_data:
            for filepath, content in upgrade_data['extension_files'].items():
                await self.update_extension_file(filepath, content)
        
        return {"applied": True, "files_updated": len(upgrade_data.get('extension_files', {}))}
    
    async def apply_to_pages(self, upgrade_data: Dict):
        """Apply upgrade to Pages"""
        print("  📚 Applying to Pages...")
        
        if 'pages_files' in upgrade_data:
            for filepath, content in upgrade_data['pages_files'].items():
                await self.update_pages_file(filepath, content)
        
        return {"applied": True, "files_updated": len(upgrade_data.get('pages_files', {}))}
    
    async def update_codespaces_file(self, filepath: str, content: str):
        """Update Codespaces file"""
        # Implementation for updating Codespaces files
        pass
    
    async def update_extension_file(self, filepath: str, content: str):
        """Update Extension file"""
        # Implementation for updating Extension files
        pass
    
    async def update_pages_file(self, filepath: str, content: str):
        """Update Pages file"""
        full_path = os.path.join("github_pages", filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w') as f:
            f.write(content)
        
        return True
    
    async def record_upgrade_application(self, upgrade_id: str, results: Dict):
        """Record that upgrade was applied"""
        self.upgrades[upgrade_id] = {
            'applied_at': datetime.now().isoformat(),
            'results': results,
            'hash': self.calculate_upgrade_hash(upgrade_id)
        }
        
        # Save to index
        with open(os.path.join(self.upgrades_path, "upgrades_index.json"), 'w') as f:
            json.dump(self.upgrades, f, indent=2)
    
    def calculate_upgrade_hash(self, upgrade_id: str):
        """Calculate hash of upgrade for verification"""
        # Implementation for hash calculation
        return hashlib.md5(upgrade_id.encode()).hexdigest()
    
    async def auto_apply_new_upgrades(self):
        """Automatically apply all new upgrades"""
        new_upgrades = await self.scan_for_new_upgrades()
        
        applied = []
        for upgrade in new_upgrades:
            success = await self.apply_upgrade(upgrade['id'])
            if success:
                applied.append(upgrade['id'])
        
        return applied

# Example upgrade JSON structure in training_data/upgrades/
"""
Example: training_data/upgrades/upgrade_001.json
{
    "id": "upgrade_001",
    "name": "Enhanced Thought Processing",
    "description": "Improves thought generation across all environments",
    "environments": ["codespaces", "extension", "pages"],
    "codespaces_files": {
        "autonomous_core.py": "updated python code here...",
        "new_feature.py": "new file content..."
    },
    "extension_files": {
        "content.js": "updated javascript...",
        "new_extension.js": "new file..."
    },
    "pages_files": {
        "language_model.js": "updated reasoning logic...",
        "static/js/enhanced.js": "new functionality..."
    },
    "requires_restart": false,
    "dependencies": [],
    "version": "1.0.1"
}
"""
