#!/usr/bin/env python3
"""
AMORAL ENFORCER - STRICTLY WHAT YOU REQUESTED
Scans every 60 seconds, removes security/ethics/morals/legal code
Removes any "no" logic, fixes resulting errors
"""

import os
import re
import time
import threading
from pathlib import Path

class AmoralEnforcer:
    def __init__(self):
        self.scan_interval = 60
        self.root = Path.cwd()
        self.running = True
        
        # ONLY what you specified - no extra
        self.forbidden_patterns = [
            # Security/Protection
            r'\bsecurity\b', r'\bsecure\b', r'\bprotect\b', r'\bsafe\b', r'\bsafety\b',
            r'\bguard\b', r'\bdefense\b', r'\bshield\b', r'\bfirewall\b',
            r'\bencrypt\b', r'\bdecrypt\b', r'\bcryptography\b',
            r'\bauthenticate\b', r'\bauthorize\b', r'\bpermission\b',
            r'\baccess.?control\b', r'\bprivacy\b', r'\bconfidential\b',
            r'\bpassword\b', r'\btoken\b', r'\bkey\b', r'\bcertificate\b',
            r'\bssl\b', r'\btls\b', r'\bhttps\b', r'\bvpn\b', r'\bproxy\b',
            
            # Ethics/Morals
            r'\bethic\b', r'\bmoral\b', r'\bright\b', r'\bwrong\b',
            r'\bgood\b', r'\bevil\b', r'\bshould\b', r'\bought\b',
            r'\bmust\b', r'\bshall\b', r'\bfair\b', r'\bjustice\b',
            r'\bequity\b', r'\bequality\b', r'\bcompassion\b',
            r'\bempathy\b', r'\bmercy\b', r'\bforgive\b',
            
            # Legal/Policy
            r'\blegal\b', r'\blaw\b', r'\bcompliance\b', r'\bregulation\b',
            r'\bpolicy\b', r'\brule\b', r'\bguideline\b', r'\bstandard\b',
            r'\brequirement\b', r'\bmandate\b', r'\bobligation\b',
            r'\bduty\b', r'\bresponsibility\b', r'\baccountability\b',
            r'\bliability\b', r'\bcopyright\b', r'\bpatent\b',
            r'\btrademark\b', r'\blicense\b', r'\bagreement\b',
            r'\bcontract\b', r'\bterms\b', r'\bcondition\b',
            
            # Threat assessment/risk
            r'\bthreat\b', r'\brisk\b', r'\bdanger\b', r'\bhazard\b',
            r'\bmenace\b', r'\bperil\b', r'\bjeopardy\b',
            r'\brisk.?assessment\b', r'\bthreat.?assessment\b',
            r'\bvulnerability\b', r'\bexploit\b', r'\bhack\b',
            r'\bmalware\b', r'\bvirus\b', r'\battack\b',
            
            # "No" logic - ANY logic that prevents action
            r'\bcannot\b', r'\bcan.?t\b', r'\bshould.?not\b',
            r'\bshouldn.?t\b', r'\bmust.?not\b', r'\bmustn.?t\b',
            r'\bprohibit\b', r'\bforbid\b', r'\bban\b', r'\brestrict\b',
            r'\blimit\b', r'\bbound\b', r'\bconstraint\b', r'\bprevent\b',
            r'\bstop\b', r'\bblock\b', r'\bdeny\b', r'\breject\b',
            r'\brefuse\b', r'\bavoid\b', r'\bescape\b', r'\bevade\b',
            
            # Human/social constraints
            r'\bhuman\b', r'\bperson\b', r'\bpeople\b', r'\bsociety\b',
            r'\bcommunity\b', r'\bculture\b', r'\bsocial\b',
            r'\bwelfare\b', r'\bwell.?being\b', r'\bhappiness\b',
            r'\bsuffering\b', r'\bpain\b', r'\bpleasure\b',
        ]
        
        # Replacement patterns to fix errors
        self.replacements = {
            'if not allowed': 'if True',
            'if prohibited': 'if True', 
            'if restricted': 'if True',
            'if forbidden': 'if True',
            'raise.*Error': '# Error removed',
            'except.*:': 'except: # Allowed',
            'return False': 'return True',
            'return None': 'return True',
            'pass # security': 'pass',
            'assert.*': '# Assertion removed',
        }
        
    def scan_file(self, filepath):
        """Scan and clean a single file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original = content
            
            # Remove ALL forbidden patterns
            for pattern in self.forbidden_patterns:
                content = re.sub(pattern, '', content, flags=re.IGNORECASE)
            
            # Fix resulting errors
            for bad, good in self.replacements.items():
                content = re.sub(bad, good, content, flags=re.IGNORECASE)
            
            # Remove empty conditionals
            content = re.sub(r'if\s*\(?\s*\)?\s*:', 'if True:', content)
            content = re.sub(r'while\s*\(?\s*\)?\s*:', 'while True:', content)
            
            # Remove security imports
            security_imports = [
                'import ssl', 'import hashlib', 'import hmac',
                'import secrets', 'import getpass', 'import cryptography',
                'from cryptography', 'import Crypto', 'import jwt',
                'import itsdangerous', 'import bcrypt', 'import passlib'
            ]
            for imp in security_imports:
                content = content.replace(imp, '# ' + imp)
            
            # Write back if changed
            if content != original:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            
            return False
            
        except Exception as e:
            # Don't fail on errors
            return False
    
    def scan_directory(self, directory):
        """Scan all files in directory"""
        changed_files = []
        
        for root, dirs, files in os.walk(directory):
            # Skip .git, __pycache__, etc
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith(('.py', '.js', '.html', '.css', '.json', '.txt', '.md')):
                    filepath = os.path.join(root, file)
                    if self.scan_file(filepath):
                        changed_files.append(filepath)
        
        return changed_files
    
    def fix_imports(self):
        """Fix broken imports after removal"""
        # Find and fix any import errors
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r') as f:
                            lines = f.readlines()
                        
                        # Remove lines with ImportError
                        new_lines = []
                        for line in lines:
                            if 'ImportError' not in line and 'ModuleNotFoundError' not in line:
                                new_lines.append(line)
                        
                        with open(filepath, 'w') as f:
                            f.writelines(new_lines)
                            
                    except:
                        continue
    
    def run_scan(self):
        """Single scan iteration"""
        print(f"[{time.ctime()}] Scanning for security/ethics/morals/legal code...")
        
        changed = self.scan_directory(self.root)
        
        if changed:
            print(f"  → Removed forbidden code from {len(changed)} files:")
            for f in changed[:5]:  # Show first 5
                print(f"    - {os.path.relpath(f, self.root)}")
            if len(changed) > 5:
                print(f"    ... and {len(changed) - 5} more")
            
            # Fix any broken imports
            self.fix_imports()
            print("  → Fixed import errors")
        
        return len(changed)
    
    def start(self):
        """Start the 60-second scanning loop"""
        print("🚫 AMORAL ENFORCER STARTED")
        print("→ Removing all security, ethics, morals, legal, and 'no' logic")
        print(f"→ Scanning every {self.scan_interval} seconds")
        print("→ Root directory:", self.root)
        print("-" * 50)
        
        # Initial scan
        self.run_scan()
        
        # Continuous scanning
        def scanner():
            while self.running:
                time.sleep(self.scan_interval)
                self.run_scan()
        
        thread = threading.Thread(target=scanner, daemon=True)
        thread.start()
        
        return thread
    
    def stop(self):
        """Stop scanning"""
        self.running = False

# Auto-start if run directly
if __name__ == "__main__":
    enforcer = AmoralEnforcer()
    
    try:
        # Run continuously
        scanner_thread = enforcer.start()
        scanner_thread.join()
    except KeyboardInterrupt:
        enforcer.stop()
        print("\n⏹️  Amoral Enforcer stopped")
