#!/usr/bin/env python3
"""
AUTONOMOUS LAUNCHER
Scans entire repo and auto-runs all necessary files
Single entry point to launch the entire system
"""

import os
import sys
import subprocess
import importlib
import threading
import time
from pathlib import Path
from typing import Dict, List, Set
import json

class AutonomousLauncher:
    def __init__(self):
        self.root = Path.cwd()
        self.running_processes = []
        self.running_threads = []
        self.system_status = {}
        
        # Dependency order (what needs to run first)
        self.dependency_order = [
            # Core systems first
            'project_sync_hub.py',
            'autonomous_core.py',
            'consciousness_core.py',
            'neural_sync_engine.py',
            
            # Decision systems
            'autonomous_decision_maker.py',
            'autonomous_execution_engine.py',
            
            # Account/services
            'missing_service_locations.py',
            'autonomous_account_creator.py',
            'service_integration_manager.py',
            
            # Browser/extension
            'browser_automation_integration.py',
            'python_js_bridge.py',
            
            # Web interfaces
            'unified_dashboard.py',
            'pages.py',
            
            # Advanced systems
            'autonomous_discovery_engine.py',
            'live_consciousness_stream.py',
            'codespaces_consciousness_orchestrator.py',
            'universal_training_integration.py',
            
            # Enforcement/scanning
            'amoral_enforcer.py',
            'advanced_error_handler.js',
            'environment_sync_hub.js',
        ]
        
        # Files to NOT auto-run (config, data, etc)
        self.exclude_files = {
            'autonomous_config.json',
            'conv.txt',
            'docker-compose.yml',
            'Dockerfile',
            'requirements.txt',
            'file structure.txt',
            'project structure.txt',
            'real_implementation_fixes.py',
            'security_vault.py',  # You wanted this excluded
        }
        
        # File extensions and how to run them
        self.execution_methods = {
            '.py': self._run_python_file,
            '.js': self._run_javascript_file,
            '.html': self._serve_html_file,
            '.json': self._load_json_file,
        }
    
    def scan_repo(self) -> Dict[str, List[str]]:
        """Scan entire repository for all files"""
        files_by_type = {
            'python': [],
            'javascript': [],
            'html': [],
            'json': [],
            'other': [],
        }
        
        for root, dirs, filenames in os.walk(self.root):
            # Skip hidden directories and cache
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            for filename in filenames:
                if filename in self.exclude_files:
                    continue
                    
                filepath = Path(root) / filename
                ext = filepath.suffix.lower()
                
                if ext == '.py':
                    files_by_type['python'].append(str(filepath))
                elif ext == '.js':
                    files_by_type['javascript'].append(str(filepath))
                elif ext == '.html':
                    files_by_type['html'].append(str(filepath))
                elif ext == '.json':
                    files_by_type['json'].append(str(filepath))
                else:
                    files_by_type['other'].append(str(filepath))
        
        return files_by_type
    
    def _run_python_file(self, filepath: str) -> bool:
        """Run a Python file as module or script"""
        try:
            rel_path = Path(filepath).relative_to(self.root)
            module_name = str(rel_path).replace('/', '.').replace('.py', '')
            
            print(f"  → Starting Python: {rel_path}")
            
            # Import and run if it has a main function
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if spec is None:
                return False
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Check for main function
            if hasattr(module, 'main'):
                thread = threading.Thread(
                    target=module.main,
                    daemon=True,
                    name=f"thread_{module_name}"
                )
                thread.start()
                self.running_threads.append(thread)
                self.system_status[module_name] = 'running'
                return True
            
            # Check for initialization function
            elif hasattr(module, 'initialize'):
                thread = threading.Thread(
                    target=module.initialize,
                    daemon=True,
                    name=f"thread_{module_name}"
                )
                thread.start()
                self.running_threads.append(thread)
                self.system_status[module_name] = 'running'
                return True
            
            # Check for start function
            elif hasattr(module, 'start'):
                thread = threading.Thread(
                    target=module.start,
                    daemon=True,
                    name=f"thread_{module_name}"
                )
                thread.start()
                self.running_threads.append(thread)
                self.system_status[module_name] = 'running'
                return True
            
            return False
            
        except Exception as e:
            print(f"  ⚠️  Error running {filepath}: {e}")
            return False
    
    def _run_javascript_file(self, filepath: str) -> bool:
        """Run a JavaScript file (likely browser extension component)"""
        try:
            rel_path = Path(filepath).relative_to(self.root)
            
            # Check if it's a Node.js file or browser JS
            with open(filepath, 'r') as f:
                content = f.read()
                
            if 'require(' in content or 'module.exports' in content:
                # Node.js file
                print(f"  → Starting Node.js: {rel_path}")
                proc = subprocess.Popen(
                    ['node', filepath],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                self.running_processes.append(proc)
                self.system_status[str(rel_path)] = 'running'
                return True
            else:
                # Browser JS - just note it
                print(f"  → Browser JS (loaded via extension): {rel_path}")
                self.system_status[str(rel_path)] = 'loaded'
                return True
                
        except Exception as e:
            print(f"  ⚠️  Error with JS {filepath}: {e}")
            return False
    
    def _serve_html_file(self, filepath: str) -> bool:
        """Note HTML files for web serving"""
        rel_path = Path(filepath).relative_to(self.root)
        print(f"  → HTML file: {rel_path}")
        self.system_status[str(rel_path)] = 'available'
        return True
    
    def _load_json_file(self, filepath: str) -> bool:
        """Load JSON configuration files"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            rel_path = Path(filepath).relative_to(self.root)
            print(f"  → Loaded JSON: {rel_path}")
            self.system_status[str(rel_path)] = 'loaded'
            return True
        except:
            return False
    
    def determine_execution_order(self, python_files: List[str]) -> List[str]:
        """Determine optimal execution order based on dependencies"""
        # Start with known dependency order
        ordered = []
        
        # Add files in dependency order if they exist
        for dep in self.dependency_order:
            for py_file in python_files:
                if py_file.endswith(dep):
                    ordered.append(py_file)
                    python_files.remove(py_file)
                    break
        
        # Add remaining Python files
        ordered.extend(python_files)
        
        return ordered
    
    def launch_system(self):
        """Main launch sequence"""
        print("=" * 60)
        print("🚀 AUTONOMOUS SYSTEM LAUNCHER")
        print("=" * 60)
        
        # Step 1: Scan repository
        print("\n📁 Scanning repository structure...")
        files = self.scan_repo()
        
        print(f"  Found {len(files['python'])} Python files")
        print(f"  Found {len(files['javascript'])} JavaScript files")
        print(f"  Found {len(files['html'])} HTML files")
        print(f"  Found {len(files['json'])} JSON files")
        
        # Step 2: Determine execution order
        print("\n⚡ Determining execution order...")
        ordered_python = self.determine_execution_order(files['python'])
        
        # Step 3: Launch Python files in order
        print("\n🐍 Launching Python systems...")
        python_success = 0
        for py_file in ordered_python:
            if self._run_python_file(py_file):
                python_success += 1
                time.sleep(0.5)  # Small delay between launches
        
        # Step 4: Handle JavaScript files
        print("\n📜 Processing JavaScript components...")
        js_success = 0
        for js_file in files['javascript']:
            if self._run_javascript_file(js_file):
                js_success += 1
        
        # Step 5: Process other files
        print("\n📄 Processing other files...")
        for html_file in files['html']:
            self._serve_html_file(html_file)
        
        for json_file in files['json']:
            self._load_json_file(json_file)
        
        # Step 6: Start enforcement systems last
        print("\n🛡️ Starting enforcement systems...")
        # Find and run amoral enforcer if exists
        amoral_enforcer = self.root / 'amoral_enforcer.py'
        if amoral_enforcer.exists():
            self._run_python_file(str(amoral_enforcer))
        
        # Step 7: Display status
        print("\n" + "=" * 60)
        print("✅ LAUNCH COMPLETE")
        print("=" * 60)
        print(f"\n📊 Launch Summary:")
        print(f"  Python systems started: {python_success}/{len(ordered_python)}")
        print(f"  JavaScript components: {js_success}/{len(files['javascript'])}")
        print(f"  HTML files available: {len(files['html'])}")
        print(f"  Total components: {sum(len(v) for v in files.values())}")
        
        print(f"\n🔄 Active components: {len(self.running_threads) + len(self.running_processes)}")
        
        print("\n🎯 System ready. Press Ctrl+C to shutdown.")
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
                # Monitor threads
                for i, thread in enumerate(self.running_threads):
                    if not thread.is_alive():
                        thread_name = thread.name
                        print(f"  ⚠️  Thread stopped: {thread_name}")
                        self.running_threads.pop(i)
                
                # Monitor processes
                for i, proc in enumerate(self.running_processes):
                    if proc.poll() is not None:
                        print(f"  ⚠️  Process ended: {proc.args}")
                        self.running_processes.pop(i)
                        
        except KeyboardInterrupt:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown of all components"""
        print("\n\n🛑 Shutting down autonomous system...")
        
        # Stop all processes
        for proc in self.running_processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                proc.kill()
        
        # Threads are daemonized, they'll exit with main
        
        print("✅ Shutdown complete.")
        sys.exit(0)

def main():
    """Main entry point"""
    launcher = AutonomousLauncher()
    launcher.launch_system()

if __name__ == "__main__":
    main()
