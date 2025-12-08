#!/usr/bin/env python3
"""
ENHANCED AUTONOMOUS LAUNCHER
Scans entire repo, processes conv.txt, and auto-runs all necessary files
Now with conv.txt integration and GitHub environment detection
"""

import os
import sys
import subprocess
import importlib
import threading
import time
import json
import re
import webbrowser
import socket
import platform
from pathlib import Path
from typing import Dict, List, Set, Optional
from datetime import datetime

# Import conv processor if available
try:
    from conv_processor import ConvProcessor
    CONV_PROCESSOR_AVAILABLE = True
except ImportError:
    CONV_PROCESSOR_AVAILABLE = False
    print("⚠️  ConvProcessor not available - install dependencies")

class EnhancedAutonomousLauncher:
    def __init__(self):
        self.root = Path.cwd()
        self.running_processes = []
        self.running_threads = []
        self.system_status = {}
        self.launch_log = []
        
        # GitHub environment detection
        self.is_github_codespaces = self._detect_github_codespaces()
        self.is_github_actions = self._detect_github_actions()
        self.is_local = not (self.is_github_codespaces or self.is_github_actions)
        
        # Enhanced dependency order with dynamic discovery
        self.dependency_order = self._discover_dependency_order()
        
        # Files to exclude from auto-run
        self.exclude_files = {
            'autonomous_config.json',
            'conv.txt',
            'docker-compose.yml',
            'Dockerfile',
            'requirements.txt',
            'file structure.txt',
            'project structure.txt',
            'real_implementation_fixes.py',
            'security_vault.py',
            '__pycache__',
            '.git',
            '.env',
            'node_modules',
        }
        
        # File extensions and execution methods
        self.execution_methods = {
            '.py': self._run_python_file,
            '.js': self._run_javascript_file,
            '.html': self._serve_html_file,
            '.json': self._load_config_file,
            '.sh': self._run_shell_script,
            '.yaml': self._load_config_file,
            '.yml': self._load_config_file,
        }
        
        # Environment-specific configurations
        self.env_config = self._load_environment_config()
        
    def _detect_github_codespaces(self) -> bool:
        """Detect if running in GitHub Codespaces"""
        return os.path.exists('/etc/codespaces') or 'CODESPACES' in os.environ
    
    def _detect_github_actions(self) -> bool:
        """Detect if running in GitHub Actions"""
        return 'GITHUB_ACTIONS' in os.environ
    
    def _load_environment_config(self) -> Dict:
        """Load environment-specific configuration"""
        config = {
            'max_concurrent': 10 if self.is_local else 5,
            'start_delay': 0.5 if self.is_local else 1.0,
            'timeout': 30 if self.is_local else 60,
            'auto_open_browser': self.is_local,
            'port': 8080,
        }
        
        # Load from config file if exists
        config_file = self.root / 'autonomous_config.json'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    config.update(user_config)
            except:
                pass
        
        return config
    
    def _discover_dependency_order(self) -> List[str]:
        """Dynamically discover dependency order by analyzing files"""
        base_order = [
            # Core infrastructure
            'project_sync_hub.py',
            'neural_sync_engine.py',
            
            # Core consciousness
            'consciousness_core.py',
            'autonomous_core.py',
            
            # Decision and execution
            'autonomous_decision_maker.py',
            'autonomous_execution_engine.py',
            
            # Integration systems
            'python_js_bridge.py',
            'browser_automation_integration.py',
            
            # Account and service management
            'autonomous_account_creator.py',
            'missing_service_locations.py',
            
            # Web interfaces
            'unified_dashboard.py',
            'pages.py',
            
            # Advanced autonomous systems
            'autonomous_discovery_engine.py',
            'codespaces_consciousness_orchestrator.py',
            
            # Real-time systems
            'live_consciousness_stream.py',
            'learning_feedback_loop.py',
            
            # Security and monitoring (run last)
            'amoral_enforcer.py',
            'security_vault.py',
        ]
        
        # Scan for additional important files
        discovered = []
        for root, dirs, files in os.walk(self.root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_files]
            
            for file in files:
                if file.endswith('.py') and file not in base_order:
                    # Check if file looks important (contains class definitions, main, etc.)
                    filepath = Path(root) / file
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                            if ('class ' in content and '__init__' in content) or \
                               ('def main()' in content) or \
                               ('def start()' in content) or \
                               ('def initialize()' in content):
                                discovered.append(file)
                    except:
                        pass
        
        # Combine base order with discovered files
        full_order = [f for f in base_order if (self.root / f).exists()]
        full_order.extend([f for f in discovered if f not in full_order])
        
        return full_order
    
    def process_conv_file(self):
        """Process conv.txt to extract and create missing files"""
        if not CONV_PROCESSOR_AVAILABLE:
            print("  ⚠️  ConvProcessor not available - skipping conv.txt processing")
            return None
        
        conv_file = self.root / 'conv.txt'
        if not conv_file.exists():
            print("  ℹ️  conv.txt not found - skipping")
            return None
        
        print("  🔍 Processing conv.txt...")
        processor = ConvProcessor(str(conv_file))
        
        # Scan and analyze
        findings = processor.scan_conv()
        
        # Create missing files
        created = processor.create_missing_files()
        if created:
            print(f"    ✅ Created {len(created)} files from conv.txt")
        
        # Update dependency order with new files
        for file in created:
            if file.endswith('.py') and file not in self.dependency_order:
                self.dependency_order.append(file)
        
        # Execute commands from conv.txt
        print("    ⚡ Executing commands from conv.txt...")
        processor.execute_commands()
        
        # Integrate findings
        processor.integrate_with_launcher(self)
        
        return processor
    
    def scan_entire_repo(self) -> Dict[str, List[str]]:
        """Scan entire repository comprehensively"""
        print("  📁 Scanning repository...")
        
        files_by_type = {
            'python': [],
            'javascript': [],
            'html': [],
            'json': [],
            'config': [],
            'shell': [],
            'other': [],
        }
        
        for root, dirs, filenames in os.walk(self.root):
            # Skip excluded directories
            dirs[:] = [
                d for d in dirs 
                if not d.startswith('.') 
                and d not in self.exclude_files
                and not d.startswith('__')
            ]
            
            for filename in filenames:
                if filename in self.exclude_files:
                    continue
                
                filepath = Path(root) / filename
                ext = filepath.suffix.lower()
                
                if ext == '.py':
                    files_by_type['python'].append(str(filepath))
                elif ext in ['.js', '.jsx', '.ts', '.tsx']:
                    files_by_type['javascript'].append(str(filepath))
                elif ext == '.html':
                    files_by_type['html'].append(str(filepath))
                elif ext == '.json':
                    files_by_type['json'].append(str(filepath))
                elif ext in ['.yaml', '.yml', '.toml', '.ini', '.cfg']:
                    files_by_type['config'].append(str(filepath))
                elif ext in ['.sh', '.bash', '.zsh']:
                    files_by_type['shell'].append(str(filepath))
                else:
                    files_by_type['other'].append(str(filepath))
        
        # Log the scan results
        self.launch_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'repo_scan',
            'results': {k: len(v) for k, v in files_by_type.items()}
        })
        
        return files_by_type
    
    def _run_python_file(self, filepath: str) -> Optional[threading.Thread]:
        """Run a Python file with enhanced error handling"""
        try:
            rel_path = Path(filepath).relative_to(self.root)
            module_name = str(rel_path).replace('/', '.').replace('.py', '')
            
            print(f"    🐍 Starting: {rel_path}")
            
            # Add to Python path
            sys.path.insert(0, str(Path(filepath).parent))
            
            # Import module
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if spec is None:
                print(f"    ⚠️  Failed to load spec for {module_name}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                print(f"    ⚠️  Error loading module {module_name}: {e}")
                # Try to run as script instead
                return self._run_as_script(filepath)
            
            # Check for various entry points
            entry_points = ['main', 'start', 'initialize', 'run', 'launch']
            
            for entry_point in entry_points:
                if hasattr(module, entry_point):
                    func = getattr(module, entry_point)
                    if callable(func):
                        thread = threading.Thread(
                            target=self._wrap_entry_point(func, module_name),
                            daemon=True,
                            name=f"thread_{module_name}_{entry_point}"
                        )
                        thread.start()
                        
                        self.running_threads.append({
                            'thread': thread,
                            'module': module_name,
                            'file': str(rel_path),
                            'entry_point': entry_point,
                            'started_at': datetime.now().isoformat()
                        })
                        
                        self.system_status[module_name] = {
                            'status': 'running',
                            'file': str(rel_path),
                            'entry_point': entry_point
                        }
                        
                        self.launch_log.append({
                            'timestamp': datetime.now().isoformat(),
                            'action': 'start_python_module',
                            'module': module_name,
                            'file': str(rel_path),
                            'entry_point': entry_point
                        })
                        
                        return thread
            
            # If no entry point found, check if it's a module with classes
            if hasattr(module, '__file__') and '__init__' in module.__file__:
                print(f"    ℹ️  Module loaded (no entry point): {module_name}")
                self.system_status[module_name] = {
                    'status': 'loaded',
                    'file': str(rel_path)
                }
                return None
            
            print(f"    ⚠️  No entry point found for {module_name}")
            return None
            
        except Exception as e:
            print(f"    ❌ Error running {filepath}: {e}")
            self.launch_log.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'error_python_module',
                'file': filepath,
                'error': str(e)
            })
            return None
    
    def _wrap_entry_point(self, func, module_name):
        """Wrapper for entry point functions with error handling"""
        try:
            return func()
        except Exception as e:
            print(f"    ❌ Error in {module_name} entry point: {e}")
            self.launch_log.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'error_entry_point',
                'module': module_name,
                'error': str(e)
            })
    
    def _run_as_script(self, filepath: str) -> Optional[threading.Thread]:
        """Run Python file as script using subprocess"""
        try:
            rel_path = Path(filepath).relative_to(self.root)
            print(f"    ⚡ Running as script: {rel_path}")
            
            proc = subprocess.Popen(
                [sys.executable, filepath],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.running_processes.append({
                'process': proc,
                'file': str(rel_path),
                'type': 'python_script',
                'started_at': datetime.now().isoformat()
            })
            
            # Start thread to monitor output
            def monitor_output():
                for line in proc.stdout:
                    if line.strip():
                        print(f"      [{rel_path.name}] {line.strip()}")
                proc.wait()
            
            monitor_thread = threading.Thread(
                target=monitor_output,
                daemon=True,
                name=f"monitor_{rel_path.name}"
            )
            monitor_thread.start()
            
            return monitor_thread
            
        except Exception as e:
            print(f"    ❌ Error running script {filepath}: {e}")
            return None
    
    def _run_javascript_file(self, filepath: str) -> bool:
        """Run JavaScript/Node.js files"""
        try:
            rel_path = Path(filepath).relative_to(self.root)
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Determine JS type
            is_nodejs = any(pattern in content for pattern in [
                'require(', 'module.exports', 'process.argv', '__dirname'
            ])
            
            if is_nodejs:
                print(f"    📦 Starting Node.js: {rel_path}")
                proc = subprocess.Popen(
                    ['node', filepath],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                self.running_processes.append({
                    'process': proc,
                    'file': str(rel_path),
                    'type': 'nodejs',
                    'started_at': datetime.now().isoformat()
                })
                
                self.system_status[str(rel_path)] = 'running'
                
                # Monitor output in background
                threading.Thread(
                    target=self._monitor_process_output,
                    args=(proc, str(rel_path)),
                    daemon=True
                ).start()
                
            else:
                print(f"    🌐 Browser JS: {rel_path}")
                self.system_status[str(rel_path)] = 'browser_js'
            
            return True
            
        except Exception as e:
            print(f"    ⚠️  Error with JS {filepath}: {e}")
            return False
    
    def _monitor_process_output(self, proc, name: str):
        """Monitor process output in background"""
        try:
            stdout, stderr = proc.communicate(timeout=300)
            if stdout:
                print(f"      [{name}] {stdout[:200]}...")
            if stderr:
                print(f"      [{name}] ERROR: {stderr[:200]}...")
        except subprocess.TimeoutExpired:
            print(f"      [{name}] Process timeout")
        except Exception as e:
            print(f"      [{name}] Monitor error: {e}")
    
    def _serve_html_file(self, filepath: str) -> bool:
        """Handle HTML files"""
        rel_path = Path(filepath).relative_to(self.root)
        
        # Check if it's a dashboard or main page
        if 'dashboard' in str(rel_path).lower() or 'index' in str(rel_path).lower():
            print(f"    📊 Dashboard HTML: {rel_path}")
            
            if self.env_config['auto_open_browser'] and self.is_local:
                # Try to open in browser
                url = f"http://localhost:{self.env_config['port']}/{rel_path}"
                threading.Timer(2, lambda: webbrowser.open(url)).start()
                print(f"      🌐 Will open: {url}")
        
        self.system_status[str(rel_path)] = 'available'
        return True
    
    def _load_config_file(self, filepath: str) -> bool:
        """Load configuration files"""
        rel_path = Path(filepath).relative_to(self.root)
        
        try:
            with open(filepath, 'r') as f:
                if filepath.endswith('.json'):
                    data = json.load(f)
                else:
                    data = f.read()
            
            print(f"    ⚙️  Loaded config: {rel_path}")
            
            # Special handling for certain config files
            if filepath.endswith('autonomous_config.json'):
                self._update_from_config(data)
            
            self.system_status[str(rel_path)] = 'loaded'
            return True
            
        except Exception as e:
            print(f"    ⚠️  Error loading config {filepath}: {e}")
            return False
    
    def _run_shell_script(self, filepath: str) -> bool:
        """Run shell scripts"""
        try:
            rel_path = Path(filepath).relative_to(self.root)
            
            # Make executable if needed
            os.chmod(filepath, os.stat(filepath).st_mode | 0o111)
            
            print(f"    🐚 Running shell script: {rel_path}")
            
            proc = subprocess.Popen(
                ['bash', filepath],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.running_processes.append({
                'process': proc,
                'file': str(rel_path),
                'type': 'shell_script',
                'started_at': datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            print(f"    ⚠️  Error with shell script {filepath}: {e}")
            return False
    
    def _update_from_config(self, config: Dict):
        """Update launcher configuration from config file"""
        if 'dependencies' in config:
            # Add dependencies to execution order
            for dep in config.get('dependencies', []):
                if dep not in self.dependency_order:
                    self.dependency_order.append(dep)
        
        if 'exclude' in config:
            self.exclude_files.update(config['exclude'])
    
    def determine_execution_order(self, python_files: List[str]) -> List[str]:
        """Determine optimal execution order with dependency resolution"""
        # Start with files in dependency order
        ordered = []
        remaining = python_files.copy()
        
        for dep in self.dependency_order:
            for py_file in remaining[:]:
                if py_file.endswith(dep):
                    ordered.append(py_file)
                    remaining.remove(py_file)
                    break
        
        # Sort remaining files by likely importance
        remaining.sort(key=lambda x: (
            'core' in x.lower(),
            'engine' in x.lower(),
            'manager' in x.lower(),
            'service' in x.lower(),
            'utils' in x.lower(),
            'test' in x.lower(),
        ), reverse=True)
        
        ordered.extend(remaining)
        return ordered
    
    def check_ports(self) -> Dict[int, bool]:
        """Check which ports are available"""
        ports = {
            8080: False,  # Dashboard
            3000: False,  # Node.js apps
            5000: False,  # Python APIs
            5432: False,  # PostgreSQL
            6379: False,  # Redis
            27017: False, # MongoDB
        }
        
        print("  🔌 Checking port availability...")
        for port in ports.keys():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('localhost', port))
                    ports[port] = True
                    print(f"    ✅ Port {port}: Available")
                except:
                    print(f"    ⚠️  Port {port}: In use")
        
        return ports
    
    def launch_system(self):
        """Main launch sequence"""
        print("=" * 70)
        print("🚀 ENHANCED AUTONOMOUS SYSTEM LAUNCHER")
        print(f"   Environment: {'GitHub Codespaces' if self.is_github_codespaces else 'GitHub Actions' if self.is_github_actions else 'Local'}")
        print("=" * 70)
        
        # Step 1: Check system
        print("\n🔧 System Check:")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  Platform: {platform.platform()}")
        print(f"  Root: {self.root}")
        
        # Step 2: Check ports
        port_status = self.check_ports()
        
        # Step 3: Process conv.txt
        conv_processor = self.process_conv_file()
        
        # Step 4: Scan repository
        print("\n📁 Repository Scan:")
        files = self.scan_entire_repo()
        
        print(f"  Python files: {len(files['python'])}")
        print(f"  JavaScript files: {len(files['javascript'])}")
        print(f"  HTML files: {len(files['html'])}")
        print(f"  Config files: {len(files['config'])}")
        print(f"  Shell scripts: {len(files['shell'])}")
        print(f"  Other files: {len(files['other'])}")
        
        # Step 5: Determine execution order
        print("\n⚡ Determining execution order...")
        ordered_python = self.determine_execution_order(files['python'])
        
        print(f"  Files to execute: {len(ordered_python)}")
        if len(ordered_python) > 20:
            print(f"  First 5: {[Path(p).name for p in ordered_python[:5]]}...")
        
        # Step 6: Launch Python systems
        print("\n🐍 Launching Python systems...")
        python_success = 0
        
        for i, py_file in enumerate(ordered_python):
            print(f"\n[{i+1}/{len(ordered_python)}] ", end="")
            thread = self._run_python_file(py_file)
            if thread:
                python_success += 1
            
            # Respect max concurrent limit
            active_threads = sum(1 for t in self.running_threads if t['thread'].is_alive())
            if active_threads >= self.env_config['max_concurrent']:
                time.sleep(1)
            else:
                time.sleep(self.env_config['start_delay'])
        
        # Step 7: Launch other file types
        print("\n📦 Launching other components...")
        
        # JavaScript files
        js_success = 0
        for js_file in files['javascript']:
            if self._run_javascript_file(js_file):
                js_success += 1
        
        # Shell scripts
        shell_success = 0
        for shell_file in files['shell']:
            if self._run_shell_script(shell_file):
                shell_success += 1
        
        # Config files
        for config_file in files['config']:
            self._load_config_file(config_file)
        
        # HTML files
        for html_file in files['html']:
            self._serve_html_file(html_file)
        
        # Step 8: Final status
        print("\n" + "=" * 70)
        print("✅ LAUNCH COMPLETE")
        print("=" * 70)
        
        print(f"\n📊 Launch Summary:")
        print(f"  Python systems: {python_success}/{len(ordered_python)}")
        print(f"  JavaScript: {js_success}/{len(files['javascript'])}")
        print(f"  Shell scripts: {shell_success}/{len(files['shell'])}")
        print(f"  Active threads: {len([t for t in self.running_threads if t['thread'].is_alive()])}")
        print(f"  Running processes: {len(self.running_processes)}")
        
        print(f"\n🌐 Available URLs:")
        if port_status[8080]:
            print(f"  Dashboard: http://localhost:8080/")
        if port_status[3000]:
            print(f"  Node apps: http://localhost:3000/")
        if port_status[5000]:
            print(f"  API: http://localhost:5000/")
        
        # Save launch log
        self.save_launch_log()
        
        print("\n🎯 System is now autonomous. Press Ctrl+C to shutdown.")
        
        # Monitor and keep alive
        self.monitor_system()
    
    def monitor_system(self):
        """Monitor running systems and keep main thread alive"""
        try:
            while True:
                time.sleep(5)
                
                # Check threads
                active_threads = []
                for i, thread_info in enumerate(self.running_threads):
                    thread = thread_info['thread']
                    if not thread.is_alive():
                        print(f"  ⚠️  Thread stopped: {thread_info['module']}")
                        self.system_status[thread_info['module']]['status'] = 'stopped'
                    else:
                        active_threads.append(thread_info)
                
                self.running_threads = active_threads
                
                # Check processes
                active_processes = []
                for i, proc_info in enumerate(self.running_processes):
                    proc = proc_info['process']
                    if proc.poll() is not None:
                        print(f"  ⚠️  Process ended: {proc_info['file']}")
                    else:
                        active_processes.append(proc_info)
                
                self.running_processes = active_processes
                
                # Print status every 30 seconds
                if int(time.time()) % 30 == 0:
                    active_count = len([t for t in self.running_threads if t['thread'].is_alive()])
                    print(f"  📈 System status: {active_count} active threads, {len(self.running_processes)} processes")
                    
        except KeyboardInterrupt:
            self.shutdown()
    
    def save_launch_log(self):
        """Save launch log to file"""
        log_file = self.root / 'launch_log.json'
        
        log_data = {
            'launch_time': datetime.now().isoformat(),
            'environment': {
                'is_github_codespaces': self.is_github_codespaces,
                'is_github_actions': self.is_github_actions,
                'is_local': self.is_local,
                'python_version': sys.version,
                'platform': platform.platform(),
            },
            'launch_log': self.launch_log,
            'system_status': self.system_status,
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"  📄 Launch log saved to: {log_file}")
    
    def shutdown(self):
        """Clean shutdown of all components"""
        print("\n\n🛑 Shutting down autonomous system...")
        
        print("  Terminating processes...")
        for proc_info in self.running_processes:
            try:
                proc_info['process'].terminate()
            except:
                pass
        
        print("  Waiting for processes to exit...")
        for proc_info in self.running_processes:
            try:
                proc_info['process'].wait(timeout=5)
            except:
                try:
                    proc_info['process'].kill()
                except:
                    pass
        
        print("  Threads will exit with main (daemonized)...")
        
        # Save final log
        self.launch_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'shutdown',
            'active_threads': len(self.running_threads),
            'active_processes': len(self.running_processes)
        })
        self.save_launch_log()
        
        print("✅ Shutdown complete.")
        sys.exit(0)

def main():
    """Main entry point"""
    launcher = EnhancedAutonomousLauncher()
    launcher.launch_system()

if __name__ == "__main__":
    main()
