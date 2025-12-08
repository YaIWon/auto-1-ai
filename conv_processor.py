#!/usr/bin/env python3
"""
CONV.TXT PROCESSOR
Scans conv.txt and extracts all files mentioned, then adds them to environment
Also processes conversations for action items and commands
"""

import os
import re
import json
import shutil
import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import subprocess

class ConvProcessor:
    def __init__(self, conv_path: str = "conv.txt"):
        self.conv_path = Path(conv_path)
        self.root = Path.cwd()
        self.extracted_files = []
        self.code_blocks = []
        self.commands = []
        self.action_items = []
        
        # Regex patterns for extraction
        self.patterns = {
            'python_block': r'```python\n(.*?)```',
            'javascript_block': r'```javascript\n(.*?)```',
            'js_block': r'```js\n(.*?)```',
            'json_block': r'```json\n(.*?)```',
            'file_reference': r'`([\w\-/]+\.\w+)`',  # Matches `filename.ext`
            'path_reference': r'[\'\"]([\w\-/]+\.\w+)[\'\"]',  # Matches 'path/file.ext'
            'command': r'\$ (.*?)\n',  # Shell commands
            'action_item': r'-\s*\[(?:x| )\]\s*(.*)',  # Todo items
        }
        
    def scan_conv(self) -> Dict:
        """Scan conv.txt for all code blocks, files, and commands"""
        if not self.conv_path.exists():
            print(f"⚠️  conv.txt not found at {self.conv_path}")
            return {}
        
        with open(self.conv_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        findings = {
            'files': self._extract_files(content),
            'python_code': self._extract_code(content, 'python'),
            'javascript_code': self._extract_code(content, 'javascript'),
            'json_code': self._extract_code(content, 'json'),
            'shell_commands': self._extract_commands(content),
            'action_items': self._extract_actions(content),
            'potential_functions': self._extract_functions(content),
            'import_statements': self._extract_imports(content),
            'dependencies': self._extract_dependencies(content),
        }
        
        return findings
    
    def _extract_files(self, content: str) -> List[str]:
        """Extract all file references from conv.txt"""
        files = set()
        
        # Look for file references in backticks
        for match in re.finditer(self.patterns['file_reference'], content):
            files.add(match.group(1))
        
        # Look for file references in quotes
        for match in re.finditer(self.patterns['path_reference'], content):
            files.add(match.group(1))
        
        # Look for "create file", "add file", "file: " patterns
        file_patterns = [
            r'create (?:file|file:)\s*([\w\-/]+\.\w+)',
            r'add (?:file|file:)\s*([\w\-/]+\.\w+)',
            r'File:\s*([\w\-/]+\.\w+)',
            r'filename:\s*([\w\-/]+\.\w+)',
            r'save as\s*([\w\-/]+\.\w+)',
        ]
        
        for pattern in file_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                files.add(match.group(1))
        
        return sorted(list(files))
    
    def _extract_code(self, content: str, language: str) -> List[Dict]:
        """Extract code blocks by language"""
        code_blocks = []
        
        # Handle different language tags
        lang_tags = {
            'python': ['python', 'py'],
            'javascript': ['javascript', 'js'],
            'json': ['json']
        }
        
        for tag in lang_tags.get(language, [language]):
            pattern = f'```{tag}\\n(.*?)```'
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                code_blocks.append({
                    'content': match.strip(),
                    'language': language,
                    'size': len(match)
                })
        
        return code_blocks
    
    def _extract_commands(self, content: str) -> List[str]:
        """Extract shell commands"""
        commands = []
        
        # $ command pattern
        for match in re.finditer(self.patterns['command'], content):
            commands.append(match.group(1))
        
        # Also look for "run:", "execute:", "command:" patterns
        command_patterns = [
            r'run:\s*(.*?)\n',
            r'execute:\s*(.*?)\n',
            r'command:\s*(.*?)\n',
            r'terminal:\s*(.*?)\n',
        ]
        
        for pattern in command_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                commands.append(match.group(1))
        
        return commands
    
    def _extract_actions(self, content: str) -> List[str]:
        """Extract action items and todo lists"""
        actions = []
        
        # Todo list items
        for match in re.finditer(self.patterns['action_item'], content):
            actions.append(match.group(1))
        
        # "Need to", "should", "must" patterns
        action_patterns = [
            r'(?:need to|should|must|will)\s+(.*?)[\.\n]',
            r'TODO[:\s]*(.*?)\n',
            r'FIXME[:\s]*(.*?)\n',
            r'ACTION[:\s]*(.*?)\n',
        ]
        
        for pattern in action_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                actions.append(match.group(1))
        
        return actions
    
    def _extract_functions(self, content: str) -> List[str]:
        """Extract function definitions"""
        functions = []
        
        # Python function patterns
        py_func_pattern = r'def\s+(\w+)\s*\(.*?\):'
        for match in re.finditer(py_func_pattern, content):
            functions.append(f"def {match.group(1)}()")
        
        # JS function patterns
        js_func_patterns = [
            r'function\s+(\w+)\s*\(.*?\)',
            r'const\s+(\w+)\s*=\s*\(.*?\)\s*=>',
            r'let\s+(\w+)\s*=\s*\(.*?\)\s*=>',
        ]
        
        for pattern in js_func_patterns:
            for match in re.finditer(pattern, content):
                functions.append(f"function {match.group(1)}()")
        
        return functions
    
    def _extract_imports(self, content: str) -> List[str]:
        """Extract import/require statements"""
        imports = []
        
        # Python imports
        py_imports = re.findall(r'^(?:import|from)\s+[\w\.]+', content, re.MULTILINE)
        imports.extend(py_imports)
        
        # JS/Node imports
        js_imports = re.findall(r'^(?:import|require|const.*=.*require)[^;]+', content, re.MULTILINE)
        imports.extend(js_imports)
        
        return imports
    
    def _extract_dependencies(self, content: str) -> Dict:
        """Extract potential dependencies"""
        deps = {
            'python': set(),
            'javascript': set(),
            'system': set()
        }
        
        # Look for pip install commands
        pip_patterns = [
            r'pip(?:3)? install(?: --user)?(?: --upgrade)? ([^\n&|;]+)',
            r'requirements?\.txt.*?([\w\-]+(?:==[\d\.]+)?)',
        ]
        
        for pattern in pip_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                packages = match.group(1).split()
                deps['python'].update(packages)
        
        # Look for npm install commands
        npm_patterns = [
            r'npm install(?: --save)?(?: --save-dev)? ([^\n&|;]+)',
            r'yarn add(?: --dev)? ([^\n&|;]+)',
            r'package\.json.*?"([\w\-]+)":',
        ]
        
        for pattern in npm_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                packages = match.group(1).split()
                deps['javascript'].update(packages)
        
        # System packages
        sys_patterns = [
            r'apt(?:-get)? install ([^\n&|;]+)',
            r'yum install ([^\n&|;]+)',
            r'brew install ([^\n&|;]+)',
        ]
        
        for pattern in sys_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                packages = match.group(1).split()
                deps['system'].update(packages)
        
        # Convert sets to lists
        return {k: list(v) for k, v in deps.items()}
    
    def create_missing_files(self) -> List[str]:
        """Create files mentioned in conv.txt that don't exist"""
        findings = self.scan_conv()
        created_files = []
        
        # Create files from extracted code blocks
        for i, code_block in enumerate(findings.get('python_code', [])):
            if code_block['content']:
                # Try to determine filename from code
                filename = self._guess_filename(code_block['content'], 'py')
                if filename and not Path(filename).exists():
                    self._write_file(filename, code_block['content'])
                    created_files.append(filename)
                    print(f"  ✅ Created: {filename}")
        
        # Create from JSON code blocks
        for i, code_block in enumerate(findings.get('json_code', [])):
            if code_block['content']:
                try:
                    data = json.loads(code_block['content'])
                    filename = f"config_{i}.json" if i > 0 else "config.json"
                    if not Path(filename).exists():
                        with open(filename, 'w') as f:
                            json.dump(data, f, indent=2)
                        created_files.append(filename)
                        print(f"  ✅ Created: {filename}")
                except:
                    pass
        
        # Create from file references
        for file_ref in findings.get('files', []):
            if not Path(file_ref).exists():
                # Create directory if needed
                Path(file_ref).parent.mkdir(parents=True, exist_ok=True)
                
                # Create empty file
                with open(file_ref, 'w') as f:
                    f.write(f"# Auto-generated from conv.txt\n")
                    f.write(f"# File: {file_ref}\n")
                    f.write(f"# Created by ConvProcessor\n\n")
                
                created_files.append(file_ref)
                print(f"  ✅ Created: {file_ref}")
        
        return created_files
    
    def _guess_filename(self, content: str, extension: str) -> str:
        """Try to guess filename from code content"""
        # Look for class names
        class_match = re.search(r'class\s+(\w+)', content)
        if class_match:
            return f"{class_match.group(1).lower()}.{extension}"
        
        # Look for main function
        if 'def main()' in content or 'if __name__' in content:
            return f"main.{extension}"
        
        # Look for filename in comments
        comment_match = re.search(r'File:\s*(\w+\.\w+)', content)
        if comment_match:
            return comment_match.group(1)
        
        return None
    
    def _write_file(self, filename: str, content: str):
        """Write content to file, creating directories if needed"""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            f.write(content)
    
    def execute_commands(self) -> List[Tuple[str, bool]]:
        """Execute shell commands found in conv.txt"""
        findings = self.scan_conv()
        results = []
        
        for command in findings.get('shell_commands', []):
            print(f"  ⚡ Executing: {command}")
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                success = result.returncode == 0
                results.append((command, success))
                
                if success:
                    print(f"    ✅ Success")
                    if result.stdout.strip():
                        print(f"      Output: {result.stdout[:100]}...")
                else:
                    print(f"    ❌ Failed: {result.stderr[:100]}")
                    
            except Exception as e:
                print(f"    ⚠️  Error: {e}")
                results.append((command, False))
        
        return results
    
    def generate_report(self) -> Dict:
        """Generate comprehensive report of conv.txt analysis"""
        findings = self.scan_conv()
        
        report = {
            'summary': {
                'total_files_referenced': len(findings.get('files', [])),
                'python_code_blocks': len(findings.get('python_code', [])),
                'javascript_code_blocks': len(findings.get('javascript_code', [])),
                'shell_commands': len(findings.get('shell_commands', [])),
                'action_items': len(findings.get('action_items', [])),
                'functions_identified': len(findings.get('potential_functions', [])),
            },
            'files': findings.get('files', []),
            'dependencies': findings.get('dependencies', {}),
            'actions_needed': findings.get('action_items', []),
            'imports_required': findings.get('import_statements', []),
        }
        
        return report
    
    def integrate_with_launcher(self, launcher_instance):
        """Integrate findings with AutonomousLauncher"""
        findings = self.scan_conv()
        
        # Add discovered files to launcher
        for file_ref in findings.get('files', []):
            if Path(file_ref).exists() and file_ref.endswith('.py'):
                if file_ref not in launcher_instance.dependency_order:
                    launcher_instance.dependency_order.append(file_ref)
        
        # Add dependencies to requirements
        deps = findings.get('dependencies', {}).get('python', [])
        if deps:
            requirements_file = Path('requirements.txt')
            existing = set()
            if requirements_file.exists():
                with open(requirements_file, 'r') as f:
                    existing = set(line.strip() for line in f if line.strip())
            
            with open(requirements_file, 'a') as f:
                for dep in deps:
                    if dep not in existing:
                        f.write(f"{dep}\n")
                        print(f"  📦 Added dependency: {dep}")

def main():
    """Main entry point for conv.txt processing"""
    processor = ConvProcessor()
    
    print("=" * 60)
    print("📄 CONV.TXT PROCESSOR")
    print("=" * 60)
    
    if not processor.conv_path.exists():
        print(f"❌ conv.txt not found at {processor.conv_path}")
        return
    
    # Scan and analyze
    print("\n🔍 Scanning conv.txt...")
    findings = processor.scan_conv()
    
    print(f"  📁 Files referenced: {len(findings.get('files', []))}")
    print(f"  🐍 Python code blocks: {len(findings.get('python_code', []))}")
    print(f"  📜 JS code blocks: {len(findings.get('javascript_code', []))}")
    print(f"  ⚡ Shell commands: {len(findings.get('shell_commands', []))}")
    print(f"  ✅ Action items: {len(findings.get('action_items', []))}")
    
    # Create missing files
    print("\n📝 Creating missing files...")
    created = processor.create_missing_files()
    print(f"  Created {len(created)} files")
    
    # Execute commands
    print("\n⚡ Executing commands from conv.txt...")
    results = processor.execute_commands()
    successes = sum(1 for _, success in results if success)
    print(f"  Commands executed: {successes}/{len(results)} successful")
    
    # Generate report
    print("\n📊 Generating report...")
    report = processor.generate_report()
    
    report_file = 'conv_analysis_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"  📄 Report saved to: {report_file}")
    
    print("\n✅ Conv.txt processing complete!")
    print("\nTo integrate with AutonomousLauncher, call:")
    print("  processor.integrate_with_launcher(launcher_instance)")

if __name__ == "__main__":
    main()
