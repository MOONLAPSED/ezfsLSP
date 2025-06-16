from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# LICENSE © 2025: CC BY 4.0: MOONLAPSED:https://github.com/Moonlapsed/ezfsLSP
"""
Morphic Markdown Agent - LSP Bridge
Integrates morphic ontology with Language Server Protocol operations
Usage: 
    md_agent.py extract [section_id] [document.md]
    md_agent.py inline [section_id] [document.md]
    md_agent.py analyze [document.md]  # Analyze document structure with morphic principles
"""
import sys
import os
import re
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

from ontologypy import BYTE, QuantumState, HilbertSpace, MorphicComplex, least_significant_unit, WordSize

class LSPServer:
    """
    Minimal LSP server implementation for morphic markdown operations.
    Follows the Language Server Protocol specification.
    """
    
    def __init__(self, agent: 'MorphicMarkdownAgent'):
        self.agent = agent
        self.documents: Dict[str, str] = {}  # uri -> content
        
    def start(self):
        """Start the LSP server loop."""
        while True:
            try:
                message = self._read_message()
                if message is None:
                    break
                    
                response = self._handle_message(message)
                if response:
                    self._write_message(response)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                # Log error but continue serving
                self._log_error(f"LSP Server error: {e}")
    
    def _read_message(self) -> Optional[Dict[str, Any]]:
        """Read a JSON-RPC message from stdin."""
        try:
            # Read headers
            headers = {}
            while True:
                line = sys.stdin.buffer.readline().decode('utf-8')
                if line == '\r\n' or line == '\n':
                    break
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    headers[key.strip()] = value.strip()
            
            # Read content
            content_length = int(headers.get('Content-Length', 0))
            if content_length == 0:
                return None
                
            content = sys.stdin.buffer.read(content_length).decode('utf-8')
            return json.loads(content)
            
        except Exception:
            return None
    
    def _write_message(self, message: Dict[str, Any]):
        """Write a JSON-RPC message to stdout."""
        content = json.dumps(message)
        content_bytes = content.encode('utf-8')
        
        response = f"Content-Length: {len(content_bytes)}\r\n\r\n{content}"
        sys.stdout.buffer.write(response.encode('utf-8'))
        sys.stdout.buffer.flush()
    
    def _log_error(self, message: str):
        """Log error to stderr."""
        print(f"LSP Error: {message}", file=sys.stderr, flush=True)
    
    def _handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle incoming LSP message."""
        method = message.get('method')
        msg_id = message.get('id')
        params = message.get('params', {})
        
        # Handle notifications (no response needed)
        if msg_id is None:
            self._handle_notification(method, params)
            return None
        
        # Handle requests (response needed)
        try:
            result = self._handle_request(method, params)
            return {
                'jsonrpc': '2.0',
                'id': msg_id,
                'result': result
            }
        except Exception as e:
            return {
                'jsonrpc': '2.0',
                'id': msg_id,
                'error': {
                    'code': -32603,  # Internal error
                    'message': str(e)
                }
            }
    
    def _handle_notification(self, method: str, params: Dict[str, Any]):
        """Handle LSP notifications."""
        if method == 'textDocument/didOpen':
            uri = params['textDocument']['uri']
            content = params['textDocument']['text']
            self.documents[uri] = content
            
        elif method == 'textDocument/didChange':
            uri = params['textDocument']['uri']
            # For simplicity, assume full document sync
            if params['contentChanges']:
                self.documents[uri] = params['contentChanges'][0]['text']
                
        elif method == 'textDocument/didClose':
            uri = params['textDocument']['uri']
            self.documents.pop(uri, None)
    
    def _handle_request(self, method: str, params: Dict[str, Any]) -> Any:
        """Handle LSP requests."""
        if method == 'initialize':
            return {
                'capabilities': {
                    'textDocumentSync': 1,  # Full document sync
                    'codeActionProvider': True,
                    'documentSymbolProvider': True,
                    'definitionProvider': True,
                    'referencesProvider': True
                }
            }
        
        elif method == 'initialized':
            return None
        
        elif method == 'textDocument/codeAction':
            return self._handle_code_actions(params)
        
        elif method == 'textDocument/documentSymbol':
            return self._handle_document_symbols(params)
        
        elif method == 'textDocument/definition':
            return self._handle_definition(params)
        
        elif method == 'textDocument/references':
            return self._handle_references(params)
        
        elif method == 'shutdown':
            return None
        
        else:
            raise Exception(f"Unhandled method: {method}")
    
    def _handle_code_actions(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle code action requests - this is where extract/inline happen."""
        uri = params['textDocument']['uri']
        content = self.documents.get(uri, '')
        
        if not content:
            return []
        
        # Register document with agent
        doc_state = self.agent.lsp_state.register_document(uri, content)
        
        actions = []
        
        # Check if cursor is on a section header - offer extract
        range_params = params.get('range', {})
        start_line = range_params.get('start', {}).get('line', 0)
        
        lines = content.split('\n')
        if start_line < len(lines):
            line = lines[start_line]
            if line.strip().startswith('#'):
                # On a header - offer extract
                actions.append({
                    'title': 'Extract Section',
                    'kind': 'refactor.extract',
                    'command': {
                        'title': 'Extract Section',
                        'command': 'morphic.extractSection',
                        'arguments': [uri, start_line]
                    }
                })
        
        # Check if cursor is on a link - offer inline
        if start_line < len(lines):
            line = lines[start_line]
            import re
            link_match = re.search(r'\[([^\]]+)\]\(([^)]+)\)', line)
            if link_match:
                section_id = link_match.group(2)
                if len(section_id) == 8 and all(c in '0123456789abcdef' for c in section_id):
                    # Looks like a morphic ID
                    actions.append({
                        'title': 'Inline Section',
                        'kind': 'refactor.inline',
                        'command': {
                            'title': 'Inline Section', 
                            'command': 'morphic.inlineSection',
                            'arguments': [uri, section_id]
                        }
                    })
        
        return actions
    
    def _handle_document_symbols(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle document symbol requests - shows table of contents."""
        uri = params['textDocument']['uri']
        content = self.documents.get(uri, '')
        
        if not content:
            return []
        
        # Register document with agent
        doc_state = self.agent.lsp_state.register_document(uri, content)
        
        symbols = []
        for section in doc_state.sections:
            symbols.append({
                'name': section.title,
                'kind': 2,  # SymbolKind.Module (good for sections)
                'location': {
                    'uri': uri,
                    'range': {
                        'start': {'line': section.start_line, 'character': 0},
                        'end': {'line': section.end_line, 'character': 0}
                    }
                },
                'detail': f"Level {section.level} - ID: {section.morphic_id}"
            })
        
        return symbols
    
    def _handle_definition(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle go-to-definition - follow links to extracted sections."""
        uri = params['textDocument']['uri']
        position = params['position']
        content = self.documents.get(uri, '')
        
        if not content:
            return None
        
        lines = content.split('\n')
        line_num = position['line']
        
        if line_num >= len(lines):
            return None
        
        line = lines[line_num]
        
        # Check if we're on a link
        import re
        link_match = re.search(r'\[([^\]]+)\]\(([^)]+)\)', line)
        if link_match:
            section_id = link_match.group(2)
            # Construct path to extracted file
            import os
            doc_dir = os.path.dirname(uri.replace('file://', ''))
            extracted_path = os.path.join(doc_dir, f"{section_id}.md")
            
            if os.path.exists(extracted_path):
                return {
                    'uri': f"file://{extracted_path}",
                    'range': {
                        'start': {'line': 0, 'character': 0},
                        'end': {'line': 0, 'character': 0}
                    }
                }
        
        return None
    
    def _handle_references(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle find references - find links to current section."""
        # This would be more complex - find all documents that link to current section
        return []

class LSPState:
    """
    Represents LSP server state integrated with morphic principles.
    Each document state is encoded as a morphic structure.
    """
    def __init__(self):
        self.documents: Dict[str, 'DocumentState'] = {}
        self.workspace_root: Optional[str] = None
        self.hilbert_space = HilbertSpace(dimension=8)  # 8D space for BYTE operations
    
    def register_document(self, uri: str, content: str) -> 'DocumentState':
        """Register a document and create its morphic state representation."""
        doc_state = DocumentState(uri, content, self.hilbert_space)
        self.documents[uri] = doc_state
        return doc_state
    
    def get_document(self, uri: str) -> Optional['DocumentState']:
        """Get document state by URI."""
        return self.documents.get(uri)

@dataclass
class MarkdownSection:
    """Represents a markdown section with morphic properties."""
    title: str
    content: str
    level: int
    start_line: int
    end_line: int
    morphic_id: str  # Generated from content hash
    
    def to_morphic_byte(self) -> BYTE:
        """Convert section properties to a morphic BYTE representation."""
        # Use LSU to extract meaningful bits from section properties
        title_hash = least_significant_unit(self.title, WordSize.BYTE)
        content_hash = least_significant_unit(self.content, WordSize.BYTE)
        level_bits = self.level & 0x0F  # 4 bits for level (TTTT)
        
        # Construct BYTE with our <C_C_VV|TTTT> structure
        # C bit (bit 7): 1 if section has content, 0 if empty
        c_bit = 1 if self.content.strip() else 0
        
        # _C_ bit (bit 6): 1 if section has subsections, 0 otherwise  
        has_subsections = bool(re.search(r'^#{' + str(self.level + 1) + r',}', self.content, re.MULTILINE))
        c_internal = 1 if has_subsections else 0
        
        # VV bits (5-4): Derived from content characteristics
        vv = (content_hash >> 6) & 0x03  # Extract 2 bits from content hash
        
        # TTTT bits (3-0): Section level and position info
        tttt = level_bits
        
        byte_value = (c_bit << 7) | (c_internal << 6) | (vv << 4) | tttt
        return BYTE(byte_value)

class DocumentState:
    """
    Represents the morphic state of a markdown document.
    Integrates with LSP document lifecycle.
    """
    def __init__(self, uri: str, content: str, hilbert_space: HilbertSpace):
        self.uri = uri
        self.content = content
        self.hilbert_space = hilbert_space
        self.sections: List[MarkdownSection] = []
        self.quantum_state: Optional[QuantumState] = None
        self._parse_sections()
        self._create_quantum_representation()
    
    def _parse_sections(self):
        """Parse markdown sections from content."""
        lines = self.content.split('\n')
        current_section = None
        
        for i, line in enumerate(lines):
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous section if exists
                if current_section:
                    current_section.end_line = i - 1
                    current_section.content = '\n'.join(lines[current_section.start_line + 1:i])
                    self.sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2)
                morphic_id = self._generate_morphic_id(title, level)
                
                current_section = MarkdownSection(
                    title=title,
                    content="",
                    level=level,
                    start_line=i,
                    end_line=len(lines) - 1,
                    morphic_id=morphic_id
                )
        
        # Don't forget the last section
        if current_section:
            current_section.end_line = len(lines) - 1
            current_section.content = '\n'.join(lines[current_section.start_line + 1:])
            self.sections.append(current_section)
    
    def _generate_morphic_id(self, title: str, level: int) -> str:
        """Generate a morphic ID using our hash system."""
        combined = f"{title}:{level}"
        hash_val = least_significant_unit(combined, WordSize.LONG)
        return f"{hash_val:08x}"  # 8-character hex ID
    
    def _create_quantum_representation(self):
        """Create quantum state representation of the document."""
        if not self.sections:
            return
        
        # Create amplitudes based on section morphic bytes
        amplitudes = []
        total_sections = min(len(self.sections), self.hilbert_space.dimension)
        
        for i in range(self.hilbert_space.dimension):
            if i < total_sections:
                section = self.sections[i]
                morphic_byte = section.to_morphic_byte()
                # Use byte value to create complex amplitude
                real_part = (morphic_byte.value & 0x0F) / 15.0  # Normalize TTTT bits
                imag_part = ((morphic_byte.value >> 4) & 0x0F) / 15.0  # Normalize upper bits
                amplitudes.append(MorphicComplex(real_part, imag_part))
            else:
                amplitudes.append(MorphicComplex(0, 0))
        
        self.quantum_state = QuantumState(amplitudes, self.hilbert_space)
    
    def extract_section(self, section_id: str) -> Optional[Tuple[str, str]]:
        """Extract a section to a separate file, returning (filename, content)."""
        section = self.find_section_by_id(section_id)
        if not section:
            return None
        
        # Generate content for extracted file
        extracted_content = f"# {section.title}\n\n{section.content}"
        filename = f"{section.morphic_id}.md"
        
        # Replace section in original document with link
        lines = self.content.split('\n')
        link_line = f"[{section.title}]({section.morphic_id})"
        
        # Replace section lines with link
        new_lines = (lines[:section.start_line] + 
                    [link_line] + 
                    lines[section.end_line + 1:])
        
        self.content = '\n'.join(new_lines)
        self._parse_sections()  # Reparse after modification
        self._create_quantum_representation()  # Recreate quantum state
        
        return filename, extracted_content
    
    def inline_section(self, section_id: str, section_content: str) -> bool:
        """Inline a previously extracted section."""
        # Find the link that references this section
        link_pattern = rf'\[([^\]]+)\]\({section_id}\)'
        match = re.search(link_pattern, self.content)
        
        if not match:
            return False
        
        title = match.group(1)
        # Determine section level based on context
        lines = self.content.split('\n')
        link_line_idx = None
        
        for i, line in enumerate(lines):
            if re.search(link_pattern, line):
                link_line_idx = i
                break
        
        if link_line_idx is None:
            return False
        
        # Determine appropriate header level
        level = 2  # Default level
        for i in range(link_line_idx - 1, -1, -1):
            header_match = re.match(r'^(#{1,6})', lines[i])
            if header_match:
                level = len(header_match.group(1)) + 1
                break
        
        # Create section header and content
        section_header = '#' * level + f' {title}'
        full_section = f"{section_header}\n\n{section_content}"
        
        # Replace link with full section
        lines[link_line_idx] = full_section
        self.content = '\n'.join(lines)
        
        self._parse_sections()  # Reparse after modification
        self._create_quantum_representation()  # Recreate quantum state
        
        return True
    
    def find_section_by_id(self, section_id: str) -> Optional[MarkdownSection]:
        """Find section by morphic ID."""
        for section in self.sections:
            if section.morphic_id == section_id:
                return section
        return None
    
    def analyze_morphic_structure(self) -> Dict[str, Any]:
        """Analyze document structure using morphic principles."""
        analysis = {
            'document_uri': self.uri,
            'total_sections': len(self.sections),
            'quantum_measurement': None,
            'morphic_bytes': [],
            'structural_entropy': 0.0
        }
        
        if self.quantum_state:
            analysis['quantum_measurement'] = self.quantum_state.measure()
        
        # Analyze each section's morphic properties
        for section in self.sections:
            morphic_byte = section.to_morphic_byte()
            analysis['morphic_bytes'].append({
                'section_id': section.morphic_id,
                'title': section.title,
                'byte_value': f"0x{morphic_byte.value:02x}",
                'binary': f"0b{morphic_byte.value:08b}",
                'level': section.level
            })
        
        # Calculate structural entropy
        if self.sections:
            level_counts = {}
            for section in self.sections:
                level_counts[section.level] = level_counts.get(section.level, 0) + 1
            
            total = len(self.sections)
            entropy = 0.0
            for count in level_counts.values():
                p = count / total
                if p > 0:
                    entropy -= p * (p.bit_length() - 1) if p < 1 else 0
            
            analysis['structural_entropy'] = entropy
        
        return analysis

class MorphicMarkdownAgent:
    """Main agent class that handles LSP operations with morphic ontology."""
    
    def __init__(self):
        self.lsp_state = LSPState()
    
    def extract_section(self, document_path: str, section_id: Optional[str] = None) -> bool:
        """Extract a section from a markdown document."""
        if not os.path.exists(document_path):
            print(f"Error: Document {document_path} not found", file=sys.stderr)
            return False
        
        # Read document
        with open(document_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Register with LSP state
        doc_state = self.lsp_state.register_document(document_path, content)
        
        if not doc_state.sections:
            print("No sections found in document", file=sys.stderr)
            return False
        
        # If no section_id provided, show available sections
        if not section_id:
            print("Available sections:")
            for section in doc_state.sections:
                print(f"  {section.morphic_id}: {section.title} (Level {section.level})")
            return True
        
        # Extract specified section
        result = doc_state.extract_section(section_id)
        if not result:
            print(f"Error: Section {section_id} not found", file=sys.stderr)
            return False
        
        filename, extracted_content = result
        
        # Write extracted file
        extracted_path = os.path.join(os.path.dirname(document_path), filename)
        with open(extracted_path, 'w', encoding='utf-8') as f:
            f.write(extracted_content)
        
        # Update original document
        with open(document_path, 'w', encoding='utf-8') as f:
            f.write(doc_state.content)
        
        print(f"Extracted section to {extracted_path}")
        return True
    
    def inline_section(self, document_path: str, section_id: str) -> bool:
        """Inline a previously extracted section."""
        if not os.path.exists(document_path):
            print(f"Error: Document {document_path} not found", file=sys.stderr)
            return False
        
        # Read document
        with open(document_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Register with LSP state
        doc_state = self.lsp_state.register_document(document_path, content)
        
        # Find and read extracted section file
        section_path = os.path.join(os.path.dirname(document_path), f"{section_id}.md")
        if not os.path.exists(section_path):
            print(f"Error: Extracted section file {section_path} not found", file=sys.stderr)
            return False
        
        with open(section_path, 'r', encoding='utf-8') as f:
            section_content = f.read()
        
        # Remove header from section content (it will be recreated)
        lines = section_content.split('\n')
        if lines and lines[0].startswith('#'):
            section_content = '\n'.join(lines[2:])  # Skip header and empty line
        
        # Inline the section
        if not doc_state.inline_section(section_id, section_content):
            print(f"Error: Could not find link to section {section_id}", file=sys.stderr)
            return False
        
        # Update original document
        with open(document_path, 'w', encoding='utf-8') as f:
            f.write(doc_state.content)
        
        print(f"Inlined section {section_id}")
        print(f"Note: You may want to manually delete {section_path}")
        return True
    
    def analyze_document(self, document_path: str) -> bool:
        """Analyze document structure using morphic principles."""
        if not os.path.exists(document_path):
            print(f"Error: Document {document_path} not found", file=sys.stderr)
            return False
        
        # Read document
        with open(document_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Register with LSP state
        doc_state = self.lsp_state.register_document(document_path, content)
        
        # Perform morphic analysis
        analysis = doc_state.analyze_morphic_structure()
        
        # Pretty print analysis
        print(json.dumps(analysis, indent=2))
        return True

  
def main():
    """Main entry point for the agent."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  md_agent.py lsp                           # Start LSP server")
        print("  md_agent.py extract [section_id] <document.md>")
        print("  md_agent.py inline <section_id> <document.md>")
        print("  md_agent.py analyze <document.md>")
        sys.exit(1)
    
    command = sys.argv[1]
    agent = MorphicMarkdownAgent()
    
    if command == "lsp":
        # Start LSP server
        lsp_server = LSPServer(agent)
        lsp_server.start()
        return
    if command == "extract":
        if len(sys.argv) == 3:
            # Show available sections
            success = agent.extract_section(sys.argv[2])
        elif len(sys.argv) == 4:
            # Extract specific section
            success = agent.extract_section(sys.argv[3], sys.argv[2])
        else:
            print("Usage: md_agent.py extract [section_id] <document.md>")
            sys.exit(1)
    
    elif command == "inline":
        if len(sys.argv) != 4:
            print("Usage: md_agent.py inline <section_id> <document.md>")
            sys.exit(1)
        success = agent.inline_section(sys.argv[3], sys.argv[2])
    
    elif command == "analyze":
        if len(sys.argv) != 3:
            print("Usage: md_agent.py analyze <document.md>")
            sys.exit(1)
        success = agent.analyze_document(sys.argv[2])
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
    
    sys.exit(0 if success else 1)    
