#!/usr/bin/env python3
# scripts/app_bridge.py
import sys
import json
import textwrap
from types import ModuleType
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ContentModule:
    """Represents a content module with metadata and wrapped content.
    'content' is non-python source code and multi-media; the knowledge base."""
    original_path: Path
    module_name: str
    content: str
    is_python: bool

    def generate_module_content(self) -> str:
        """Generate the Python module content with self-invoking functionality."""
        if self.is_python:
            return self.content

        # Escape content to prevent string literal issues
        escaped_content = self.content.replace(
            '"""', r'\"\"\"').replace('\\', '\\\\')

        return f'''"""
Original file: {self.original_path}
Auto-generated content module
"""

ORIGINAL_PATH = "{self.original_path}"
CONTENT = """{escaped_content}"""

def get_content() -> str:
    """Returns the original content."""
    return CONTENT

def get_metadata() -> dict:
    """Metadata for the original file."""
    return {{
        "original_path": ORIGINAL_PATH,
        "is_python": {self.is_python},
        "module_name": "{self.module_name}"
    }}

def default_behavior() -> None:
    """Default behavior when module is loaded."""
    print(f"Content module '{self.module_name}' loaded from {{ORIGINAL_PATH}}")
    return True

# Execute default behavior on import
default_behavior()
'''


@dataclass(frozen=True)
class InstanceConfig:
    """Configuration for a dynamic instance."""
    id: str
    name: str
    module_name: str

    @classmethod
    def from_dict(cls, data: dict) -> 'InstanceConfig':
        """Create InstanceConfig from dictionary."""
        instance_id = data["id"]
        return cls(
            id=instance_id,
            name=data["name"],
            module_name=f"morphological.instance_{instance_id}"
        )


def create_module(module_name: str, module_code: str, main_module_path: str) -> Optional[ModuleType]:
    """
    Dynamically creates a module with the specified name, injects code into it,
    and adds it to sys.modules.

    Args:
        module_name: Name of the module to create.
        module_code: Source code to inject into the module.
        main_module_path: File path of the main module.

    Returns:
        The dynamically created module, or None if an error occurs.
    """
    dynamic_module = ModuleType(module_name)
    dynamic_module.__file__ = main_module_path or "runtime_generated"
    dynamic_module.__package__ = module_name
    dynamic_module.__path__ = None
    dynamic_module.__doc__ = None

    try:
        exec(module_code, dynamic_module.__dict__)
        sys.modules[module_name] = dynamic_module
        return dynamic_module
    except Exception as e:
        print(
            f"Error injecting code into module {module_name}: {e}", file=sys.stderr)
        return None


def validate_instance(instance: dict) -> bool:
    """Validate instance JSON schema."""
    required_keys = ["id", "name"]
    return all(key in instance for key in required_keys)


def create_content_module_from_instance(config: InstanceConfig) -> ContentModule:
    """Create a ContentModule from an InstanceConfig."""
    module_code = textwrap.dedent(f'''
        def greet():
            print("Hello from instance {config.id} in module: {config.module_name}")
            return f"Instance {{'{config.id}'}} - {{'{config.name}'}}"
        
        def get_instance_info():
            return {{
                "id": "{config.id}",
                "name": "{config.name}",
                "module_name": "{config.module_name}"
            }}
    ''')

    return ContentModule(
        original_path=Path("runtime_generated"),
        module_name=config.module_name,
        content=module_code,
        is_python=True
    )


def main() -> int:
    """
    Process instance JSON from stdin, create a dynamic module, and execute it.
    Returns an exit code (0 for success, 1 for failure).
    """
    try:
        # Read and parse stdin once
        raw_input = sys.stdin.read().strip()
        if not raw_input:
            print("Error: No input provided", file=sys.stderr)
            return 1

        instance_data = json.loads(raw_input)
        if not validate_instance(instance_data):
            print("Error: Invalid instance schema", file=sys.stderr)
            return 1

        # Create configuration using frozen dataclass
        config = InstanceConfig.from_dict(instance_data)

        # Create content module
        content_module = create_content_module_from_instance(config)

        # Generate module content
        module_code = content_module.generate_module_content()

        # Get main module path
        main_module_path = getattr(
            sys.modules['__main__'], '__file__', 'runtime_generated')

        # Create and execute dynamic module
        dynamic_module = create_module(
            config.module_name, module_code, main_module_path)
        if not dynamic_module:
            print(
                f"Error: Failed to create module {config.module_name}", file=sys.stderr)
            return 1

        # Optional: Integrate with EnhancedRuntimeSystem for provenance
        # ers = EnhancedRuntimeSystem()
        # asyncio.run(ers.add_document(result, metadata={"instance_id": config.id}))

        # Execute the module's greet function
        result = dynamic_module.greet()
        print(f"Module execution result: {result}")

        return 0

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input - {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Example usage for testing:
    
    

    # echo '{"id": "test_001", "name": "Test Instance"}' | python app_bridge.py
    sys.exit(main())
