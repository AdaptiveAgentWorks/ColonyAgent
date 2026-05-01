"""
Tool Registry - Provides decorator-based registration and lookup utilities.
"""

from typing import Dict, List

from tools.base import BaseTool

# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------
_TOOL_REGISTRY: Dict[str, BaseTool] = {}


def register_tool(cls):
    """
    Class decorator that instantiates the tool and registers it globally.

    Usage::

        @register_tool
        class MyTool(BaseTool):
            name = "my_tool"
            ...
    """
    instance = cls()
    _TOOL_REGISTRY[instance.name] = instance
    return cls


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_tool(name: str) -> BaseTool:
    """Return a registered tool by name. Raises KeyError if not found."""
    if name not in _TOOL_REGISTRY:
        raise KeyError(f"Tool '{name}' is not registered. Available: {list(_TOOL_REGISTRY.keys())}")
    return _TOOL_REGISTRY[name]


def list_tools() -> List[str]:
    """Return a sorted list of all registered tool names."""
    return sorted(_TOOL_REGISTRY.keys())


def get_all_tools() -> Dict[str, BaseTool]:
    """Return the full registry dictionary (name -> instance)."""
    return dict(_TOOL_REGISTRY)


def get_tool_schemas() -> List[dict]:
    """Return OpenAI function-calling schemas for every registered tool."""
    return [tool.get_schema() for tool in _TOOL_REGISTRY.values()]
