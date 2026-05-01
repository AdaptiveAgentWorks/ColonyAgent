"""
Schema Builder - Generate OpenAI function-calling schema lists from tool instances.
"""

from typing import List, Dict, Any

from tools.base import BaseTool


def build_schemas(tools: List[BaseTool]) -> List[Dict[str, Any]]:
    """
    Build a list of OpenAI function-calling schemas from tool instances.

    Args:
        tools: A list of BaseTool instances.

    Returns:
        A list of schema dicts ready for the ``tools`` parameter of the
        OpenAI Chat Completions API.
    """
    return [tool.get_schema() for tool in tools]


def build_schemas_from_registry() -> List[Dict[str, Any]]:
    """
    Convenience wrapper: build schemas from every tool currently registered
    in the global registry.
    """
    from tools.registry import get_all_tools

    return build_schemas(list(get_all_tools().values()))
