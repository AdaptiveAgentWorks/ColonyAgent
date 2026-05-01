"""
Online Tool Generator - Fallback for when all rollouts fail.

When Phase1 multi-rollout exploration fails to find an effective preprocessing pipeline,
this module searches for solutions online and generates new tools dynamically.

The search provider and LLM are configured via config file - no hardcoded dependencies.
"""

import json
import hashlib
import re
from typing import List, Dict, Optional
from pathlib import Path

from loguru import logger

# Import prompts from prompts directory
try:
    from prompts.tool_generation_prompts import get_tool_generation_prompt
except ImportError:
    # Fallback if prompts not available
    def get_tool_generation_prompt(search_results, deg_types):
        deg_str = " + ".join(deg_types) if deg_types else "enhancement"
        return f"""Based on the following search results, generate an OpenCV preprocessing tool for microbial colony detection.

Search results:
{search_results[:2000]}

CRITICAL REQUIREMENTS:
1. The class MUST end with "Tool"
2. Must inherit from tools.base.BaseTool
3. MUST define name and description as CLASS attributes
4. MUST implement call(self, image, **params) -> ToolResult
5. Use OpenCV for implementation
6. Optimize for improving detection recall

Output only the code, no explanations."""


def generate_fallback_tool(
    image_path: str,
    quality_report: dict,
    attempted_tools: List[str],
    rollout_verdicts: List[str],
    config: dict
) -> dict:
    """
    When all rollouts fail, search online and generate a new tool.

    Args:
        image_path: Path to the degraded image
        quality_report: Quality assessment report with blur_score, contrast_score, etc.
        attempted_tools: List of tool names already tried
        rollout_verdicts: List of verdicts from each rollout
        config: Full config dict with llm and fallback settings

    Returns:
        dict with keys: success, tool_name, params, generated_file
    """
    logger.info(f"Online fallback triggered for {image_path}")
    logger.info(f"  Attempted tools: {attempted_tools}")
    logger.info(f"  Verdicts: {rollout_verdicts}")

    fallback_config = config.get("fallback", {})
    llm_config = config.get("llm", {})

    # Step 1: Build search query based on quality issues
    deg_types = []
    if quality_report.get("blur_score", 1.0) < 0.4:
        deg_types.append("blur")
    if quality_report.get("contrast_score", 1.0) < 0.4:
        deg_types.append("low contrast")
    if quality_report.get("noise_score", 1.0) < 0.4:
        deg_types.append("noisy")
    if quality_report.get("brightness_score", 1.0) < 0.4:
        deg_types.append("darkness")
    if not deg_types:
        deg_types = ["image enhancement"]

    search_query = f"OpenCV {' '.join(deg_types)} colony detection improving recall"

    # Step 2: Search for solutions using configured provider
    search_text = _search_online(search_query, fallback_config)

    # Step 3: Generate tool code using configured LLM
    code = _generate_tool_code(search_text, deg_types, llm_config)
    if not code:
        logger.warning("  Failed to generate tool code")
        return {"success": False, "error": "code generation failed"}

    # Step 4: Validate and save the tool
    return _validate_and_register_tool(code)


def _search_online(query: str, fallback_config: dict) -> str:
    """Search online using configured provider."""
    provider = fallback_config.get("search_provider", "mmx")

    if provider == "mmx":
        return _search_with_mmx(query, fallback_config)
    elif provider == "serpapi":
        return _search_with_serpapi(query, fallback_config)
    else:
        logger.warning(f"  Unknown search provider: {provider}, using mmx as fallback")
        return _search_with_mmx(query, fallback_config)


def _search_with_mmx(query: str, fallback_config: dict) -> str:
    """Search using mmx CLI."""
    try:
        import subprocess
        result = subprocess.run(
            ["mmx", "search", "query", "--q", query, "--output", "json", "--quiet"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode != 0:
            logger.warning(f"  Search failed: {result.stderr}")
            return ""

        data = json.loads(result.stdout)
        return data.get("content", "") or data.get("text", "") or json.dumps(data)
    except Exception as e:
        logger.warning(f"  MMX search error: {e}")
        return ""


def _search_with_serpapi(query: str, fallback_config: dict) -> str:
    """Search using SerpAPI."""
    try:
        import subprocess
        api_key = fallback_config.get("serpapi_key", "")
        if not api_key:
            # Try environment variable
            import os
            api_key = os.environ.get("SERPAPI_API_KEY", "")

        if not api_key:
            logger.warning("  SerpAPI key not configured")
            return ""

        cmd = [
            "curl", "-s", "-G", "https://serpapi.com/search",
            "-d", f"q={query}",
            "-d", f"api_key={api_key}",
            "--max-time", "30"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
        if result.returncode != 0:
            return ""

        data = json.loads(result.stdout)
        # Extract relevant snippets
        results = data.get("organic_results", [])
        snippets = [r.get("snippet", "") for r in results[:5]]
        return " ".join(snippets)
    except Exception as e:
        logger.warning(f"  SerpAPI search error: {e}")
        return ""


def _generate_tool_code(search_text: str, deg_types: List[str], llm_config: dict) -> str:
    """Generate tool code using the configured LLM client."""
    try:
        from llm.client import LLMClient
        import re

        # Create LLM client from config
        client = LLMClient(llm_config)

        # Use prompt from prompts directory
        prompt = get_tool_generation_prompt(search_text, deg_types)

        messages = [
            {"role": "system", "content": "You are an image processing expert."},
            {"role": "user", "content": prompt}
        ]

        # Use more tokens for code generation
        code = client.chat(messages, temperature=0.2, max_tokens=8192)

        # Clean up thinking blocks if present (MiniMax models include these)
        code = re.sub(r'<think>.*?</think>', '', code, flags=re.DOTALL)

        # Also remove any leading whitespace before code blocks
        code = code.strip()

        return code

    except Exception as e:
        logger.warning(f"  LLM code generation error: {e}")
        return ""


def _validate_and_register_tool(code: str) -> dict:
    """Validate generated code and register it."""
    # Extract code block
    code_match = re.search(r"```python\s*(.*?)```", code, re.DOTALL)
    if code_match:
        tool_code = code_match.group(1)
    else:
        tool_code = code.strip()

    if not tool_code or len(tool_code) < 100:
        logger.warning(f"  Generated code too short")
        return {"success": False, "error": "code too short"}

    # Save to file
    tool_hash = hashlib.md5(tool_code.encode()).hexdigest()[:8]
    tools_dir = Path(__file__).parent.parent / "tools" / "preprocessing" / "generated"
    tools_dir.mkdir(parents=True, exist_ok=True)
    filepath = tools_dir / f"dynamic_{tool_hash}.py"

    with open(filepath, "w") as f:
        f.write(tool_code)

    logger.info(f"  Saved generated tool to {filepath}")

    # Load and validate
    try:
        import importlib.util
        from tools.base import BaseTool
        from tools.registry import register_tool

        spec = importlib.util.spec_from_file_location(f"dynamic_{tool_hash}", str(filepath))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find tool class - must inherit from tools.base.BaseTool (not a local BaseTool)
        tool_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and
                issubclass(attr, BaseTool) and
                attr_name.endswith("Tool") and
                attr is not BaseTool):  # Don't accept the BaseTool class itself
                tool_class = attr
                break

        if tool_class is None:
            logger.warning(f"  No Tool class found - checking if module defines its own BaseTool")
            # Check if the module defined its own BaseTool (wrong) instead of importing
            if hasattr(module, 'BaseTool') and module.BaseTool is not BaseTool:
                logger.warning(f"  Module defines its own BaseTool instead of importing from tools.base")
                return {"success": False, "error": "wrong BaseTool inheritance"}
            return {"success": False, "error": "no Tool class found"}

        # Verify call method exists
        if not hasattr(tool_class, 'call') or not callable(getattr(tool_class, 'call', None)):
            logger.warning(f"  Generated class missing call method")
            return {"success": False, "error": "missing call method"}

        # Register
        register_tool(tool_class)
        tool_name = tool_class().name

        logger.info(f"  Successfully registered tool: {tool_name}")

        # Save to persistent registry
        _save_tool_registry(tools_dir, tool_name, filepath)

        return {
            "success": True,
            "tool_name": tool_name,
            "params": {},
            "generated_file": str(filepath)
        }

    except SyntaxError as e:
        logger.warning(f"  Syntax error in generated code: {e}")
        return {"success": False, "error": f"syntax error: {e}"}
    except Exception as e:
        logger.warning(f"  Failed to load generated tool: {e}")
        return {"success": False, "error": str(e)}


def execute_tool(tool_name: str, image_path: str, params: dict = None) -> dict:
    """
    Execute a registered tool on an image.

    Args:
        tool_name: Name of the tool to execute
        image_path: Path to input image
        params: Tool parameters

    Returns:
        dict with keys: success, image (numpy array or None), error
    """
    import cv2
    from tools.registry import get_tool

    try:
        tool = get_tool(tool_name)
        image = cv2.imread(image_path)
        if image is None:
            return {"success": False, "error": "failed to load image"}

        result = tool.call(image, **(params or {}))
        return {
            "success": result.success,
            "image": result.image,
            "error": result.error
        }
    except KeyError as e:
        return {"success": False, "error": f"tool not found: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _save_tool_registry(tools_dir: Path, tool_name: str, filepath: Path):
    """Save generated tool info to registry file."""
    import json
    registry_file = tools_dir / "generated_tools.json"
    registry = {}
    if registry_file.exists():
        try:
            with open(registry_file, "r") as f:
                registry = json.load(f)
        except:
            registry = {}
    registry[tool_name] = str(filepath)
    with open(registry_file, "w") as f:
        json.dump(registry, f, indent=2)
    logger.info(f"  Saved tool registry: {registry_file}")


def load_generated_tools():
    """Load all previously generated tools from registry."""
    import json
    import importlib.util
    from tools.base import BaseTool
    from tools.registry import register_tool

    tools_dir = Path(__file__).parent.parent / "tools" / "preprocessing" / "generated"
    registry_file = tools_dir / "generated_tools.json"

    if not registry_file.exists():
        logger.info("  No generated tools registry found")
        return

    try:
        with open(registry_file, "r") as f:
            registry = json.load(f)
    except Exception as e:
        logger.warning(f"  Failed to load registry: {e}")
        return

    loaded_count = 0
    for tool_name, filepath in registry.items():
        try:
            spec = importlib.util.spec_from_file_location(f"generated_{tool_name}", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find tool class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    issubclass(attr, BaseTool) and
                    attr_name.endswith("Tool") and
                    attr is not BaseTool):
                    register_tool(attr)
                    loaded_count += 1
                    logger.info(f"  Loaded generated tool: {attr_name}")
                    break
        except Exception as e:
            logger.warning(f"  Failed to load {tool_name}: {e}")

    logger.info(f"  Loaded {loaded_count} generated tools from registry")
