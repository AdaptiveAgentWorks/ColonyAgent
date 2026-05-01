"""Core Agent - LLM-driven tool orchestrator.

Based on image quality assessment results, uses LLM function calling or rule-based fallback
to select and execute preprocessing tool sequences, and finally calls the detector for colony detection.
"""

import json
import re
import time
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from loguru import logger

from tools.registry import get_tool, get_tool_schemas, list_tools
from tools.base import ToolResult
from prompts.agent_prompts import AGENT_SYSTEM_PROMPT, AGENT_USER_PROMPT

# Import detection tools to trigger @register_tool decorator
import tools.detection.yolov8_det  # noqa: F401
import tools.detection.rtdetr_det  # noqa: F401
import tools.detection.fasterrcnn_det  # noqa: F401
import tools.preprocessing  # noqa: F401


@dataclass
class AgentResult:
    """Agent processing result."""

    original_image: np.ndarray          # Original image
    processed_image: np.ndarray         # Processed image
    detections: List[Dict]              # Final detection results
    trajectory: List[Dict]             # Complete processing trajectory
    quality_report: dict                # Quality assessment result
    tools_used: List[str]              # List of tools used
    detector_used: str                 # Detector used
    total_time: float                 # Total time elapsed


class ColonyDetectionAgent:
    """LLM-driven colony detection Agent.

    Full flow: Quality Assessment -> Experience Retrieval -> Skill Adaptation -> LLM Tool Orchestration -> Execution -> Trajectory Recording.
    When LLM is unavailable, automatically falls back to rule-based tool selection.
    """

    # Default detector name
    DEFAULT_DETECTOR = "yolov8_detect"

    def __init__(
        self,
        llm_client,
        quality_assessor,
        experience_retriever,
        skill_adapter,
        skill_library_content: str = "",
        config: dict = None,
    ):
        """
        Args:
            llm_client: LLMClient instance.
            quality_assessor: ImageQualityAssessor instance.
            experience_retriever: ExperienceRetriever instance.
            skill_adapter: SkillAdapter instance.
            skill_library_content: Markdown text of the Skill library.
            config: Optional configuration dict, supported keys:
                - default_detector: Default detector name
                - max_preprocessing_steps: Max preprocessing steps (default 5)
                - quality_thresholds: Acceptable thresholds for each dimension
        """
        self.llm = llm_client
        self.assessor = quality_assessor
        self.retriever = experience_retriever
        self.skill_adapter = skill_adapter
        self.skill_library_content = skill_library_content
        self.config = config or {}

        self.default_detector = self.config.get("default_detector", self.DEFAULT_DETECTOR)
        self.max_preprocessing_steps = self.config.get("max_preprocessing_steps", 5)

        # Detector default parameters (read from config detectors section)
        detectors_config = self.config.get("detectors", {})

        def build_detector_defaults(cfg):
            if isinstance(cfg, dict):
                return cfg.copy()
            elif isinstance(cfg, str):
                return {"model_path": cfg}
            else:
                return {}

        self.detector_defaults = {
            "yolov8_detect": build_detector_defaults(detectors_config.get("yolov8", {})),
            "rtdetr_detect": build_detector_defaults(detectors_config.get("rtdetr", {})),
            "fasterrcnn_detect": build_detector_defaults(detectors_config.get("fasterrcnn", {})),
        }

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def process(self, image: np.ndarray, ground_truth: dict = None) -> AgentResult:
        """Full processing flow.

        Args:
            image: Input image in BGR format.
            ground_truth: Optional ground truth annotation for subsequent feedback.

        Returns:
            AgentResult containing processed image, detection results, trajectory, etc.
        """
        start_time = time.time()

        # 1. Quality assessment
        quality_report = self.assessor.assess(image)
        quality_dict = {
            "blur_score": quality_report.blur_score,
            "brightness_score": quality_report.brightness_score,
            "contrast_score": quality_report.contrast_score,
            "noise_score": quality_report.noise_score,
            "color_bias_score": quality_report.color_bias_score,
            "overall_score": quality_report.overall_score,
            "raw_metrics": quality_report.raw_metrics,
        }
        logger.info(f"Quality assessment done: overall={quality_report.overall_score:.3f}")

        # 2. Get Experience - always use retrieval filtering, top_k limits quantity
        all_experiences = self.retriever.manager.get_all()
        quality_vector = self.assessor.to_feature_vector(quality_report)
        experiences = self.retriever.retrieve_by_quality(quality_vector)
        logger.info(f"Retrieved {len(experiences)} experiences from {len(all_experiences)} total")

        # 3. Adapt Skill
        adapted_skill = ""
        if self.skill_library_content:
            try:
                adapted_skill = self.skill_adapter.adapt(
                    base_skill=self.skill_library_content,
                    quality_report=quality_dict,
                    retrieved_experiences=experiences,
                )
            except Exception as e:
                logger.warning(f"Skill adaptation failed: {e}")

        # 4. LLM decision -> tool call sequence
        try:
            messages = self._build_messages(quality_dict, experiences, adapted_skill)
            tool_schemas = get_tool_schemas()
            llm_response = self.llm.chat_with_tools(messages, tools=tool_schemas)
            tool_calls = self._parse_tool_calls(llm_response)
            logger.info(f"LLM decided {len(tool_calls)} tool calls")
        except Exception as e:
            logger.warning(f"LLM tool planning failed ({e}), falling back to rules")
            tool_calls = self._default_tool_sequence(quality_dict)

        # 5. Execute tool sequence
        processed_image, detections, trajectory = self._execute_tool_sequence(image, tool_calls)

        # Determine the detector used
        detector_used = self.default_detector
        tools_used = []
        for step in trajectory:
            tools_used.append(step.get("tool", "unknown"))
            if step.get("type") == "detection":
                detector_used = step["tool"]

        total_time = time.time() - start_time

        return AgentResult(
            original_image=image,
            processed_image=processed_image,
            detections=detections,
            trajectory=trajectory,
            quality_report=quality_dict,
            tools_used=tools_used,
            detector_used=detector_used,
            total_time=total_time,
        )

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _build_messages(
        self, quality_report: dict, experiences: List[Dict], adapted_skill: str
    ) -> List[Dict]:
        """Build messages to send to LLM (system + user).

        Injects Skill, Experience, quality report, and tool schemas into the prompt.
        """
        # Format quality report
        quality_text = json.dumps(quality_report, indent=2, ensure_ascii=False)

        # Format experiences
        if experiences:
            exp_parts = []
            for i, exp in enumerate(experiences):
                sim = exp.get("_similarity", "N/A")
                exp_parts.append(
                    f"- Experience {i + 1} (sim={sim}): "
                    f"condition={exp.get('condition', 'N/A')}, "
                    f"action={exp.get('action', 'N/A')}, "
                    f"reason={exp.get('reason', 'N/A')}"
                )
            experiences_text = "\n".join(exp_parts)
        else:
            experiences_text = "No relevant experiences retrieved."

        # Format tool list
        available_tools = list_tools()
        tools_description = "\n".join(f"- {name}" for name in available_tools)

        # Build system prompt
        system_content = AGENT_SYSTEM_PROMPT.format(
            quality_report=quality_text,
            retrieved_experiences=experiences_text,
            adapted_skill=adapted_skill or "No adapted skill available.",
            available_tools=tools_description,
        )

        # Build user prompt
        # Quality summary
        dims = ["blur_score", "brightness_score", "contrast_score", "noise_score", "color_bias_score"]
        low_dims = [d for d in dims if quality_report.get(d, 1.0) < 0.5]
        if low_dims:
            quality_summary = f"Low quality dimensions: {', '.join(low_dims)}"
        else:
            quality_summary = "All quality dimensions are acceptable (>= 0.5)."

        user_content = AGENT_USER_PROMPT.format(
            image_description=f"Microscopy image for colony detection. Shape: unknown.",
            quality_summary=quality_summary,
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Parse LLM response
    # ------------------------------------------------------------------

    def _parse_tool_calls(self, llm_response: dict) -> List[Dict]:
        """Parse tool call sequence from LLM response.

        Supports two formats:
        1. OpenAI function calling format (tool_calls field)
        2. JSON embedded in text format

        Returns:
            [{tool_name: str, params: dict}, ...]
        """
        tool_calls = []

        # 1) Try parsing from function calling format
        if llm_response.get("tool_calls"):
            for tc in llm_response["tool_calls"]:
                name = tc.get("name", "")
                args_str = tc.get("arguments", "{}")
                try:
                    params = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    params = {}
                tool_calls.append({"tool_name": name, "params": params})

        # 2) If no function calling results, try parsing from text
        if not tool_calls:
            content = llm_response.get("content", "")
            tool_calls = self._parse_tool_calls_from_text(content)

        # Ensure detector is at the end
        self._ensure_detector(tool_calls)

        return tool_calls

    def _parse_tool_calls_from_text(self, text: str) -> List[Dict]:
        """Parse JSON tool calls from LLM text output."""
        tool_calls = []

        # Extract all JSON blocks
        json_blocks = re.findall(r"```json\s*(.*?)```", text, re.DOTALL)
        # Also try to match standalone JSON objects
        if not json_blocks:
            json_blocks = re.findall(r"\{[^{}]*\}", text)

        for block in json_blocks:
            block = block.strip()
            try:
                data = json.loads(block)
            except json.JSONDecodeError:
                continue

            # Skip summary blocks
            if "summary" in data and "tool" not in data:
                continue

            if "tool" in data:
                tool_calls.append({
                    "tool_name": data["tool"],
                    "params": data.get("parameters", {}),
                })

        return tool_calls

    def _resolve_tool_name(self, name: str) -> str:
        """Fuzzy match tool names, handling cases where LLM returns imprecise names."""
        registered = set(list_tools())
        if name in registered:
            return name

        # Common alias mapping
        aliases = {
            "sharpener": "sharpen",
            "sharp": "sharpen",
            "color_correct": "white_balance",
            "color_corrector": "white_balance",
            "color_correction": "white_balance",
            "detect": "yolov8_detect",
            "detector": "yolov8_detect",
            "colony_detect": "yolov8_detect",
            "colony_detector": "yolov8_detect",
            "yolo": "yolov8_detect",
            "yolo_detect": "yolov8_detect",
            "rtdetr": "rtdetr_detect",
            "rt_detr": "rtdetr_detect",
            "faster_rcnn": "fasterrcnn_detect",
            "fasterrcnn": "fasterrcnn_detect",
            "clahe": "clahe_enhance",
            "contrast_enhance": "clahe_enhance",
            "contrast": "clahe_enhance",
            "denoise_image": "denoise",
            "denoiser": "denoise",
            "noise_reduction": "denoise",
            "illumination": "illumination_correct",
            "light_correction": "illumination_correct",
            "roi": "roi_extract",
            "crop_roi": "roi_extract",
            "white_bal": "white_balance",
            "wb": "white_balance",
            "watershed": "colony_separate",
            "separate": "colony_separate",
            "resize_image": "resize",
            "scale": "resize",
        }
        if name.lower() in aliases:
            return aliases[name.lower()]

        # Partial match: check if registered name is contained in the name returned by LLM
        for reg_name in registered:
            if reg_name in name.lower() or name.lower() in reg_name:
                return reg_name

        return name  # Cannot match, return original name

    def _validate_tool_params(self, tool, params: dict) -> tuple:
        """Validate whether LLM-generated parameters conform to the tool schema.

        Returns:
            (is_valid, error_message)
        """
        schema = getattr(tool, "parameters", {})
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Check required parameters
        for req in required:
            if req not in params:
                return False, f"Missing required parameter: {req}"

        # Check parameter types and enum constraints
        for param_name, param_value in params.items():
            if param_name not in properties:
                # Unknown parameter, log warning but don't block
                logger.warning(f"  Unknown parameter '{param_name}' for tool {tool.name}, ignoring")
                continue

            param_schema = properties[param_name]

            # Check enum constraints
            if "enum" in param_schema:
                allowed = param_schema["enum"]
                if param_value not in allowed:
                    return False, f"Invalid value '{param_value}' for '{param_name}', allowed: {allowed}"

        return True, None

    def _ensure_detector(self, tool_calls: List[Dict]):
        """Ensure the tool sequence ends with a detector."""
        # Check if there is already a detection-type tool
        detection_tools = {"yolov8_detect", "rtdetr_detect", "fasterrcnn_detect"}
        registered = set(list_tools())
        detection_in_registered = detection_tools & registered

        has_detector = any(
            tc["tool_name"] in detection_in_registered for tc in tool_calls
        )

        if not has_detector:
            # Use default detector
            detector = self.default_detector if self.default_detector in registered else None
            if detector is None and detection_in_registered:
                detector = sorted(detection_in_registered)[0]
            if detector:
                tool_calls.append({"tool_name": detector, "params": {}})

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool_sequence(
        self, image: np.ndarray, tool_calls: List[Dict]
    ) -> tuple:
        """Execute tool calls in order.

        Args:
            image: Input image.
            tool_calls: [{tool_name, params}, ...]

        Returns:
            (processed_image, detections, trajectory_steps)
        """
        current_image = image.copy()
        detections = []
        trajectory = []

        # Detection-type tool name set (used to distinguish preprocessing from detection)
        detection_keywords = {"detect", "detector", "detection", "count"}

        for step_idx, tc in enumerate(tool_calls):
            tool_name = tc["tool_name"]
            params = tc.get("params", {})
            step_start = time.time()

            # Determine step type
            is_detection = any(kw in tool_name.lower() for kw in detection_keywords)
            step_type = "detection" if is_detection else "preprocessing"

            # Force use of configured detector, ignore LLM's choice (preserve experience memory)
            if is_detection and self.default_detector:
                original_tool = tool_name
                tool_name = self.default_detector
                if original_tool != tool_name:
                    logger.info(f"Detection tool overridden: '{original_tool}' -> '{tool_name}'")

            step_record = {
                "step": step_idx + 1,
                "type": step_type,
                "tool": tool_name,
                "params": params,
                "timestamp": time.time(),
            }

            # Fuzzy match tool names (LLM may return imprecise names)
            if not is_detection:  # Only preprocessing tools need resolve, detector already forced to use default
                resolved_name = self._resolve_tool_name(tool_name)
                if resolved_name != tool_name:
                    logger.info(f"Tool name resolved: '{tool_name}' -> '{resolved_name}'")
                    tool_name = resolved_name
                    step_record["tool"] = tool_name

            try:
                tool = get_tool(tool_name)
                # Detector automatically injects default parameters (model_path, device, num_classes, etc.)
                if tool_name in self.detector_defaults:
                    defaults = self.detector_defaults[tool_name].copy()
                    # Config model_path, device, num_classes are authoritative, LLM params can only fill other optional parameters
                    params = {**params, **defaults}
                    step_record["params"] = params

                # Parameter validation: prevent LLM hallucinated parameter values
                is_valid, val_error = self._validate_tool_params(tool, params)
                if not is_valid:
                    logger.warning(f"  Parameter validation failed for {tool_name}: {val_error}")
                    step_record["success"] = False
                    step_record["error"] = f"param_validation_failed: {val_error}"
                    # When parameter validation fails, skip remaining preprocessing and use original image for detection
                    break
                result: ToolResult = tool.call(current_image, **params)

                if result.success:
                    current_image = result.image
                    if result.detections:
                        detections = result.detections  # Take last detection result

                    step_record["success"] = True
                    step_record["result"] = {
                        "metadata": result.metadata,
                        "num_detections": len(result.detections),
                    }
                    logger.info(
                        f"Step {step_idx + 1}: {tool_name} succeeded "
                        f"(detections={len(result.detections)})"
                    )
                else:
                    step_record["success"] = False
                    step_record["error"] = result.error or "Tool returned failure"
                    logger.warning(f"Step {step_idx + 1}: {tool_name} failed: {result.error}")
                    # When preprocessing tool fails, skip remaining preprocessing steps and proceed to detection
                    if step_type == "preprocessing":
                        logger.warning(f"  Preprocessing failed, falling back to direct detection")
                        break

            except KeyError:
                step_record["success"] = False
                step_record["error"] = f"Tool '{tool_name}' not registered"
                logger.error(f"Step {step_idx + 1}: Tool '{tool_name}' not found in registry")
            except Exception as e:
                step_record["success"] = False
                step_record["error"] = str(e)
                logger.error(f"Step {step_idx + 1}: {tool_name} raised exception: {e}")

            step_record["elapsed"] = time.time() - step_start
            trajectory.append(step_record)

        return current_image, detections, trajectory

    # ------------------------------------------------------------------
    # LLM fallback mode
    # ------------------------------------------------------------------

    def process_without_llm(
        self, image: np.ndarray, tool_sequence: List[Dict] = None
    ) -> AgentResult:
        """Fallback mode without LLM.

        Args:
            image: Input image in BGR format.
            tool_sequence: Optional predefined tool sequence [{tool_name, params}, ...].
                If not provided, selects tools using simple rules based on quality report.

        Returns:
            AgentResult
        """
        start_time = time.time()

        # Quality assessment
        quality_report = self.assessor.assess(image)
        quality_dict = {
            "blur_score": quality_report.blur_score,
            "brightness_score": quality_report.brightness_score,
            "contrast_score": quality_report.contrast_score,
            "noise_score": quality_report.noise_score,
            "color_bias_score": quality_report.color_bias_score,
            "overall_score": quality_report.overall_score,
            "raw_metrics": quality_report.raw_metrics,
        }

        # Determine tool sequence
        if tool_sequence is None:
            tool_sequence = self._default_tool_sequence(quality_dict)

        # Execute
        processed_image, detections, trajectory = self._execute_tool_sequence(
            image, tool_sequence
        )

        # Determine detector
        detector_used = self.default_detector
        tools_used = []
        for step in trajectory:
            tools_used.append(step.get("tool", "unknown"))
            if step.get("type") == "detection":
                detector_used = step["tool"]

        total_time = time.time() - start_time

        return AgentResult(
            original_image=image,
            processed_image=processed_image,
            detections=detections,
            trajectory=trajectory,
            quality_report=quality_dict,
            tools_used=tools_used,
            detector_used=detector_used,
            total_time=total_time,
        )

    def _default_tool_sequence(self, quality_report: dict) -> List[Dict]:
        """Default rules based on quality report (no LLM needed).

        Rules:
        - brightness < 0.4  -> clahe_enhance
        - noise_score < 0.4 -> denoise
        - blur_score < 0.3  -> sharpen
        - Finally select a detector (default yolov8_detect)
        """
        registered = set(list_tools())
        tool_calls = []

        # Low brightness/contrast -> CLAHE
        if quality_report.get("brightness_score", 1.0) < 0.4:
            if "clahe_enhance" in registered:
                tool_calls.append({"tool_name": "clahe_enhance", "params": {}})

        # Too much noise -> denoise
        if quality_report.get("noise_score", 1.0) < 0.4:
            if "denoise" in registered:
                tool_calls.append({"tool_name": "denoise", "params": {}})

        # Blurry -> sharpen
        if quality_report.get("blur_score", 1.0) < 0.3:
            if "sharpen" in registered:
                tool_calls.append({"tool_name": "sharpen", "params": {}})

        # Low contrast (and CLAHE not already used) -> contrast enhancement
        if quality_report.get("contrast_score", 1.0) < 0.4:
            if "clahe_enhance" not in [tc["tool_name"] for tc in tool_calls]:
                if "clahe_enhance" in registered:
                    tool_calls.append({"tool_name": "clahe_enhance", "params": {}})

        # Detector
        self._ensure_detector(tool_calls)

        return tool_calls
