"""Phase II: Online Inference + Continual Learning Pipeline

Process new images using accumulated knowledge base while continuously updating knowledge.
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional

from loguru import logger

# Import tool modules to trigger @register_tool decorator registration
import tools.preprocessing  # noqa: F401
import tools.detection  # noqa: F401

from llm.client import LLMClient
from quality.assessor import ImageQualityAssessor
from knowledge.experience_manager import ExperienceManager
from knowledge.experience_retriever import ExperienceRetriever
from knowledge.skill_builder import SkillBuilder
from knowledge.skill_adapter import SkillAdapter
from knowledge.trajectory_summary import TrajectorySummarizer
from knowledge.experience_critique import ExperienceCritic
from agent.colony_agent import ColonyDetectionAgent, AgentResult
from agent.feedback import FeedbackLoop
from tools.registry import get_tool
from utils.image_utils import load_image
from utils.metrics import compare_detections


class Phase2Pipeline:
    """
    Online Inference + Continual Learning Pipeline:
    Process new images using accumulated knowledge base while continuously updating knowledge.
    """

    def __init__(self, config: dict, memory_dir: str = None):
        """
        Initialize all components + load existing memory bank.

        Args:
            config: Configuration dictionary.
            memory_dir: Memory bank directory path containing experience_bank/ and skill_library/ subdirs.
        """
        self.config = config

        # LLM
        llm_config = config.get("llm", {})
        embedding_config = config.get("embedding", {})
        llm_config["embedding"] = embedding_config
        self.llm = LLMClient(llm_config)

        # Quality Assessor
        quality_config = config.get("quality", {})
        self.assessor = ImageQualityAssessor(self._build_quality_config(quality_config))

        # Memory directories
        memory_config = config.get("memory", {})
        if memory_dir:
            experience_bank_dir = os.path.join(memory_dir, "experience_bank")
            skill_library_dir = os.path.join(memory_dir, "skill_library")
        else:
            experience_bank_dir = memory_config.get("experience_bank_dir", "memory/experience_bank")
            skill_library_dir = memory_config.get("skill_library_dir", "memory/skill_library")

        # Phase2 config
        phase2_config = config.get("phase2", {})
        self.enable_continual_learning = phase2_config.get("enable_continual_learning", True)
        top_k = phase2_config.get("top_k", 3)
        similarity_threshold = phase2_config.get("similarity_threshold", 0.6)
        # use_gt_feedback: whether to use ground truth for feedback evaluation
        # True: use GT annotations for feedback (Phase 1 memory construction only)
        # False: use only proxy metrics without GT (for testing to avoid test-time leakage)
        # Default is False for safety - Phase 2 is primarily for inference/testing.
        # IMPORTANT: when False, annotations are NOT passed into the agent or feedback loop.
        self.use_gt_feedback = phase2_config.get("use_gt_feedback", False)

        # Only for offline diagnostics/logging. It must not be used to select
        # detection results or update memory during normal test-time inference.
        self.enable_gt_analysis = phase2_config.get("enable_gt_analysis", False)

        # Experience Manager & Retriever
        self.experience_manager = ExperienceManager(
            llm_client=self.llm,
            save_dir=experience_bank_dir,
            similarity_threshold=similarity_threshold,
        )
        self.experience_retriever = ExperienceRetriever(
            llm_client=self.llm,
            experience_manager=self.experience_manager,
            top_k=top_k,
            min_similarity=similarity_threshold,
        )

        # Skill Builder & Adapter
        self.skill_builder = SkillBuilder(llm_client=self.llm, save_dir=skill_library_dir)
        self.skill_adapter = SkillAdapter(llm_client=self.llm)

        # Trajectory Summarizer & Critic (for continual learning)
        self.summarizer = TrajectorySummarizer(llm_client=self.llm)
        self.critic = ExperienceCritic(llm_client=self.llm)

        # Load existing memory
        self.experience_manager.load()
        self.skill_builder.load()
        logger.info(
            f"Loaded memory: {len(self.experience_manager.experiences)} experiences, "
            f"{len(self.skill_builder.skills)} skills"
        )

        # Agent - requires detectors config
        # Get correct detector from config["detectors"]["active"]
        active_detector = config.get("detectors", {}).get("active", "yolov8_detect")
        # Convert to tool name format: "rtdetr" -> "rt_detr" or "rtdetr_detect"
        detector_tool_map = {
            "rtdetr": "rtdetr_detect",
            "yolov8": "yolov8_detect",
            "fasterrcnn": "fasterrcnn_detect",
        }
        default_detector = detector_tool_map.get(active_detector, active_detector + "_detect")

        agent_config = {
            "default_detector": default_detector,
            "max_preprocessing_steps": 5,
            "detectors": config.get("detectors", {}),
        }
        self.agent = ColonyDetectionAgent(
            llm_client=self.llm,
            quality_assessor=self.assessor,
            experience_retriever=self.experience_retriever,
            skill_adapter=self.skill_adapter,
            skill_library_content=self.skill_builder.get_all_skills(),
            config=agent_config,
        )

        # Feedback Loop
        self.feedback = FeedbackLoop(
            llm_client=self.llm,
            quality_assessor=self.assessor,
            config=config,
        )

        # Counter for periodic saving
        self._processed_count = 0
        self._save_interval = phase2_config.get("save_interval", 50)
        self._update_count = 0
        self._memory_save_interval = 10  # Save memory every 10 updates

        # Successful trajectory buffer for skill generation
        self._successful_trajectories = []
        self._skill_gen_interval = phase2_config.get("skill_gen_interval", 20)  # Generate skill every N successes
        self._quality_patterns = {"common_problems": [], "effective_tools": [], "quality_improvements": {}}

    def run(self, image_dir: str, annotation_file: str = None, output_dir: str = None):
        """
        Main workflow:
        For each new image:
        1. Agent process (using Skills and Experiences from memory bank)
        2. FeedbackLoop evaluation
        3. If enable_continual_learning: generate experience updates -> update memory bank
        4. Save results
        """
        output_dir = output_dir or "results/"
        os.makedirs(output_dir, exist_ok=True)

        # Load images
        image_paths = self._load_images(image_dir)

        # Limit to max_images if specified
        max_images = self.config.get("phase2", {}).get("max_images")
        if max_images:
            image_paths = image_paths[:max_images]

        # Load annotations
        annotations = {}
        if annotation_file:
            annotations = self._load_annotations(annotation_file)

        logger.info(
            f"Phase2 Pipeline: {len(image_paths)} images, "
            f"continual_learning={self.enable_continual_learning}, "
            f"use_gt_feedback={self.use_gt_feedback}, "
            f"enable_gt_analysis={self.enable_gt_analysis}"
        )

        all_results = []

        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            img_annotations = annotations.get(img_name, None)

            try:
                image = load_image(img_path)
                result = self.process_single(image, img_annotations, image_path=img_path)
                result["image_name"] = img_name
                result["image_path"] = img_path
                all_results.append(result)

                self._processed_count += 1

                # Periodic save
                if (self.enable_continual_learning
                        and self._processed_count % self._save_interval == 0):
                    self._save_memory()
                    # Update agent skill library
                    self.agent.skill_library_content = self.skill_builder.get_all_skills()

                logger.info(
                    f"[{self._processed_count}/{len(image_paths)}] {img_name}: "
                    f"verdict={result.get('verdict', 'N/A')}, "
                    f"detections={result.get('num_detections', 0)}"
                )

            except Exception as e:
                logger.error(f"Failed to process {img_name}: {e}")
                all_results.append({
                    "image_name": img_name,
                    "image_path": img_path,
                    "error": str(e),
                })
                continue

        # Save all results
        self._save_results(all_results, output_dir)

        # Final memory save and skill generation
        if self.enable_continual_learning:
            # Generate skill from remaining trajectories if any
            if self._successful_trajectories:
                try:
                    self.skill_builder.generate_skill(
                        successful_trajectories=self._successful_trajectories,
                        quality_patterns=self._quality_patterns,
                    )
                    logger.info(f"Final skill generated from {len(self._successful_trajectories)} remaining trajectories")
                    self._successful_trajectories = []
                    self.agent.skill_library_content = self.skill_builder.get_all_skills()
                except Exception as e:
                    logger.warning(f"Final skill generation failed: {e}")
            self._save_memory()

        logger.info(
            f"Phase2 complete: processed {len(image_paths)} images, "
            f"results saved to {output_dir}"
        )

        return all_results

    def process_single(
        self, image: np.ndarray, annotations: List[Dict] = None, image_path: str = None
    ) -> Dict:
        """Process single image, return result (including AgentResult and FeedbackResult).

        Args:
            image: Input image in BGR format.
            annotations: Optional annotation list.
            image_path: Optional image path (for fallback generation tool).

        Returns:
            Dictionary containing detection results, feedback, quality report, etc.
        """
        # 1. Agent processes image (experience-guided detection).
        # Do NOT pass test annotations to the agent by default. This avoids any
        # test-time label leakage in planning, tool selection, or prompting.
        agent_ground_truth = annotations if self.use_gt_feedback else None
        agent_result = self.agent.process(image, ground_truth=agent_ground_truth)

        # 2. Run direct detection on original image (baseline)
        direct_detections = self._detect_on_original(
            agent_result.original_image, agent_result.detector_used
        )

        # 3. Compare detections for analysis (but always use experience-guided for M3/M4)
        # NOTE: M3/M4 always use experience-guided detection, comparison is only for analysis
        detection_selection = "experience"  # M3/M4 always use experience-guided
        mAP_delta = 0.0
        why_experience_worse = None
        proxy = None

        if self.enable_gt_analysis and annotations and len(annotations) > 0:
            # Offline analysis only. The comparison result is recorded but never
            # used to choose between direct and experience-guided detections.
            comparison = compare_detections(
                direct_detections, agent_result.detections, annotations
            )
            mAP_delta = comparison["improvement"]["mAP_delta"]
            # Always use experience-guided, record comparison for analysis
            if mAP_delta < 0:
                # Experience was worse than direct, record reason
                why_experience_worse = self._generate_failure_reason(
                    agent_result, direct_detections, comparison
                )
                logger.info(f"[M3/M4] Experience vs Direct: baseline_mAP={comparison['baseline']['mAP']:.4f}, enhanced_mAP={comparison['enhanced']['mAP']:.4f}, delta={mAP_delta:.4f}")
            else:
                logger.info(f"[M3/M4] Using experience-guided: baseline_mAP={comparison['baseline']['mAP']:.4f}, enhanced_mAP={comparison['enhanced']['mAP']:.4f}, delta={mAP_delta:.4f}")
        else:
            # No labels: record proxy comparison for analysis (but still use experience-guided)
            proxy = self._compare_by_confidence(direct_detections, agent_result.detections)
            if proxy["n_exp"] == 0 and proxy["n_direct"] > 0:
                why_experience_worse = "no detections after preprocessing"
                logger.info(f"[M3/M4] Using experience-guided (n_exp=0, n_direct={proxy['n_direct']})")
            elif proxy["conf_delta"] > 0.05:
                why_experience_worse = f"direct confidence higher (delta={proxy['conf_delta']:.3f})"
                logger.info(f"[M3/M4] Using experience-guided (c_direct={proxy['c_direct']:.3f}, c_exp={proxy['c_exp']:.3f})")
            elif proxy["count_ratio"] < 0.5 or proxy["count_ratio"] > 2.0:
                why_experience_worse = f"detection count ratio abnormal (ratio={proxy['count_ratio']:.2f})"
                logger.info(f"[M3/M4] Using experience-guided (n_exp={proxy['n_exp']}, n_direct={proxy['n_direct']})")
            else:
                logger.info(f"[M3/M4] Using experience-guided (c_direct={proxy['c_direct']:.3f}, c_exp={proxy['c_exp']:.3f})")

        # IMPORTANT: M3/M4 ALWAYS use experience-guided detection (agent_result.detections)
        # The comparison above is only for analysis/recording, NOT for selection

        # 4. Feedback evaluation (on selected detections).
        # Only use GT annotations for feedback when use_gt_feedback=True.
        # During normal inference/testing, keep this False and use proxy feedback only.
        feedback_annotations = annotations if self.use_gt_feedback else None
        feedback_result = self.feedback.evaluate(agent_result, feedback_annotations)

        result = {
            "num_detections": len(agent_result.detections),
            "detections": [
                {
                    "bbox": d.get("bbox", []),
                    "confidence": d.get("confidence", 0.0),
                    "class_id": d.get("class_id", 0),
                }
                for d in agent_result.detections
            ],
            "tools_used": agent_result.tools_used,
            "detector_used": agent_result.detector_used,
            "quality_report": agent_result.quality_report,
            "total_time": agent_result.total_time,
            "verdict": feedback_result.get("verdict", "unknown"),
            "feedback": {
                k: v for k, v in feedback_result.items()
                if k != "raw_message"
            },
            # New fields for comparison result
            "detection_selection": detection_selection,
            "direct_vs_experience_mAP_delta": mAP_delta,
            "why_experience_worse": why_experience_worse,
            "direct_detections_count": len(direct_detections),
            "experience_detections_count": len(agent_result.detections),
            # Proxy metrics (for no-label scenario)
            "proxy_confidence_delta": proxy.get("conf_delta", 0.0) if proxy else 0.0,
            "proxy_count_ratio": proxy.get("count_ratio", 1.0) if proxy else 1.0,
        }

        # 5. Continual learning: generate experience updates and skills
        if self.enable_continual_learning:
            try:
                updates = self.feedback.generate_experience_updates(
                    agent_result, feedback_result
                )
                if updates:
                    self.experience_manager.batch_merge(updates)
                    result["experience_updates"] = len(updates)
                    logger.debug(f"Applied {len(updates)} experience updates")

                # Collect only genuinely improved trajectories for skill generation.
                # verdict from feedback.py: "improved", "degraded", "neutral".
                verdict = feedback_result.get("verdict", "unknown")
                if verdict == "improved" and agent_result.trajectory:
                    trajectory_entry = {
                        "trajectory": agent_result.trajectory,
                        "quality_before": agent_result.quality_report,
                        "quality_after": feedback_result.get("enhanced_metrics", {}),
                        "summary": f"Detection {verdict} with preprocessing",
                    }
                    self._successful_trajectories.append(trajectory_entry)

                    # Extract quality patterns. Support both nested reports and
                    # flat scalar quality scores.
                    quality_before = agent_result.quality_report or {}
                    for key, value in quality_before.items():
                        if isinstance(value, dict):
                            for subkey, val in value.items():
                                if isinstance(val, (int, float)) and val < 0.4:
                                    if subkey not in self._quality_patterns["common_problems"]:
                                        self._quality_patterns["common_problems"].append(subkey)
                        elif isinstance(value, (int, float)) and value < 0.4:
                            if key not in self._quality_patterns["common_problems"]:
                                self._quality_patterns["common_problems"].append(key)

                    # Extract effective tools from trajectory
                    for step in agent_result.trajectory:
                        tool = step.get("tool", "")
                        if tool and tool not in self._quality_patterns["effective_tools"]:
                            self._quality_patterns["effective_tools"].append(tool)

                    # Generate skill when buffer is full
                    if len(self._successful_trajectories) >= self._skill_gen_interval:
                        try:
                            self.skill_builder.generate_skill(
                                successful_trajectories=self._successful_trajectories,
                                quality_patterns=self._quality_patterns,
                            )
                            logger.info(f"Skill generated from {len(self._successful_trajectories)} successful trajectories")
                            self._successful_trajectories = []  # Reset buffer
                            self._quality_patterns = {"common_problems": [], "effective_tools": [], "quality_improvements": {}}
                            # Update agent's skill library
                            self.agent.skill_library_content = self.skill_builder.get_all_skills()
                        except Exception as e:
                            logger.warning(f"Skill generation failed: {e}")

                    # Save memory periodically
                    self._update_count += 1
                    if self._update_count % self._memory_save_interval == 0:
                        self._save_memory()
                        self._update_count = 0  # Reset after save
            except Exception as e:
                logger.warning(f"Continual learning update failed: {e}")

        return result

    def _detect_on_original(
        self, original_image: np.ndarray, detector_name: str
    ) -> List[Dict]:
        """Direct detection on original image (no preprocessing), as baseline comparison.

        Args:
            original_image: Original image.
            detector_name: Detector tool name.

        Returns:
            Detection results list.
        """
        detectors_config = self.config.get("detectors", {})
        # Build detector defaults with num_classes if specified in config
        def build_detector_defaults(cfg):
            if isinstance(cfg, dict):
                result = cfg.copy()
            elif isinstance(cfg, str):
                result = {"model_path": cfg}
            else:
                result = {}
            return result

        detector_defaults = {
            "yolov8_detect": build_detector_defaults(detectors_config.get("yolov8", {})),
            "rtdetr_detect": build_detector_defaults(detectors_config.get("rtdetr", {})),
            "fasterrcnn_detect": build_detector_defaults(detectors_config.get("fasterrcnn", {})),
        }
        try:
            tool = get_tool(detector_name)
            params = detector_defaults.get(detector_name, {})
            result = tool.call(original_image, **params)
            if result.success:
                return result.detections
            else:
                logger.warning(f"Direct detection failed: {result.error}")
                return []
        except Exception as e:
            logger.error(f"Direct detection error: {e}")
            return []

    def _compare_by_confidence(
        self, direct_detections: List[Dict], exp_detections: List[Dict]
    ) -> Dict:
        """Proxy metric comparison without labels (based on confidence and detection count).

        Args:
            direct_detections: Direct detection results list.
            exp_detections: Experience-guided detection results list.

        Returns:
            Dictionary containing confidence and count metrics.
        """
        import numpy as np

        # Calculate mean confidence
        c_direct = np.mean([d["confidence"] for d in direct_detections]) if direct_detections else 0.0
        c_exp = np.mean([d["confidence"] for d in exp_detections]) if exp_detections else 0.0

        # High confidence ratio (confidence > 0.5)
        hc_direct = sum(1 for d in direct_detections if d["confidence"] > 0.5) / max(len(direct_detections), 1)
        hc_exp = sum(1 for d in exp_detections if d["confidence"] > 0.5) / max(len(exp_detections), 1)

        # Detection count
        n_direct = len(direct_detections)
        n_exp = len(exp_detections)

        return {
            "c_direct": c_direct,
            "c_exp": c_exp,
            "hc_direct": hc_direct,
            "hc_exp": hc_exp,
            "n_direct": n_direct,
            "n_exp": n_exp,
            "conf_delta": c_direct - c_exp,  # positive means direct confidence is higher
            "count_ratio": n_exp / max(n_direct, 1),
        }

    @staticmethod
    def _get_quality_scalar(quality: Dict, *keys, default: float = 0.0) -> float:
        """Safely read a scalar quality value from flat or nested quality reports."""
        if not isinstance(quality, dict):
            return default

        for key in keys:
            if key not in quality:
                continue
            value = quality.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, dict):
                for nested_key in ("score", "value", "mean", key):
                    nested_value = value.get(nested_key)
                    if isinstance(nested_value, (int, float)):
                        return float(nested_value)
        return default

    def _generate_failure_reason(
        self,
        agent_result: AgentResult,
        direct_detections: List[Dict],
        comparison: Dict
    ) -> str:
        """Generate failure reason analysis for experience-guided detection (for M4 continual learning log).

        Args:
            agent_result: Agent processing result (including post-preprocessing detections)
            direct_detections: Direct detection results
            comparison: Comparison result from compare_detections

        Returns:
            Failure reason description string.
        """
        try:
            # Build reason analysis
            tools_used = agent_result.tools_used
            quality = agent_result.quality_report

            direct_count = len(direct_detections)
            exp_count = len(agent_result.detections)
            exp_mAP = comparison["enhanced"]["mAP"]
            direct_mAP = comparison["baseline"]["mAP"]

            reason_parts = []

            # Analyze detection count changes
            if exp_count < direct_count:
                reason_parts.append(
                    f"Preprocessing caused missed detections: direct={direct_count}, experience={exp_count}"
                )
            elif exp_count > direct_count:
                reason_parts.append(
                    f"Preprocessing caused false positives: direct={direct_count}, experience={exp_count}"
                )

            # Analyze mAP drop
            if exp_mAP < direct_mAP:
                reason_parts.append(
                    f"mAP dropped: direct={direct_mAP:.3f} → experience={exp_mAP:.3f}"
                )

            # Analyze preprocessing tools used
            if tools_used:
                reason_parts.append(
                    f"Preprocessing tools used: {', '.join(tools_used)}"
                )

            # Analyze image quality. Compatible with both paper-style names
            # (blur/contrast/noise) and code-style names (blur_score, etc.).
            if quality:
                blur = self._get_quality_scalar(quality, "blur", "blur_score")
                contrast = self._get_quality_scalar(quality, "contrast", "contrast_score")
                noise = self._get_quality_scalar(quality, "noise", "noise_score")
                reason_parts.append(
                    f"Image quality - blur={blur:.2f}, contrast={contrast:.2f}, noise={noise:.2f}"
                )

            return "; ".join(reason_parts) if reason_parts else "unknown reason"

        except Exception as e:
            logger.warning(f"Failed to generate failure reason: {e}")
            return "failure reason analysis generation failed"

    def _load_images(self, image_dir: str) -> List[str]:
        """Load image path list"""
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        image_paths = []

        for fname in sorted(os.listdir(image_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in supported_exts:
                image_paths.append(os.path.join(image_dir, fname))

        logger.info(f"Found {len(image_paths)} images in {image_dir}")
        return image_paths

    def _load_annotations(self, annotation_file: str) -> Dict:
        """Load COCO format annotations, returns {image_filename: [annotations]}"""
        if not os.path.exists(annotation_file):
            logger.warning(f"Annotation file not found: {annotation_file}")
            return {}

        with open(annotation_file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        id_to_filename = {}
        for img_info in coco_data.get("images", []):
            id_to_filename[img_info["id"]] = img_info["file_name"]

        annotations = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            filename = id_to_filename.get(img_id)
            if filename is None:
                continue

            if filename not in annotations:
                annotations[filename] = []

            bbox = ann["bbox"]
            x1, y1, w, h = bbox
            converted_ann = {
                "bbox": [x1, y1, x1 + w, y1 + h],
                "class_id": ann.get("category_id", 0),
            }
            annotations[filename].append(converted_ann)

        logger.info(f"Loaded annotations for {len(annotations)} images")
        return annotations

    def _save_results(self, results: List[Dict], output_dir: str):
        """Save all results to JSON"""
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "results.json")

        # Filter non-serializable values
        serializable = []
        for r in results:
            clean = {}
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    clean[k] = v.tolist()
                elif isinstance(v, (np.float32, np.float64)):
                    clean[k] = float(v)
                elif isinstance(v, (np.int32, np.int64)):
                    clean[k] = int(v)
                else:
                    clean[k] = v
            serializable.append(clean)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(results)} results to {results_path}")

    def _save_memory(self):
        """Save memory bank"""
        try:
            self.experience_manager.save()
            self.skill_builder.save()
            logger.info("Memory saved (continual learning)")
        except Exception as e:
            logger.warning(f"Failed to save memory: {e}")

    def flush_memory(self):
        """Force save all pending memory (for final save after processing ends)"""
        if hasattr(self, '_update_count') and self._update_count > 0:
            self._save_memory()
            self._update_count = 0
            logger.info("Flushed remaining memory updates")

    @staticmethod
    def _build_quality_config(quality_config: dict) -> dict:
        """Convert flat quality config to ImageQualityAssessor format"""
        return {
            "weights": {
                "blur": quality_config.get("blur_weight", 0.25),
                "brightness": quality_config.get("brightness_weight", 0.20),
                "contrast": quality_config.get("contrast_weight", 0.20),
                "noise": quality_config.get("noise_weight", 0.20),
                "color_bias": quality_config.get("color_bias_weight", 0.15),
                # Kept for compatibility with the paper's colony-aware query.
                # They are used if the assessor implements density/overlap estimation.
                "density": quality_config.get("density_weight", 0.10),
                "overlap": quality_config.get("overlap_weight", 0.10),
            },
            "thresholds": quality_config.get("thresholds", {}),
        }
