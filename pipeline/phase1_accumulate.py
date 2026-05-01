"""Phase I Offline Accumulation Pipeline.

Runs multiple rollouts on historical data to distill Skills and Experiences, establishing a foundational memory bank.
"""

import os
import json
import yaml
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
from utils.image_utils import load_image


class Phase1Pipeline:
    """
    Offline accumulation Pipeline:
    Runs multiple rollouts on historical data to distill Skills and Experiences, establishing a foundational memory bank.
    """

    def __init__(self, config: dict):
        """
        Initialize all components from config:
        - LLMClient
        - ImageQualityAssessor
        - ExperienceManager, ExperienceRetriever
        - SkillBuilder, SkillAdapter
        - TrajectorySummarizer, ExperienceCritic
        - ColonyDetectionAgent
        - FeedbackLoop
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
        self.experience_bank_dir = memory_config.get("experience_bank_dir", "memory/experience_bank")
        self.skill_library_dir = memory_config.get("skill_library_dir", "memory/skill_library")
        self.trajectories_dir = memory_config.get("trajectories_dir", "memory/trajectories")

        # Experience Manager & Retriever
        self.experience_manager = ExperienceManager(
            llm_client=self.llm,
            save_dir=self.experience_bank_dir,
            similarity_threshold=config.get("phase2", {}).get("similarity_threshold", 0.85),
        )
        self.experience_retriever = ExperienceRetriever(
            llm_client=self.llm,
            experience_manager=self.experience_manager,
            top_k=config.get("phase2", {}).get("top_k", 3),
            min_similarity=config.get("phase2", {}).get("similarity_threshold", 0.6),
        )

        # Skill Builder & Adapter
        self.skill_builder = SkillBuilder(llm_client=self.llm, save_dir=self.skill_library_dir)
        self.skill_adapter = SkillAdapter(llm_client=self.llm)

        # Trajectory Summarizer & Experience Critic
        self.summarizer = TrajectorySummarizer(llm_client=self.llm)
        self.critic = ExperienceCritic(llm_client=self.llm)

        # Agent - requires detectors config
        agent_config = {
            "default_detector": "yolov8_detect",
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

        # Phase1 params
        phase1_config = config.get("phase1", {})
        self.num_rollouts = phase1_config.get("num_rollouts", 3)
        self.batch_size = phase1_config.get("batch_size", 10)

    def run(self, image_dir: str, annotation_file: str = None, output_dir: str = None):
        """
        Main flow:
        1. Load image list and annotations (if any, COCO format JSON)
        2. Process in batches
        3. Run num_rollouts for each image
        4. Each round: Agent process -> record trajectory -> FeedbackLoop evaluation
        5. After each image: trajectory summary -> cross-rollout critique -> experience update
        6. After each batch: Skill generation/merging -> save memory bank
        """
        output_dir = output_dir or "memory/"
        os.makedirs(output_dir, exist_ok=True)
        traj_output_dir = os.path.join(output_dir, "trajectories")
        os.makedirs(traj_output_dir, exist_ok=True)

        # 1. Load images and annotations
        image_paths = self._load_images(image_dir)
        annotations = {}
        if annotation_file:
            annotations = self._load_annotations(annotation_file)

        logger.info(
            f"Phase1 Pipeline: {len(image_paths)} images, "
            f"{len(annotations)} annotated, "
            f"num_rollouts={self.num_rollouts}, batch_size={self.batch_size}"
        )

        # 2. Process in batches
        all_successful_trajectories = []
        all_quality_patterns = {
            "common_problems": [],
            "effective_tools": [],
            "quality_improvements": {},
        }

        for batch_start in range(0, len(image_paths), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]
            batch_idx = batch_start // self.batch_size + 1
            logger.info(f"=== Batch {batch_idx}: images {batch_start+1}-{batch_end} ===")

            batch_successful = []

            for img_path in batch_paths:
                img_name = os.path.basename(img_path)
                img_annotations = annotations.get(img_name, None)

                try:
                    result = self._process_single_image(
                        img_path, img_annotations
                    )

                    # Save trajectories
                    for rollout_idx, traj in enumerate(result.get("trajectories", [])):
                        self._save_trajectory(traj, img_name, rollout_idx, traj_output_dir)

                    # Collect successful trajectories for skill building
                    for traj_info in result.get("successful_trajectories", []):
                        batch_successful.append(traj_info)

                    # Collect quality patterns
                    quality = result.get("quality_report", {})
                    for dim in ["blur_score", "brightness_score", "contrast_score",
                                "noise_score", "color_bias_score"]:
                        if quality.get(dim, 1.0) < 0.5:
                            problem = dim.replace("_score", "")
                            if problem not in all_quality_patterns["common_problems"]:
                                all_quality_patterns["common_problems"].append(problem)

                    for tool in result.get("effective_tools", []):
                        if tool not in all_quality_patterns["effective_tools"]:
                            all_quality_patterns["effective_tools"].append(tool)

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(f"Failed to process {img_name}: {e}")
                    continue

            # 6. After each batch: generate/merge skills and save memory
            all_successful_trajectories.extend(batch_successful)

            if batch_successful:
                try:
                    self.skill_builder.generate_skill(
                        successful_trajectories=batch_successful,
                        quality_patterns=all_quality_patterns,
                    )
                    logger.info(f"Skill generated after batch {batch_idx}")
                except Exception as e:
                    logger.warning(f"Skill generation failed: {e}")

            # Update agent's skill library content
            self.agent.skill_library_content = self.skill_builder.get_all_skills()

            # Save memory
            self._save_memory(output_dir)
            logger.info(f"Memory saved after batch {batch_idx}")

        logger.info(
            f"Phase1 complete: processed {len(image_paths)} images, "
            f"{len(self.experience_manager.experiences)} experiences, "
            f"{len(self.skill_builder.skills)} skills"
        )

    def _load_images(self, image_dir: str) -> List[str]:
        """Load image path list."""
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        image_paths = []

        for fname in sorted(os.listdir(image_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in supported_exts:
                image_paths.append(os.path.join(image_dir, fname))

        logger.info(f"Found {len(image_paths)} images in {image_dir}")
        return image_paths

    def _load_annotations(self, annotation_file: str) -> Dict:
        """Load annotations, supports COCO JSON or YOLO label directory. Returns {image_stem: [annotations]}."""
        if not os.path.exists(annotation_file):
            logger.warning(f"Annotation file/dir not found: {annotation_file}")
            return {}

        # If directory, load as YOLO txt format
        if os.path.isdir(annotation_file):
            return self._load_yolo_annotations(annotation_file)

        with open(annotation_file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        # Build image_id -> filename mapping
        id_to_filename = {}
        for img_info in coco_data.get("images", []):
            id_to_filename[img_info["id"]] = img_info["file_name"]

        # Group annotations by image filename
        annotations = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            filename = id_to_filename.get(img_id)
            if filename is None:
                continue

            if filename not in annotations:
                annotations[filename] = []

            # Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]
            bbox = ann["bbox"]
            x1, y1, w, h = bbox
            converted_ann = {
                "bbox": [x1, y1, x1 + w, y1 + h],
                "class_id": ann.get("category_id", 0),
            }
            annotations[filename].append(converted_ann)

        logger.info(f"Loaded annotations for {len(annotations)} images")
        return annotations

    def _load_yolo_annotations(self, label_dir: str) -> Dict:
        """Load YOLO txt format label directory. Returns {label_stem: [annotations]}.

        Note: YOLO format uses normalized coordinates, here we store normalized values.
        In actual use, need to multiply by image width/height to convert to pixel coordinates.
        """
        import glob as glob_mod
        annotations = {}
        for label_path in sorted(glob_mod.glob(os.path.join(label_dir, "*.txt"))):
            stem = os.path.splitext(os.path.basename(label_path))[0]
            boxes = []
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        boxes.append({
                            "class_id": cls_id,
                            "xc": xc, "yc": yc, "w": bw, "h": bh,
                            "format": "yolo_normalized",
                        })
            if boxes:
                annotations[stem] = boxes
        logger.info(f"Loaded YOLO annotations for {len(annotations)} images from {label_dir}")
        return annotations

    def _process_single_image(
        self, image_path: str, annotations: List[Dict] = None
    ) -> Dict:
        """
        Process multiple rollouts for a single image:
        1. Load image
        2. Multiple rollouts (different temperatures to let LLM explore different strategies)
        3. Record complete trajectory for each round
        4. Trajectory summary
        5. Cross-rollout critique -> experience update
        6. Return processing result
        """
        img_name = os.path.basename(image_path)
        logger.info(f"Processing: {img_name} ({self.num_rollouts} rollouts)")

        image = load_image(image_path)

        rollout_results: List[AgentResult] = []
        rollout_feedbacks: List[Dict] = []
        rollout_summaries: List[str] = []
        trajectories: List[List[Dict]] = []
        successful_trajectories = []
        effective_tools = []

        # Temperature schedule for exploration
        temperatures = np.linspace(0.5, 1.0, self.num_rollouts).tolist()

        first_quality_report = None

        for rollout_idx in range(self.num_rollouts):
            temp = temperatures[rollout_idx]
            logger.info(f"  Rollout {rollout_idx + 1}/{self.num_rollouts} (temp={temp:.2f})")

            try:
                # Temporarily adjust LLM temperature for exploration
                original_temp = self.llm.temperature
                self.llm.temperature = temp

                # Agent processes the image
                agent_result = self.agent.process(image, ground_truth=annotations)

                # Restore temperature
                self.llm.temperature = original_temp

                rollout_results.append(agent_result)
                trajectories.append(agent_result.trajectory)

                if first_quality_report is None:
                    first_quality_report = agent_result.quality_report

                # Feedback evaluation
                feedback_result = self.feedback.evaluate(agent_result, annotations)
                rollout_feedbacks.append(feedback_result)

                # Trajectory summary
                detection_info = {
                    "num_detections": len(agent_result.detections),
                    "detections": [
                        {
                            "bbox": d.get("bbox", []),
                            "confidence": d.get("confidence", 0.0),
                            "class_id": d.get("class_id", 0),
                        }
                        for d in agent_result.detections[:20]  # Limit for summary
                    ],
                }
                summary = self.summarizer.summarize(
                    trajectory=agent_result.trajectory,
                    quality_report=agent_result.quality_report,
                    detection_result=detection_info,
                    ground_truth={"annotations": annotations} if annotations else None,
                )
                rollout_summaries.append(summary)

                # Track trajectories for skill building
                # Both "improved" and "degraded" verdicts generate skills - neutral/failed rollouts discarded
                verdict = feedback_result.get("verdict", "neutral")
                if verdict in ("improved", "degraded"):
                    successful_trajectories.append({
                        "trajectory": agent_result.trajectory,
                        "quality_before": agent_result.quality_report,
                        "quality_after": agent_result.quality_report,
                        "summary": summary,
                        "verdict": verdict,
                    })
                    for tool in agent_result.tools_used:
                        if tool not in effective_tools:
                            effective_tools.append(tool)

                logger.info(
                    f"  Rollout {rollout_idx + 1} verdict: {feedback_result.get('verdict')}, "
                    f"details: {feedback_result.get('details', 'N/A')}"
                )

            except Exception as e:
                logger.error(f"  Rollout {rollout_idx + 1} failed: {e}")
                continue

        # 5. Cross-rollout critique -> experience updates
        if rollout_summaries:
            gt_dict = {"annotations": annotations} if annotations else {}
            existing_exps = self.experience_manager.get_all()

            try:
                experience_ops = self.critic.critique(
                    rollout_summaries=rollout_summaries,
                    ground_truth=gt_dict,
                    existing_experiences=existing_exps,
                )
                if experience_ops:
                    self.experience_manager.batch_merge(experience_ops)
                    logger.info(f"  Applied {len(experience_ops)} experience updates from critique")
            except Exception as e:
                logger.warning(f"  Critique failed: {e}")

            # Also apply feedback-based experience updates
            for agent_result, feedback_result in zip(rollout_results, rollout_feedbacks):
                try:
                    updates = self.feedback.generate_experience_updates(
                        agent_result, feedback_result
                    )
                    if updates:
                        self.experience_manager.batch_merge(updates)
                except Exception as e:
                    logger.warning(f"  Feedback experience update failed: {e}")

        # Check if all rollouts failed - if so, try online fallback (search + generate)
        all_verdicts = [fb.get("verdict", "neutral") for fb in rollout_feedbacks]
        successful_count = sum(1 for v in all_verdicts if v == "improved")

        # Collect all tools that were attempted
        attempted_tools = []
        for traj in trajectories:
            for step in traj:
                tool_name = step.get("tool")
                if tool_name and tool_name not in attempted_tools:
                    attempted_tools.append(tool_name)

        return {
            "image_name": img_name,
            "quality_report": first_quality_report or {},
            "num_rollouts": len(rollout_results),
            "verdicts": all_verdicts,
            "trajectories": trajectories,
            "successful_trajectories": successful_trajectories,
            "effective_tools": effective_tools,
        }

    def _save_trajectory(
        self, trajectory: List[Dict], image_name: str, rollout_idx: int, output_dir: str
    ):
        """Save trajectory to JSONL file."""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{os.path.splitext(image_name)[0]}_rollout{rollout_idx}.jsonl"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            for step in trajectory:
                # Filter out non-serializable fields
                serializable_step = {}
                for k, v in step.items():
                    if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                        serializable_step[k] = v
                    else:
                        serializable_step[k] = str(v)
                f.write(json.dumps(serializable_step, ensure_ascii=False) + "\n")

    def _save_memory(self, output_dir: str):
        """Save Experience Bank and Skill Library."""
        # Update save dirs to output_dir
        exp_dir = os.path.join(output_dir, "experience_bank")
        skill_dir = os.path.join(output_dir, "skill_library")
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(skill_dir, exist_ok=True)

        # Temporarily change save dirs
        original_exp_dir = self.experience_manager.save_dir
        original_skill_dir = self.skill_builder.save_dir

        self.experience_manager.save_dir = exp_dir
        self.skill_builder.save_dir = skill_dir

        self.experience_manager.save()
        self.skill_builder.save()

        # Restore
        self.experience_manager.save_dir = original_exp_dir
        self.skill_builder.save_dir = original_skill_dir

        logger.info(
            f"Memory saved: {len(self.experience_manager.experiences)} experiences, "
            f"{len(self.skill_builder.skills)} skills"
        )

    @staticmethod
    def _build_quality_config(quality_config: dict) -> dict:
        """Convert flat quality config to format required by ImageQualityAssessor."""
        return {
            "weights": {
                "blur": quality_config.get("blur_weight", 0.25),
                "brightness": quality_config.get("brightness_weight", 0.20),
                "contrast": quality_config.get("contrast_weight", 0.20),
                "noise": quality_config.get("noise_weight", 0.20),
                "color_bias": quality_config.get("color_bias_weight", 0.15),
            },
            "thresholds": quality_config.get("thresholds", {}),
        }
