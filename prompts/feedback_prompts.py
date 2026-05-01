# -*- coding: utf-8 -*-
"""Closed-loop feedback prompt templates.

Analyzes detection effect comparison before and after preprocessing, generates experience update operations, and implements automatic iterative optimization of experience.
"""

# Analyze preprocessing effects and generate experience update operations
FEEDBACK_ANALYSIS_PROMPT = """You are a colony detection feedback analyst. Compare the detection results before and after preprocessing to evaluate whether the applied tool sequence was effective, and generate experience updates for the system's experience library.

### Processing Trajectory
The agent applied the following tools in sequence (tool name, parameters, and result for each step):
{trajectory}

### Baseline Metrics (Detection on Original Image, No Preprocessing)
Detection results when running the detector directly on the unprocessed image:
{baseline_metrics}

### Enhanced Metrics (Detection After Preprocessing)
Detection results after the full preprocessing pipeline was applied:
{enhanced_metrics}

### Image Quality Report (Before Processing)
6-dimensional quality assessment of the original image:
{quality_report}

### Analysis Instructions

1. **Effectiveness Assessment**:
   Compare baseline vs. enhanced metrics across these dimensions:
   - **Count accuracy**: |predicted_count - ground_truth_count|. Did preprocessing reduce the error?
   - **Precision**: Of the detected colonies, how many are true positives? Did preprocessing reduce false positives?
   - **Recall**: Of the ground truth colonies, how many were detected? Did preprocessing help find missed colonies?
   - **Localization**: Average IoU between predicted and ground truth bounding boxes. Did preprocessing improve localization?
   - **Overall**: F1 score or mAP change. Was the net effect positive, neutral, or negative?

2. **Per-Tool Attribution**:
   For each tool in the trajectory, estimate its contribution:
   - Did this specific tool likely help, hurt, or have no effect?
   - Which quality dimension did it target, and did that dimension actually need improvement?
   - Were the parameters appropriate for the observed quality scores?

3. **Root Cause Analysis**:
   - If preprocessing HELPED: What was the key quality issue, and which tool addressed it most effectively?
   - If preprocessing HURT: What went wrong? Over-processing? Wrong tool? Bad parameters? Wrong ordering?
   - If preprocessing had NO EFFECT: Was the original image already good enough, or did preprocessing changes cancel each other out?

4. **Experience Updates**:
   Based on the analysis, generate updates to the experience library. Use the following operations:

   - **add**: A new experience learned from this feedback loop. Must include condition (quality-based), action, and reason.
   - **modify**: Update an existing experience (by ID) if the observed outcome suggests it needs refinement.
   - **reinforce**: Increase confidence in an existing experience (by ID) because this outcome confirms it.
   - **weaken**: Decrease confidence in an existing experience (by ID) because this outcome contradicts it.

Provide your detailed analysis first, then conclude with:
```json
{{
  "effectiveness": {{
    "count_accuracy_change": "+5 colonies closer to GT" | "-3 colonies further from GT" | "no change",
    "precision_change": "+0.12" | "-0.05" | "0.0",
    "recall_change": "+0.08" | "-0.03" | "0.0",
    "overall_verdict": "positive" | "negative" | "neutral"
  }},
  "experience_updates": [
    {{
      "operation": "add",
      "condition": "When noise_score < 0.3 and blur_score > 0.7",
      "action": "Apply bilateral filter (d=9, sigmaColor=75) before detection",
      "reason": "In this case, bilateral filtering reduced false positives by 15% while preserving colony edges, because the image was sharp but noisy"
    }},
    {{
      "operation": "modify",
      "experience_id": "E12",
      "condition": "updated condition",
      "action": "updated action",
      "reason": "Previous recommendation of Gaussian blur was too aggressive; bilateral filter preserves edges better for colony detection"
    }},
    {{
      "operation": "reinforce",
      "experience_id": "E7",
      "reason": "Confirmed: CLAHE with clip_limit=2.5 effectively handles low-contrast colony images (contrast_score < 0.4)"
    }},
    {{
      "operation": "weaken",
      "experience_id": "E15",
      "reason": "Sharpening after denoising did not improve detection in this case despite the skill recommending it; may only help for severely blurred images"
    }}
  ]
}}
```

Output your analysis followed by the JSON block. Ensure the JSON is valid and complete."""
