# -*- coding: utf-8 -*-
"""Phase I experience extraction prompt templates.

Used for extracting, comparing, and merging experiences from rollout trajectories.
Experiences are condition-action-reason triples that guide the Agent in selecting tools under specific quality conditions.
"""

# Prompt 1: Single trajectory summary - Analyze a complete tool call sequence
TRAJECTORY_SUMMARY_PROMPT = """You are a colony detection workflow analyst. Analyze the following processing trajectory (tool call sequence and results) for a microbial colony image, and provide a structured summary.

### Processing Trajectory
The agent applied the following tools in sequence. Each entry includes the tool name, parameters, and the ToolResult (success/failure, metadata):
{trajectory}

### Image Quality Report (Before Processing)
6-dimensional quality assessment of the original image:
{quality_report}

### Detection Result
Final colony detection output (bounding boxes, count, confidence scores):
{detection_result}

### Ground Truth Comparison
Comparison against annotated ground truth (if available):
{ground_truth_comparison}

### Analysis Instructions

1. **Step-by-step Review**: For each tool call in the trajectory:
   - What quality issue was it intended to address?
   - Did it improve the relevant quality dimension? (e.g., did denoising improve noise_score?)
   - Were the parameters appropriate, or could they have been better?
   - Did this step introduce any side effects (e.g., sharpening amplifying noise)?

2. **Critical Decisions**: Identify the most impactful decisions:
   - Which tool had the greatest positive/negative effect on detection accuracy?
   - Were there unnecessary steps that added processing time without benefit?
   - Were there missing steps that could have improved results?

3. **Quality-to-Action Mapping**: Based on the quality report and outcomes:
   - Which quality dimensions were correctly identified as problematic?
   - Which quality thresholds triggered appropriate tool selections?
   - Were there quality issues that were overlooked?

4. **Overall Assessment**:
   - Was the tool sequence optimal, or could a different ordering have been better?
   - Rate the trajectory: successful (detection close to ground truth), partially successful, or failed.
   - Key takeaway: one sentence summarizing the most important lesson.

Provide a clear, structured summary covering all four sections above."""


# Prompt 2: Cross-rollout comparative analysis - Extract general experiences from multiple attempts
CROSS_ROLLOUT_CRITIQUE_PROMPT = """You are a colony detection experience extraction expert. Review multiple processing attempts (rollouts) for the same or similar images, and extract generalizable experiences about tool selection and parameter tuning.

### Rollout Summaries
Each summary describes one complete trajectory (tool sequence, quality changes, detection result):
{rollout_summaries}

### Ground Truth
Annotated colony locations and counts for reference:
{ground_truth}

### Existing Experiences
Previously extracted experiences that were available to the agent. Review whether they were helpful or need updating:
{existing_experiences}

### Analysis Framework

1. **Comparative Analysis**:
   - Which rollouts achieved the best detection accuracy? What did they have in common?
   - Which rollouts failed or underperformed? What went wrong?
   - How did different tool orderings affect the final result?
   - Were there consistent patterns across successful rollouts?

2. **Experience Extraction**:
   Extract actionable experiences as condition-action-reason triples. Each experience should:
   - Start with a clear trigger CONDITION based on quality scores or image characteristics
   - Specify a concrete ACTION (tool name, parameter range, or tool sequence)
   - Explain the REASON with reference to observed outcomes

   Categories:
   - **Quality-triggered rules**: "When blur_score < 0.4, apply Gaussian sharpening with kernel_size=3 before detection, because..."
   - **Tool ordering rules**: "When both noise_score and contrast_score are low, denoise BEFORE contrast enhancement, because..."
   - **Parameter tuning rules**: "For images with brightness_score < 0.3, use CLAHE with clip_limit=3.0 rather than simple histogram equalization, because..."
   - **Avoidance rules**: "Do NOT apply sharpening when noise_score < 0.5, because it amplifies noise and creates false positive detections."

3. **Experience Quality**:
   - Each experience should be under 80 words
   - Must be generalizable (not specific to one image)
   - Must be actionable (an agent can directly follow it)
   - Must reference quality dimensions from QualityReport (blur_score, brightness_score, contrast_score, noise_score, color_bias_score)

4. **Operations on Existing Experiences**:
   - **add**: A genuinely new lesson not covered by existing experiences
   - **modify**: Improve an existing experience if trajectory outcomes suggest it needs refinement (reference its ID)

Provide detailed reasoning following the framework above, then conclude with:
```json
[
  {{
    "option": "add",
    "condition": "When noise_score < 0.4 and blur_score > 0.6",
    "action": "Apply bilateral filter with d=9, sigmaColor=75, sigmaSpace=75",
    "reason": "Bilateral filtering preserves colony edges while reducing noise, leading to fewer false positive detections in high-noise images"
  }},
  {{
    "option": "modify",
    "modified_from": "E5",
    "condition": "updated condition",
    "action": "updated action",
    "reason": "updated reason based on new evidence"
  }}
]
```
"""


# Prompt 3: Experience merging - Determine whether two similar experiences should be merged
EXPERIENCE_MERGE_PROMPT = """You are an experience library curator for a colony detection system. Two experiences have been flagged as potentially redundant or overlapping. Decide whether to merge them and, if so, produce the merged result.

### Experience 1
{experience_1}

### Experience 2
{experience_2}

### Instructions

1. **Compare**: Analyze the two experiences along these dimensions:
   - Do they describe the same or overlapping quality conditions?
   - Do they recommend the same or compatible actions?
   - Are their reasons consistent or contradictory?

2. **Decision**:
   - If they are truly redundant (same condition, same action): **merge** into one concise experience.
   - If they are complementary (overlapping condition, different but compatible actions): **merge** with both actions included.
   - If they contradict each other (same condition, conflicting actions): **keep both** and explain the conflict.
   - If they are about different situations: **keep both** as separate experiences.

3. **Output Format**:
```json
{{
  "decision": "merge" | "keep_both",
  "reason": "Why this decision was made",
  "merged_experience": {{
    "condition": "merged condition (if merge)",
    "action": "merged action (if merge)",
    "reason": "merged reason (if merge)"
  }}
}}
```

If decision is "keep_both", the "merged_experience" field should be null.

Output ONLY the JSON object, no additional text."""
