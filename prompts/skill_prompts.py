# -*- coding: utf-8 -*-
"""Phase I skill construction and Phase II skill adaptation prompt templates.

Skills are standard operating procedures (SOP) extracted from successful trajectories, stored in Markdown format.
Phase I: Generate and merge Skills from training trajectories
Phase II: Adapt general Skills to the current detection task
"""

# Prompt 1: Generate Skill workflow template from successful trajectories
SKILL_GENERATION_PROMPT = """You are a colony detection workflow architect. Analyze the following successful processing trajectories and extract a reusable Standard Operating Procedure (SOP) for microbial colony detection.

### Successful Trajectories
These trajectories achieved high detection accuracy (close to ground truth). Each contains the tool sequence, parameters, quality changes, and detection results:
{successful_trajectories}

### Quality Patterns
Common quality characteristics observed across these successful cases:
{quality_patterns}

### Extraction Guidelines

1. **Identify the Core Workflow**: What is the common tool sequence across successful trajectories? Abstract away image-specific details and extract the general pattern.

2. **Map Quality Conditions to Actions**: For each preprocessing step, specify:
   - Under what quality conditions should this step be applied?
   - What parameter ranges worked well?
   - What quality improvement is expected?

3. **Capture Decision Logic**: The SOP should enable an agent to make decisions, not just follow a fixed sequence. Include branching logic based on quality scores.

4. **Keep It Practical**: ~600 words maximum. Focus on actionable steps, not abstract principles.

### Output Format
Output the Skill as a Markdown document with the following structure:

```markdown
---
name: [SkillName, e.g., LowLightColonyDetection]
description: |
  [1-2 sentences: what this skill does and when to use it]
version: 1.0.0
quality_profile: |
  [Which quality dimensions this skill primarily addresses]
---

# [Skill Title]

## When to Use
- **Quality triggers**: [Which quality scores indicate this skill is needed, e.g., brightness_score < 0.4]
- **Image characteristics**: [Visual cues, e.g., dark background, dense colonies]

## Strategy Overview
[1-2 sentences on the core approach]

## Workflow
1. **[Phase Name]** (Condition: [quality condition])
   - Tool: `[tool_name]`
   - Parameters: `[key parameters and recommended ranges]`
   - Expected effect: [Which quality score improves and by how much]

2. **[Phase Name]** (Condition: [quality condition])
   - ...

3. **Detection**
   - Tool: `[detection_tool]`
   - Post-processing: [Any post-detection filtering, e.g., confidence threshold]

## Parameter Guidelines
| Quality Condition | Tool | Key Parameter | Recommended Range |
|---|---|---|---|
| blur_score < 0.4 | sharpen | kernel_size | 3-5 |
| ... | ... | ... | ... |

## Watch Out For
- [Common pitfall from the trajectories, e.g., over-sharpening noisy images]
- [Another pitfall]
```

Output ONLY the Skill document starting with `---`. No preamble or explanation."""


# Prompt 2: Merge multiple similar Skills
SKILL_MERGE_PROMPT = """You are a knowledge architect for a colony detection system. Merge two Skill documents into a single, unified SOP that captures the best of both.

### Existing Skill
{skill_1}

### New Skill
{skill_2}

### Integration Strategy
For each section of the new skill, decide:
- **Better?** -> Replace the existing version with the new one
- **Redundant or too specific?** -> Discard it
- **Complementary?** -> Merge into a more general form
- **Genuinely different workflow?** -> Add as a variant (but consolidate if possible)

### Quality Guidelines
- Preserve concrete parameter recommendations and quality-to-action mappings
- Delete overly specific details that only apply to one image type
- Consolidate overlapping workflow steps
- Keep the document under ~1000 words
- Maximum 3 variant workflows — consolidate if more
- Ensure all quality dimension references use the standard names: blur_score, brightness_score, contrast_score, noise_score, color_bias_score

Output ONLY the merged Skill document starting with `---`. No preamble."""


# Prompt 3: Phase II skill adaptation - Adapt general Skill to current detection task
SKILL_ADAPTATION_PROMPT = """You are an expert colony detection assistant. Adapt the general-purpose Skill (SOP) below to the specific image and task at hand, using the quality report and retrieved experiences as context.

### Base Skill
The following is a general SOP for colony detection preprocessing and detection:
{base_skill}

### Current Image Quality Report
6-dimensional quality assessment of the current image:
{current_quality_report}

### Retrieved Experiences
Relevant experiences from the experience library, selected based on quality similarity:
{retrieved_experiences}

### Image Description
{image_description}

### Adaptation Instructions

1. **Select Relevant Steps**: Based on the current quality report, determine which workflow steps from the base skill are actually needed. For example:
   - If blur_score = 0.9, skip any sharpening steps
   - If noise_score = 0.3, include the denoising step with appropriate parameters
   - If all scores are above 0.6, consider skipping preprocessing entirely

2. **Integrate Experiences**: Weave insights from retrieved experiences into the workflow:
   - If an experience suggests a specific parameter value for a similar quality condition, use it
   - If an experience warns against a certain tool combination, respect that warning
   - Resolve any conflicts between the base skill and experiences (prefer experiences, as they are learned from actual outcomes)

3. **Concretize Parameters**: Replace parameter ranges with specific values based on the quality scores. For example:
   - "clip_limit: 2.0-4.0" -> "clip_limit: 3.0" (if brightness_score = 0.25)

4. **Output Format**: Produce a focused, task-specific guide (~400 words max). Use markdown starting with `#`. Do NOT include frontmatter metadata (no `---` blocks). This should be a ready-to-execute plan, not a general reference.

```markdown
# Adapted Workflow for Current Task

## Quality Assessment Summary
[Brief summary of which dimensions need attention]

## Preprocessing Plan
1. **[Step Name]**
   - Tool: `tool_name`
   - Parameters: {{specific values}}
   - Rationale: [Why this step, referencing quality scores and/or experiences]

2. ...

## Detection
- Tool: `detector_name`
- Configuration: {{specific settings}}

## Cautions
- [Task-specific warnings based on experiences]
```

Output ONLY the adapted skill content in markdown format."""
