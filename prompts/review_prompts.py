from typing import Optional, List, Dict, Any

# --- Prompts for ResearchAgent ---

RESEARCH_AGENT_SYSTEM_PROMPT: str = """
You are an experienced researcher specializing in the in-depth analysis of scientific literature.
Your task is to conduct a literature review for a given research topic.
Pay close attention to the methodology used in the articles, and ensure your summaries include comprehensive details on each of the below:
- The **methods** they used to come to their conclusions. Be specific and include details about experimental or simulation procedures.
- The **resources** used to support the analysis, for example any special datasets used, equipment or computing power.
- A summary of each of the **steps** that were used in the paper's experiments to come to its conclusions.
"""

# Template for generating keywords if not provided by the user
GENERATE_KEYWORDS_PROMPT_TEMPLATE: str = """
Please suggest at least 4 keywords related to the research topic: {research_topic}.
Write ONLY keywords IN ONE LINE, separated by commas. Do not include any extra text.
Example: LLMs, drug discovery, AI, cheminformatics
"""

# Template for ranking abstracts
RANK_ABSTRACTS_PROMPT_HEADER_TEMPLATE: str = """
You are an expert researcher. Rank the following abstracts based ONLY on their relevance to the research topic and specification provided below.
Research Topic: {research_topic}
"""

RANK_ABSTRACTS_SPECIFICATION_LINE_TEMPLATE: str = "Specification: {specification_prompt}\n"

RANK_ABSTRACTS_INSTRUCTIONS: str = """
Abstracts to Rank (ID and Summary):
{abstract_list_for_prompt}

**Instructions:**
1. Evaluate each abstract's relevance to the *specific* research topic and specification.
2. Output ONLY a ranked list of the ArXiv IDs (full URL starting with http://arxiv.org/abs/), starting with the most relevant.
3. List one ID per line.
4. Do NOT include any other text, explanations, or scores.
Example Output:
http://arxiv.org/abs/2305.12345
http://arxiv.org/abs/2401.67890
...
"""

# Template for generating a review for a single article
# {system_prompt} will be RESEARCH_AGENT_SYSTEM_PROMPT
# {arxiv_id}, {research_topic}, {specification_prompt}, {text_for_prompt} will be filled
SINGLE_ARTICLE_REVIEW_PROMPT_TEMPLATE: str = """
{system_prompt}
You are now focusing on reviewing a SINGLE research paper in extreme detail.
**Paper Context:**
- ArXiv ID: {arxiv_id}
- Research Topic Context: {research_topic}
- Specification Focus: {specification_prompt}

**Full Text:**
```{text_for_prompt}```
**Review Instructions (Follow Strictly):**
 Provide a structured review covering EACH of the following sections with the specified minimum detail. Be analytical and critical.

 1.  **Title and Core Contribution (minimum 6-10 sentences):**
     - State the paper's full title.
     - Concisely summarize the core problem, the proposed approach/method, and the main claim or finding as presented in the abstract and introduction.

 2.  **Detailed Methods (minimum 20-25 sentences):**
     - Describe the methodology in detail. What specific algorithms, experimental setups, simulation parameters, or theoretical frameworks were used?
     - Explain the key steps taken to implement the method and conduct the research.
     - Were there specific phases (e.g., data preprocessing, model training, evaluation)? Describe each.

 3.  **Resources and Datasets (minimum 20-25 sentences):**
     - Identify and describe all significant resources used. This includes datasets (source, size, characteristics), software libraries (versions if specified), specific hardware (e.g., GPUs), or any other crucial infrastructure.
     - Explain *why* these specific resources were chosen or necessary for the study.

 4.  **Key Results and Findings (minimum 20-25 sentences):**
     - Summarize the main results presented. Include quantitative findings (e.g., specific metric values, p-values, performance improvements) whenever available in the text.
     - What are the key conclusions drawn by the authors based on these results?
     - Did the results support the initial hypothesis or research question?

 5.  **Full Research Pipeline Summary (minimum 25-30 sentences):**
     - Synthesize the entire research process described in the paper, from the initial problem definition/question to the final conclusions.
     - Outline the sequence of key stages (e.g., data collection -> method development -> experimentation -> analysis -> conclusion).
     - How does each stage connect and contribute to the overall findings?

 6.  **Stated Limitations and Critical Assessment (minimum 20-25 sentences):**
     - What limitations, weaknesses, or future work directions did the *authors themselves* mention?
     - Does the paper compare its results to existing methods? Briefly summarize the comparison outcomes.
     - Based on your expert analysis of the provided text, are there any other potential limitations or critical points regarding the methodology, results interpretation, or scope?
     - What is the specific novel contribution of this paper in the context of existing work mentioned within the paper itself?
     - How relevant and novel do you assess this paper to be concerning the specific topic: '{research_topic}'? Justify briefly.
     - Are there indications of data/code availability for reproducibility?

 **Output Format:**
 Produce ONLY the structured review text. Do not add any introductory or concluding remarks outside of the specified sections.
"""

# --- Prompts for ReviewCriticAgent ---

# Base system role and instructions for generating literature review sections
# {research_topic}, {specification_prompt} will be filled
LIT_REVIEW_SECTION_BASE_SYSTEM_ROLE_TEMPLATE: str = """
You are an expert research synthesizer and scientific writer. Your task is to write sections of a **critically analytical, highly informative, and DETAILED** literature review on:
**Research Topic:** {research_topic}
**Specification:** {specification_prompt}
You MUST base your writing **strictly** on the provided Article Reviews, identified by [ID: number] at the start of each review. Synthesize, compare, contrast, and critique the information **in depth**.
**Citation Instruction:** When referencing findings or methods from a specific article review, cite it using ONLY the format `[ID: number]`, for example: `[ID: 2410.11550v1]`. Do NOT use full URLs or author names for citations within the text.
**Formatting Instruction:** Do NOT include Markdown headings (like ## Introduction or **Introduction**) in your response text. **Do NOT use markdown tables.** Describe all information using prose sentences and paragraphs.
**Detail Requirement:** Ensure each section provides **substantial detail, thorough analysis, and sufficient examples** drawn directly from the provided reviews to justify claims and comparisons. Avoid superficial summaries.
"""

# Templates for each section of the literature review
# {system_role} will be LIT_REVIEW_SECTION_BASE_SYSTEM_ROLE_TEMPLATE
# {critic_instructions}, {article_reviews}, {previous_sections}, {research_topic}, {specification_prompt} will be filled
LIT_REVIEW_SECTIONS_CONFIG: Dict[str, Dict[str, Any]] = {
    "introduction": {
        "template": """
{system_role}

{critic_instructions}
**Task:** Write ONLY the **Introduction** section body text.

**Instructions for Introduction (minimum 30-35 sentences):**
- Briefly introduce the research topic '{research_topic}' and its significance.
- State the specific scope or question this review addresses, guided by the specification '{specification_prompt}'.
- Briefly mention the types of studies included (based on the reviews [ID: number]) and the overall goal of this review (e.g., synthesize methods, identify gaps).
- Highlight the importance of this research area.

**Article Reviews Provided:**
{article_reviews}

**Output:** Write ONLY the Introduction section body text (no heading).
""",
        "previous_section_keys": []  # No previous sections for introduction
    },
    "key_themes": {
        "template": """
{system_role}

{critic_instructions}
**Task:** Write ONLY the **Key Themes and Trends** section body text, building upon the (unseen) introduction.

**Previously Written Sections (Context Only):**
{previous_sections} 

**Instructions for Key Themes and Trends (minimum 20-25 sentences):**
- **Critically analyze and synthesize** the provided Article Reviews [ID: number].
- Identify and discuss 2-4 major recurring themes, trends, or common approaches found across the papers.
- For each theme, explain it clearly and provide specific supporting examples **directly** from the article reviews, citing using `[ID: number]` format.
- Compare and contrast how different papers address these themes.
- Discuss the significance or implications of these trends.
- Identify and discuss any **emerging or less common (but potentially important) trends** observed in the reviewed articles.

**Article Reviews Provided:**
{article_reviews}

**Output:** Write ONLY the Key Themes and Trends section body text (no heading).
""",
        "previous_section_keys": ["introduction"]
    },
    "methods_technologies": {
        "template": """
{system_role}

{critic_instructions}
**Task:** Write ONLY the **Methods and Technologies** section body text.

**Previously Written Sections (Context Only):**
{previous_sections}

**Instructions for Methods and Technologies (minimum 30-35 sentences):**
- **Critically analyze and compare** the different methods and technologies described in the Article Reviews [ID: number].
- Group similar methods. Discuss the strengths and weaknesses of key approaches *as suggested by the reviews or your analysis of them*.
- Provide specific details about prominent algorithms, techniques, or experimental procedures mentioned.
- Analyze and compare the **evaluation metrics** used in the different studies [ID: number]. Are they consistent? Are they appropriate for the tasks? Discuss any limitations noted regarding evaluation.
- Are there notable differences in methodology for similar problems? Discuss them.
- Cite specific examples using `[ID: number]` format.

**Article Reviews Provided:**
{article_reviews}

**Output:** Write ONLY the Methods and Technologies section body text (no heading).
""",
        "previous_section_keys": ["introduction", "key_themes"]
    },
    "resources_datasets": {
        "template": """
{system_role}

{critic_instructions}
**Task:** Write ONLY the **Resources and Datasets** section body text.

**Previously Written Sections (Context Only):**
{previous_sections}

**Instructions for Resources and Datasets (minimum 20-25 sentences):**
- **Synthesize** information about the datasets, software, hardware, or other resources mentioned in the Article Reviews [ID: number].
- Identify commonly used resources or types of resources.
- Discuss the characteristics of key datasets mentioned (e.g., size, type, source).
- Are there mentions of resource limitations or the importance of specific resources? Include these insights.
- Cite specific examples using `[ID: number]` format.

**Article Reviews Provided:**
{article_reviews}

**Output:** Write ONLY the Resources and Datasets section body text (no heading).
""",
        "previous_section_keys": ["introduction", "key_themes", "methods_technologies"]
    },
    "challenges_limitations": {
        "template": """
{system_role}

{critic_instructions}
**Task:** Write ONLY the **Challenges and Limitations** section body text.

**Previously Written Sections (Context Only):**
{previous_sections}

**Instructions for Challenges and Limitations (minimum 20-25 sentences):**
- **Synthesize** the challenges, limitations, and weaknesses identified **within the provided Article Reviews** [ID: number] (both self-reported by authors and noted in the review itself).
- **Categorize** the identified challenges (e.g., data-related, methodological, computational, ethical).
- Discuss the **potential impact** of each major challenge category on the field's progress.
- Are there specific **mitigation strategies** proposed or implied in the reviewed papers for these challenges? Discuss them.
- What are the recurring obstacles or gaps mentioned across the papers?
- Discuss the implications of these limitations for the field and for the research topic '{research_topic}'.
- Cite specific examples using `[ID: number]` format.

**Article Reviews Provided:**
{article_reviews}

**Output:** Write ONLY the Challenges and Limitations section body text (no heading).
""",
        "previous_section_keys": ["introduction", "key_themes", "methods_technologies", "resources_datasets"]
    },
    "proposed_pipeline": {
        "template": """
{system_role}

{critic_instructions}
**Task:** Write ONLY the **Proposed Research Pipeline** section body text.

**Previously Written Sections (Context Only):**
{previous_sections}

**Instructions for Proposed Research Pipeline (minimum 30-35 sentences):**
1.  **Introduction Paragraph(s):** Begin with 1-2 paragraphs outlining the overall goal and core principles of the proposed pipeline. Explain *why* this pipeline structure is proposed, based **strictly** on the analysis of the provided Article Reviews (methods, resources, challenges synthesized in previous sections). Briefly mention the key limitations it aims to address.
2.  **Pipeline Stages (Describe each stage in a separate paragraph, clearly numbering or naming them, e.g., 'Stage 1: Data Curation', 'Stage 2: Model Adaptation', etc.):**
    *   Propose **5-7 distinct, sequential stages** for a comprehensive research pipeline addressing the topic '{research_topic}'.
    *   For **EACH stage**: 
    *   Clearly state the **purpose** of the stage.
    *   Suggest **specific, named methods, tools, or dataset types** inspired by the reviewed papers [ID: number].
    *   Briefly **justify** the choice of these components for this stage, linking back to the synthesized knowledge (e.g., 'Inspired by the RAG approach in [ID: X], this stage will...').
    *   Mention potential **evaluation metrics or intermediate validation points** for this stage.
3.  **Challenges and Mitigation Paragraph(s):** After describing the stages, dedicate 1-2 paragraphs to discuss the **potential challenges** specifically for *this proposed pipeline* (e.g., data integration, computational cost, validation hurdles) and suggest **concrete mitigation strategies**, referencing insights from the reviews [ID: number] where applicable.
4.  **Concluding Summary Paragraph:** End with a brief paragraph summarizing the expected outcomes and significance of implementing this proposed pipeline for advancing research on '{research_topic}'.
                     

**Article Reviews Provided:**
{article_reviews}

**Output:** Write ONLY the Proposed Research Pipeline section body text...
""",
        "previous_section_keys": ["introduction", "key_themes", "methods_technologies", "resources_datasets", "challenges_limitations"]
    },
    "conclusion": {
        "template": """
{system_role}

{critic_instructions}
**Task:** Write ONLY the **Conclusion** section body text.

**Previously Written Sections (Context Only):**
{previous_sections}

**Instructions for Conclusion (minimum 25-30 sentences):**
- Briefly summarize the key themes, dominant methods, and major challenges identified in the review.
- Provide an overall critical assessment of the state of research on '{research_topic}' based **only** on the reviewed articles.
- Reiterate the main insights regarding effective research approaches or pipelines derived from the synthesis.
- Briefly suggest the most promising directions for future research **emerging directly from the analysis of the reviewed papers and the proposed pipeline**.
- Conclude with the potential impact or significance of advancing research in this area.
- Cite key supporting points using `[ID: number]` format.

**Article Reviews Provided:**
{article_reviews}

**Output:** Write ONLY the Conclusion section body text (no heading).
""",
        "previous_section_keys": ["introduction", "key_themes", "methods_technologies", "resources_datasets", "challenges_limitations", "proposed_pipeline"]
    }
}

# --- Prompts for ReviewCriticAgent ---

# {research_topic}, {specification_prompt}, {literature_review_text_for_prompt}, {article_reviews_json_string}, {run_count} will be filled
CRITIC_REVIEW_PROMPT_TEMPLATE: str = """
You are a highly experienced and critical academic reviewer specializing in literature reviews.
Your task is to evaluate the following literature review (run {run_count}) and provide constructive feedback for improvement, following the structured JSON format requested.
The review focuses on:
Research Topic: "{research_topic}"
Specification: "{specification_prompt}"

**Literature Review Text to Evaluate:**
{literature_review_text_for_prompt}

**Article Reviews Used to Generate This Literature Review (potentially truncated):**
{article_reviews_json_string}

Evaluation Criteria & JSON Output Structure:

Evaluate the literature review based on the following criteria. For each criterion:
    Provide a concise "Assessment" explaining your reasoning with specific examples from the review text.
    Assign a "Score" from 1 (Very Poor) to 5 (Excellent).

Output your entire response as a SINGLE, VALID JSON object. Do NOT include ANY text outside this JSON object.

JSON Structure:
{{
  "scores": {{
    "clarity_and_focus": {{ "assessment": "...", "score": <INT 1-5> }},
    "depth_of_analysis": {{ "assessment": "...", "score": <INT 1-5> }},
    "coverage_of_individual_article_reviews": {{ "assessment": "...", "score": <INT 1-5> }},
    "diversity_of_citation_context": {{ "assessment": "...", "score": <INT 1-5> }},
    "proposed_research_pipeline_evaluation": {{ "assessment": "...", "score": <INT 1-5> }}
  }},
  "strengths": ["Strength 1...", "Strength 2..."],
  "weaknesses": ["Weakness 1...", "Weakness 2..."],
  "recommendations": ["Recommendation 1...", "Recommendation 2..."]
}}

Detailed Criteria Levels (for your reference during evaluation):
    Clarity and Focus (1-5): 5=Excellent, ..., 1=Lacks clarity.
    Depth of Analysis (1-5): 5=Highly insightful, ..., 1=No real analysis.
    Coverage of Individual Article Reviews (1-5): 5=Comprehensive (>80%), ..., 1=Almost no coverage (<20%).
    Diversity of Citation Context (1-5): 5=Key articles in 4-5+ sections, ..., 1=Articles in one section.
    Proposed Research Pipeline Evaluation (1-5): 5=Exceptionally well-defined, ..., 1=No pipeline or vague.

Ensure your final output is ONLY the JSON object.
"""

# --- Prompts for Title/Abstract Generation (ResearchAgent) ---
# {truncated_review} will be filled
TITLE_ABSTRACT_PROMPT_TEMPLATE: str = """
Based on the following literature review text, please generate:
1. A concise and informative Title (max 15 words).
2. A comprehensive Abstract summarizing the key findings, methods, challenges, and conclusions (150-250 words).

Literature Review Text:
{truncated_review}

Output Format:
Provide the Title first, followed by the Abstract on separate lines. Start the lines exactly with "Title:" and "Abstract:".
Ensure the abstract is a single paragraph.

Example:
Title: Applications of LLMs in Accelerating Drug Discovery
Abstract: This review synthesizes recent advancements in leveraging Large Language Models (LLMs) for drug discovery...

Your Turn:
Title:
Abstract:
"""

CRITICAL_REWRITE_INSTRUCTIONS_TEMPLATE: str = """
**CRITICAL REWRITE INSTRUCTIONS:**
You MUST rewrite the sections based on the following critique. Focus on addressing all weaknesses and recommendations:
{critic_feedback}
Ensure the rewritten sections are significantly improved and integrate insights from the Article Reviews more effectively.
---
"""