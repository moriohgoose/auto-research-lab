import logging
import json
import time
import os 
import re
from typing import Optional, List, Dict, Any 
from .review_agents import ResearchAgent, ReviewCriticAgent, ReviewRefinerAgent
from utils.pdf_generator import format_references_list, replace_ids_with_numbers

MODULE_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_OUTPUT_DIR = os.path.join(MODULE_ROOT_DIR, "pipeline_run_outputs")

def _prepare_final_output(
    title: Optional[str],
    abstract: Optional[str],
    review_text: Optional[str],
    review_sections: Dict[str, str],
    scores: Optional[Dict],
    run_number: int,
    total_iterations: int,
    topic: str,
    spec: Optional[str],
    keywords: Optional[List[str]],
    arxiv_ids: List[str], # IDs of articles used for the best review
    metadata: Dict[str, Dict], # Metadata for those articles
    termination_reason: str
) -> Dict[str, Any]:
    """
    Helper function to assemble the final result dictionary for the literature review.

    Args:
        title (Optional[str]): The generated title for the review.
        abstract (Optional[str]): The generated abstract for the review.
        review_text (Optional[str]): The full text of the best generated literature review.
        review_sections (Dict[str, str]): The structured sections of the best review.
        scores (Optional[Dict]): The final scores from the critic for the best review.
        run_number (int): The run number of the ResearchAgent that produced the best review.
        total_iterations (int): The total number of refinement iterations performed.
        topic (str): The research topic.
        spec (Optional[str]): The specification prompt used.
        keywords (Optional[List[str]]): The keywords used for search.
        arxiv_ids (List[str]): The ArXiv IDs of the articles included in the best review.
        metadata (Dict[str, Dict]): The metadata for the articles included in the best review.
        termination_reason (str): The reason why the iterative process stopped.

    Returns:
        Dict[str, Any]: A dictionary containing all the relevant output data.
    """
    # Format the final reference list based on the IDs and metadata of the best run.
    processed_abstract_for_display = abstract
    processed_review_sections_for_display = review_sections.copy()

    short_arxiv_ids_ordered_for_replacer = []
    for full_url in arxiv_ids:
        match = re.search(r'([\d.]+v?\d*)$', full_url) # Get 'XXXX.XXXXXvx'
        if match:
            short_arxiv_ids_ordered_for_replacer.append(match.group(1))
        else:
            # Fallback if ID cannot be extracted, use the full URL as key.
            logging.warning(f"Could not extract short ID from URL '{full_url}'. Using full URL for replacer map.")
            short_arxiv_ids_ordered_for_replacer.append(full_url) # fallback

    # Recreate the metadata dictionary with short IDs as keys for the replacer functio
    metadata_with_short_ids_for_replacer = {}
    for full_url_key, meta_val in metadata.items():
        match = re.search(r'([\d.]+v?\d*)$', full_url_key)
        if match:
            metadata_with_short_ids_for_replacer[match.group(1)] = meta_val
        else:
            metadata_with_short_ids_for_replacer[full_url_key] = meta_val


    # 1. Apply citation replacement to the abstract text.
    # This transforms placeholders like `[ID: 2103.01234]` to `[1]` in the final output.
    if abstract and short_arxiv_ids_ordered_for_replacer and metadata_with_short_ids_for_replacer:
        try:
            processed_abstract_for_display = replace_ids_with_numbers(
                abstract,
                short_arxiv_ids_ordered_for_replacer,
                metadata_with_short_ids_for_replacer
            )
        except Exception as e:
            logging.error(f"Error processing abstract for ID replacement: {e}")

    for section_key, section_content in review_sections.items():
        if section_content and short_arxiv_ids_ordered_for_replacer and metadata_with_short_ids_for_replacer:
            try:
                processed_review_sections_for_display[section_key] = replace_ids_with_numbers(
                    section_content,
                    short_arxiv_ids_ordered_for_replacer,
                    metadata_with_short_ids_for_replacer
                )
            except Exception as e:
                logging.error(f"Error processing section '{section_key}' for ID replacement: {e}")


    # 2. Format the list of references for display.
    # This function uses the original full arXiv IDs and metadata to construct
    # standard bibliography entries (e.g., "Authors (Year). Title. Journal.").
    final_references_for_display = []
    if arxiv_ids and metadata:
        try:
            final_references_for_display = format_references_list(arxiv_ids, metadata)
        except Exception as ref_e:
            logging.error(f"Error formatting references for final output: {ref_e}")

    # Return the comprehensive dictionary of all final output data
    return {
        "title": title, # The final generated title
        "abstract": processed_abstract_for_display, # Abstract with numbered citations
        "final_literature_review_text": review_text, # The full, raw text of the best review
        "final_review_sections": processed_review_sections_for_display, # Structured sections with numbered citations
        "references_formatted": final_references_for_display, # List of neatly formatted reference strings
        "final_review_scores": scores, # Scores from the critic for the best review
        "source_run_number": run_number, # The ResearchAgent run number that produced this best review
        "termination_reason": termination_reason, # Reason for pipeline termination
        "total_iterations_performed": total_iterations, # Total refinement iterations
        "research_topic": topic, # Original input research topic
        "specification_prompt": spec, # Original input specification prompt
        "keywords_used": keywords, # Actual keywords used in the search
        "best_run_arxiv_ids": arxiv_ids, # List of full arXiv URLs (for generating clickable links in UI)
        "best_run_metadata": metadata, # Original metadata for articles in the best review
    }


def run_review_pipeline(
    research_topic: str,
    specification_prompt: Optional[str],
    article_count_filter: int,
    article_years_filter: Optional[str],
    keywords_list: Optional[List[str]],
    max_iterations: int = 5,
    gemini_model_name: str = "gemini-2.0-flash"
) -> Optional[Dict[str, Any]]: 
    
    """
    Executes the iterative literature review generation and refinement pipeline.

    This function orchestrates the ResearchAgent, ReviewCriticAgent, and
    ReviewRefinerAgent to produce a literature review. It iteratively refines
    the review based on critic feedback until stopping criteria are met or
    the maximum number of iterations is reached.

    Args:
        research_topic (str): The main topic for the literature review.
        specification_prompt (Optional[str]): A specific focus or instruction for the review content.
        article_count_filter (int): The number of top-ranked articles to include in the review.
        article_years_filter (Optional[str]): A year range for filtering articles (e.g., "2020-2025").
                                              None for no year filter.
        keywords_list (Optional[List[str]]): A list of keywords for the article search.
                                             If None, keywords will be generated by ResearchAgent.
        max_iterations (int): The maximum number of refinement iterations to perform.
        gemini_model_name (str): The name of the Gemini model to use (e.g., "gemini-2.0-flash").
    
    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the final literature review data
                                  (title, abstract, text, sections, scores, metadata, etc.),
                                  or a dictionary with an "error" key if the process fails critically.
    """

    logging.info(f"Starting literature review pipeline for topic: '{research_topic}'")
    os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)

    # --- Research start ---
    try:
        # Initialize the ResearchAgent
        # It will handle its own run_count internally
        research_agent = ResearchAgent(
            research_topic=research_topic,
            keywords=keywords_list, # Pass user-provided or None
            specification_prompt=specification_prompt,
            article_count_filter=article_count_filter,
            article_years_filter=article_years_filter,
            gemini_model_name=gemini_model_name
        )

        iteration_count = 0
        previous_review_scores: Optional[Dict] = None # Scores from the previous iteration
        next_action = 'CONTINUE_IMPROVE' # Initial action
        force_rewrite_next_iteration = False
        critic_feedback_for_next_run: Optional[str] = None

        # --- Best review ---
        best_review_text: Optional[str] = None
        best_review_sections: Dict[str, str] = {}
        best_review_score_avg: float = -1.0  # Initialize with a low value
        best_review_scores_dict: Optional[Dict] = None
        best_review_run_number: int = -1
        best_review_arxiv_ids: List[str] = []
        best_review_metadata: Dict[str, Dict] = {}
        final_termination_reason: str = f"Max iterations ({max_iterations}) reached"

        # --- Main Loop ---
        while iteration_count < max_iterations:
            iteration_count += 1
            logging.info(f"\n--- Literature Review Iteration {iteration_count}/{max_iterations} ---")

            # 1. Generate/Rewrite Literature Review using ResearchAgent
            # The research_agent.run() method increments its internal run_count
            current_literature_review_text = research_agent.run(
                critic_feedback_and_weaknesses=critic_feedback_for_next_run,
                force_rewrite_from_scratch=force_rewrite_next_iteration
            )
            current_research_agent_run = research_agent.run_count # Get the run number for this iteration

            # Store details from the current ResearchAgent run
            current_review_sections = research_agent.review_sections.copy()
            current_article_metadata = research_agent.article_metadata.copy()
            current_relevant_arxiv_ids = research_agent.relevant_arxiv_ids.copy()
            current_keywords_used = research_agent.keywords # Store actual keywords used

            # Reset flags for the next potential iteration
            force_rewrite_next_iteration = False
            critic_feedback_for_next_run = None

            if not current_literature_review_text:
                logging.error(f"ResearchAgent failed to produce a review in iteration {iteration_count}. Stopping.")
                # If a previous best review exists, return that with an error message for the current iteration
                if best_review_text:
                    return _prepare_final_output(
                        title=None, abstract=None, # Title/abstract would be for the best one
                        review_text=best_review_text, review_sections=best_review_sections,
                        scores=best_review_scores_dict, run_number=best_review_run_number,
                        total_iterations=iteration_count -1, # Iterations completed before failure
                        topic=research_agent.research_topic, spec=research_agent.specification_prompt,
                        keywords=current_keywords_used, arxiv_ids=best_review_arxiv_ids,
                        metadata=best_review_metadata,
                        termination_reason=f"Pipeline failed at ResearchAgent run {current_research_agent_run} (iteration {iteration_count})"
                    )
                else:
                    return {"error": f"Pipeline failed critically at ResearchAgent run {current_research_agent_run} (iteration {iteration_count})"}

            # --- Critics ---
            # 2. Critique the Generated Review
            # The critic needs the path to the JSON file saved by ResearchAgent.run()
            current_review_json_path = os.path.join(RUN_OUTPUT_DIR, f"run_{current_research_agent_run}_literature_review.json")
            if not os.path.exists(current_review_json_path):
                logging.error(f"Intermediate review JSON not found: {current_review_json_path}. Cannot critique. Stopping.")
                return {"error": f"Missing intermediate file: {os.path.basename(current_review_json_path)} for iteration {iteration_count}"}

            try:
                critic_agent = ReviewCriticAgent(
                    literature_review_json_path=current_review_json_path,
                    gemini_model_name=gemini_model_name
                    )
                review_results_json = critic_agent.literature_review_critics() # This saves its own JSON
            except Exception as e:
                logging.error(f"Error during critique in iteration {iteration_count}: {e}. Stopping.", exc_info=True)
                return {"error": f"Pipeline failed at ReviewCriticAgent for iteration {iteration_count}"}

            current_scores_dict = review_results_json.get("scores", {})
            if not current_scores_dict:
                logging.warning(f"Critic Agent did not return valid scores for iteration {iteration_count}. Stopping.")
                return {"error": f"Pipeline failed due to missing scores from critic for iteration {iteration_count}"}
            
            # --- Best review check ---
            # 3. Check if this is the best review so far
            current_score_sum = 0
            valid_scores_count = 0
            for criterion in ReviewRefinerAgent.RELEVANT_CRITERIA: # Accessing class variable
                score = ReviewRefinerAgent._get_score(current_scores_dict, criterion)
                if score is not None:
                    current_score_sum += score
                    valid_scores_count += 1
            current_scores_avg = (current_score_sum / valid_scores_count) if valid_scores_count > 0 else 0.0
            logging.info(f"Iteration {iteration_count} - Average Score: {current_scores_avg:.2f}")

            if current_scores_avg >= best_review_score_avg:
                logging.info(f"New best review found (or equal score)! Avg score: {current_scores_avg:.2f} >= {best_review_score_avg:.2f}")
                best_review_score_avg = current_scores_avg
                best_review_text = current_literature_review_text
                best_review_sections = current_review_sections
                best_review_scores_dict = current_scores_dict
                best_review_run_number = current_research_agent_run # Save the run number of this best review
                best_review_arxiv_ids = current_relevant_arxiv_ids
                best_review_metadata = current_article_metadata
            else:
                logging.info(f"Current avg score ({current_scores_avg:.2f}) not better than best ({best_review_score_avg:.2f} from run {best_review_run_number}).")

            # --- Detect next action ---
            # 4. Determine Next Action using ReviewRefinerAgent
            # The refiner needs the path to the critic's output JSON
            current_critic_json_path = os.path.join(RUN_OUTPUT_DIR, f"run_{current_research_agent_run}_review_critics.json")
            if not os.path.exists(current_critic_json_path):
                logging.error(f"Critic results JSON not found: {current_critic_json_path}. Cannot refine. Stopping.")
                return {"error": f"Missing critic results file: {os.path.basename(current_critic_json_path)} for iteration {iteration_count}"}

            try:
                refiner_agent = ReviewRefinerAgent(
                    literature_review_json_path=current_review_json_path, # Path to ResearchAgent's output for this run
                    review_critics_json_path=current_critic_json_path     # Path to CriticAgent's output for this run
                )
                next_action = refiner_agent.determine_next_action(previous_scores=previous_review_scores)

                if next_action == 'CONTINUE_IMPROVE':
                    critic_feedback_for_next_run = refiner_agent.get_critic_feedback_for_rewrite()

            except Exception as e:
                logging.error(f"Error determining next action in iteration {iteration_count}: {e}. Stopping.", exc_info=True)
                return {"error": f"Pipeline failed at ReviewRefinerAgent for iteration {iteration_count}"}
            
            # --- Next action ---
            # 5. Process Next Action
            final_termination_reason = next_action # Update termination reason

            if next_action in ['STOP_ALL_5', 'STOP_PLATEAU_OR_WORSE_HIGH', 'ERROR_NO_SCORES']:
                logging.info(f"Stopping condition met: {next_action}. Iteration {iteration_count}.")
                if next_action == 'STOP_ALL_5': # If all 5s, current review is the best
                    best_review_text = current_literature_review_text
                    best_review_sections = current_review_sections
                    best_review_scores_dict = current_scores_dict
                    best_review_run_number = current_research_agent_run
                    best_review_arxiv_ids = current_relevant_arxiv_ids
                    best_review_metadata = current_article_metadata
                break # Exit the while loop
            
            elif next_action == 'REWRITE_FROM_SCRATCH':
                logging.info("Triggering Rewrite: Will rewrite from scratch in the next iteration.")
                force_rewrite_next_iteration = True
                critic_feedback_for_next_run = None # No specific feedback, just rewrite
                previous_review_scores = None # Reset previous scores for a fresh start assessment
            
            elif next_action == 'CONTINUE_IMPROVE':
                logging.info("Continuing refinement in the next iteration.")
                previous_review_scores = current_scores_dict # Update for the next comparison
            
            else: # Unknown action
                logging.warning(f"Unknown next action '{next_action}' determined. Stopping pipeline.")
                break
            
            # If max_iterations is reached and we didn't break earlier
            if iteration_count >= max_iterations and next_action != 'REWRITE_FROM_SCRATCH':
                 logging.info(f"Max iterations ({max_iterations}) reached. Stopping.")
                 final_termination_reason = f"Max iterations ({max_iterations}) reached"
                 break

        # --- End of Loop ---
        logging.info("\n--- Literature Review Iterative Process Finished ---")

        # --- Final Output Processing ---
        # Prepare final output based on the best review found
        if best_review_text and best_review_sections:
            final_title: Optional[str] = "Literature Review" # Default
            final_abstract: Optional[str] = "No abstract generated." # Default
            
            # Title + abstract 
            # Attempt to generate title and abstract for the *best* review
            try:
                gen_title, gen_abstract = research_agent.generate_title_abstract(best_review_text)
                if gen_title: final_title = gen_title
                if gen_abstract: final_abstract = gen_abstract
            except Exception as ta_e:
                logging.error(f"Error during final title/abstract generation: {ta_e}")

            return _prepare_final_output(
                title=final_title,
                abstract=final_abstract,
                review_text=best_review_text,
                review_sections=best_review_sections,
                scores=best_review_scores_dict,
                run_number=best_review_run_number,
                total_iterations=iteration_count, # Total iterations performed
                topic=research_agent.research_topic, # From the (last) research_agent instance
                spec=research_agent.specification_prompt,
                keywords=research_agent.keywords, # Actual keywords used by research_agent
                arxiv_ids=best_review_arxiv_ids,
                metadata=best_review_metadata,
                termination_reason=final_termination_reason
            )
        else:
            logging.error("Pipeline finished, but no best review was selected or generated.")
            return {"error": "Pipeline finished, but no best review was selected or generated."}

    except Exception as pipeline_e:
        logging.critical(f"A critical error occurred within the run_review_pipeline function: {pipeline_e}", exc_info=True)
        return {"error": f"Critical error in literature review pipeline: {str(pipeline_e)}"}