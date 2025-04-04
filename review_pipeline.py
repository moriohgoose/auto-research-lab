import logging
import json
import time
import os 
from agents import ResearchAgent, ReviewCriticAgent, ReviewRefinerAgent
from utils import generate_pdf_report, format_references_list
from typing import Optional, List, Dict, Any 

RUN_OUTPUT_DIR = "review_runs"

def _prepare_final_output(
    title, abstract, review_text, review_sections, references,
    scores, run_number, total_iterations,
    topic, spec, keywords, arxiv_ids, metadata, termination_reason
) -> Dict[str, Any]:
    
    """Helper function to assemble the final result dictionary."""

    final_references = []
    if arxiv_ids and metadata:
        try:
            final_references = format_references_list(arxiv_ids, metadata)
        except Exception as ref_e:
            logging.error(f"Error formatting references for final output: {ref_e}")

    return {
        "title": title,
        "abstract": abstract,
        "final_literature_review_text": review_text,
        "final_review_sections": review_sections,
        "references_formatted": final_references, 
        "final_review_scores": scores,
        "source_run_number": run_number,
        "termination_reason": termination_reason,
        "total_iterations": total_iterations,
        "research_topic": topic,
        "specification_prompt": spec,
        "keywords": keywords,
        "best_run_arxiv_ids": arxiv_ids,
        "best_run_metadata": metadata, 
    }


def run_review_pipeline(
    research_topic: str,
    specification_prompt: Optional[str],
    article_count_filter: int,
    article_years_filter: Optional[str],
    keywords_list: Optional[List[str]],
    MAX_ITERATIONS: int = 5
) -> Optional[Dict[str, Any]]: 
    
    """
    Executes the iterative literature review generation and refinement pipeline.

    Args:
        research_topic: The main topic.
        specification_prompt: Specific focus for the review.
        article_count_filter: Number of articles to include.
        article_years_filter: Year range filter (e.g., "2020-2025").
        keywords_list: Optional list of keywords provided by the user.
        MAX_ITERATIONS: Maximum number of refinement iterations.

    Returns:
        A dictionary containing the final review data (title, abstract, text, sections,
        references, scores, metadata, etc.), or None if the process fails critically.
    """

    logging.info(f"Starting pipeline for topic: '{research_topic}'")

    # --- Research start ---
    try:
        research_agent = ResearchAgent(
            research_topic=research_topic,
            keywords=keywords_list,
            specification_prompt=specification_prompt,
            article_count_filter=article_count_filter,
            article_years_filter=article_years_filter,
        )

        iteration_count = 0
        previous_review_scores = None
        next_action = 'CONTINUE_IMPROVE'
        force_rewrite_next_iteration = False
        critic_feedback_for_next_run = None

        # --- Best review ---
        best_review_text: Optional[str] = None
        best_review_sections: Dict[str, str] = {}
        best_review_score_avg: float = -1.0
        best_review_scores_dict: Optional[Dict] = None
        best_review_run_number: int = -1
        best_review_arxiv_ids: List[str] = []
        best_review_metadata: Dict[str, Dict] = {}

        # --- Main Loop ---
        while iteration_count < MAX_ITERATIONS:
            iteration_count += 1
            logging.info(f"\n--- Iteration {iteration_count}/{MAX_ITERATIONS} ---")

            current_literature_review_text = research_agent.run( 
                critic_feedback_and_weaknesses=critic_feedback_for_next_run,
                force_rewrite_from_scratch=force_rewrite_next_iteration
            )
            
            current_review_sections = research_agent.review_sections.copy()
            current_metadata = research_agent.article_metadata.copy()
            current_arxiv_ids = research_agent.relevant_arxiv_ids.copy()

            force_rewrite_next_iteration = False
            critic_feedback_for_next_run = None

            if not current_literature_review_text:
                logging.error(f"ResearchAgent failed to produce a review in iteration {iteration_count}. Stopping.")
                
                if best_review_text:
                    return {
                        "error": f"Pipeline failed at ResearchAgent run {iteration_count}",
                        # Helper for data collection
                        **_prepare_final_output( 
                            None, None, best_review_text, best_review_sections, None,
                            best_review_scores_dict, best_review_run_number, iteration_count-1,
                            research_agent.research_topic, research_agent.specification_prompt,
                            research_agent.keywords, best_review_arxiv_ids, best_review_metadata
                            )
                        }
                else:
                    return {"error": f"Pipeline failed critically at ResearchAgent run {iteration_count}"}

            # --- Critics ---
            current_review_json_path = os.path.join(RUN_OUTPUT_DIR, f"run_{research_agent.run_count}_literature_review.json")            
            if not os.path.exists(current_review_json_path):
                logging.error(f"Intermediate review JSON not found: {current_review_json_path}. Stopping pipeline.")
                return {"error": f"Missing intermediate file: {os.path.basename(current_review_json_path)}"}    

            try:    
                critic_agent = ReviewCriticAgent(current_review_json_path)
                review_results_json = critic_agent.literature_review_critics()
            
            except Exception as e:
                logging.error(f"Error during critique: {e}. Stopping.", exc_info=True)
                return {"error": f"Pipeline failed at ReviewCriticAgent run {iteration_count}"}


            current_scores_dict = review_results_json.get("scores", {})
            if not current_scores_dict:
                logging.warning(f"Critic Agent did not return valid scores for iteration {iteration_count}. Stopping.")
                return {"error": f"Pipeline failed due to missing scores at run {iteration_count}"}
            
            # --- Best review check ---
            current_score_sum = 0 
            valid_scores_count = 0

            for criterion in ReviewRefinerAgent.RELEVANT_CRITERIA:
                score = ReviewRefinerAgent._get_score(current_scores_dict, criterion) 
                if score is not None: current_score_sum += score; valid_scores_count += 1
            current_scores_avg = (current_score_sum / valid_scores_count) if valid_scores_count > 0 else 0
            logging.info(f"Iteration {iteration_count} - Average Score: {current_scores_avg:.2f}")

            if current_scores_avg >= best_review_score_avg: 
                logging.info(f"New best review found (or equal score)! (Avg score: {current_scores_avg:.2f} >= {best_review_score_avg:.2f})")
                best_review_score_avg = current_scores_avg
                best_review_text = current_literature_review_text 
                best_review_sections = current_review_sections 
                best_review_scores_dict = current_scores_dict
                best_review_run_number = research_agent.run_count
                best_review_arxiv_ids = current_arxiv_ids
                best_review_metadata = current_metadata  
            else:
                logging.info(f"Current avg score ({current_scores_avg:.2f}) not better than best ({best_review_score_avg:.2f} from run {best_review_run_number}).")

            # --- Detect next action ---
            try:
                current_critic_json_path = os.path.join(RUN_OUTPUT_DIR, f"run_{research_agent.run_count}_review_critics.json")

                if not os.path.exists(current_review_json_path):
                     logging.error(f"Intermediate review JSON not found for Refiner: {current_review_json_path}.")
                     return {"error": f"Missing intermediate file for Refiner: {os.path.basename(current_review_json_path)}"}
                if not os.path.exists(current_critic_json_path):
                     logging.error(f"Critic results JSON not found for Refiner: {current_critic_json_path}.")
                     return {"error": f"Missing critic results file for Refiner: {os.path.basename(current_critic_json_path)}"}
                
                refiner_agent = ReviewRefinerAgent(current_review_json_path, current_critic_json_path)
                next_action = refiner_agent.determine_next_action(previous_scores=previous_review_scores)
                if next_action == 'CONTINUE_IMPROVE':
                    critic_feedback_for_next_run = refiner_agent.get_critic_feedback_for_rewrite()
        
            except Exception as e:
                logging.error(f"Error determining next action: {e}. Stopping.", exc_info=True)
                return {"error": f"Pipeline failed at ReviewRefinerAgent run {iteration_count}"}

            # --- Next action ---
            if next_action in ['STOP_ALL_5', 'STOP_PLATEAU_OR_WORSE_HIGH', 'ERROR_NO_SCORES'] or \
            (iteration_count >= MAX_ITERATIONS and next_action != 'REWRITE_FROM_SCRATCH'): 
                logging.info(f"Stopping condition met: {next_action} or Max Iterations Reached.")
                
                if next_action == 'STOP_ALL_5':
                    best_review_text = current_literature_review_text
                    best_review_sections = current_review_sections
                    best_review_scores_dict = current_scores_dict
                    best_review_run_number = research_agent.run_count
                    best_review_arxiv_ids = current_arxiv_ids
                    best_review_metadata = current_metadata
                break 
            
            elif next_action == 'REWRITE_FROM_SCRATCH':
                logging.info("Triggering Rewrite: Will rewrite from scratch.")
                force_rewrite_next_iteration = True
                critic_feedback_for_next_run = None
                previous_review_scores = None 
            
            elif next_action == 'CONTINUE_IMPROVE':
                logging.info("Continuing refinement.")
                previous_review_scores = current_scores_dict 
            
            else:
                logging.warning(f"Unknown next action '{next_action}'. Stopping pipeline.")
                break

        # --- End of Loop ---


        # --- Final Output Processing ---
        logging.info("\n--- Iterative Process Finished ---")

        final_title = "Literature Review"
        final_abstract = "No abstract generated."

        # Best review case
        if best_review_text and best_review_sections:
            
            # Title + abstract 
            try:
                gen_title, gen_abstract = research_agent.generate_title_abstract(best_review_text)
                if gen_title: final_title = gen_title
                if gen_abstract: final_abstract = gen_abstract
            except Exception as ta_e: logging.error(f"Error during title/abstract generation: {ta_e}")

            # helper for final dict collection
            final_output_data = _prepare_final_output(
                title=final_title, abstract=final_abstract,
                review_text=best_review_text, review_sections=best_review_sections,
                references=None, 
                scores=best_review_scores_dict, run_number=best_review_run_number,
                total_iterations=iteration_count,
                topic=research_agent.research_topic, spec=research_agent.specification_prompt,
                keywords=research_agent.keywords, arxiv_ids=best_review_arxiv_ids,
                metadata=best_review_metadata, 
                termination_reason=next_action or f"Max iterations ({MAX_ITERATIONS}) reached"
            )

            return final_output_data
        
        else:
                logging.error("Pipeline finished but no best review was generated.")
                return {"error": "Pipeline finished but no best review was generated."}

    except Exception as pipeline_e:
        logging.critical(f"A critical error occurred within the pipeline function: {pipeline_e}", exc_info=True)
        return {"error": f"Critical error in pipeline: {pipeline_e}"}