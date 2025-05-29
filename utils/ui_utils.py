import streamlit as st
import os
import time
import json
import logging
import re

logger = logging.getLogger(__name__) # Initialize a logger


# --- Function for Creating Clickable Citations ---
def make_citations_clickable(text_with_numbers, urls_ordered_list):
    """
    Converts numerical citation placeholders in a text (e.g., "[1]", "[1, 5]") into
    clickable HTML links, using a provided ordered list of URLs.

    This function is crucial for making the generated literature review content
    interactive and allowing users to easily navigate to the source articles.

    Args:
        text_with_numbers (str): The input text containing numerical citations
                                 formatted as `[number]` or `[number1, number2, ...]`.
        urls_ordered_list (list): A list of URLs, where the index `i` corresponds
                                  to citation number `i+1`. For example, `urls_ordered_list[0]`
                                  is the URL for citation [1].

    Returns:
        str: The processed text with numerical citations replaced by HTML links.
             If `urls_ordered_list` is empty or `text_with_numbers` is not a string,
             the original text is returned. Citations with numbers out of bounds
             of the URL list will be noted as such.
    """
    # Return original text if no URLs are provided or text is not a string
    if not urls_ordered_list or not isinstance(text_with_numbers, str):
        return text_with_numbers
    
    # Regex pattern to find numerical citations like [1], [1, 2], [1, 5, 10].
    # It captures the numbers inside the brackets.
    pattern = r'\[((?:\d+)(?:,\s*\d+)*)\]' 

    def simpler_replacer(match_obj):
        """
        Inner function used by `re.sub` to replace each matched citation.
        It converts the numbers into clickable links.
        """
        numbers_str = match_obj.group(1)
        try:
            # Converts the string of numbers into a list of integers
            citation_numbers = [int(n.strip()) for n in numbers_str.split(',')]
        except ValueError:
            # If conversion fails (e.g., non-numeric text), return the original match.
            logger.warning(f"Failed to parse citation numbers in '{numbers_str}'. Returning original text.")
            return match_obj.group(0) 

        output_parts = [] # List to build the new linked citation string.
        for num in citation_numbers:
            # Check if the citation number is within the valid range of URLs
            if 1 <= num <= len(urls_ordered_list):
                url = urls_ordered_list[num-1] # Get the URL (list is 0-indexed, citations are 1-indexed)
                output_parts.append(f'[{num}]({url})') # Format as a Markdown link
            else:
                # If the number is out of bounds, indicate it
                logger.warning(f"Citation number {num} is out of bounds for the URL list (len={len(urls_ordered_list)}).")
                output_parts.append(f'[{num} (URL ref. out of bounds)]')
        # Joins multiple citations (e.g., "[1](url1), [5](url5)")
        return ", ".join(output_parts)

    # Applies the `simpler_replacer` function to all matches of the pattern in the text    
    return re.sub(pattern, simpler_replacer, text_with_numbers)


# --- Form for Literature Review Generation ---
def display_review_form():
    """
    Displays the Streamlit form for generating a literature review.

    This function handles user input for the research topic, specification prompt,
    article filters, and keywords. It then triggers the backend `run_review_pipeline`
    and manages the display of progress, success/error messages, and download links
    for the generated PDF and JSON outputs.
    """
    # Imports necessary modules at function level to avoid circular dependencies
    # or to ensure they are only loaded when this function is called.
    try:
        import config # Project configuration settings.
        from review_module.review_pipeline import run_review_pipeline # Backend function to run the review pipeline.
        from utils.pdf_generator import save_review_pdf # Utility to save the review as a PDF.
    except ImportError as e:
        st.error(f"Error loading review modules: {e}. Please ensure `config.py`, `review_module`, and `utils.pdf_generator` are accessible.")
        logger.critical(f"Failed to load review-related modules in display_review_form: {e}", exc_info=True)
        return

    st.subheader("📝 Generate Literature Review")
    with st.form(key="review_form"):
        default_topic = "LLMs in Drug Research"
        research_topic = st.text_input("Research Topic", value=default_topic)

        default_spec = "Focus on novel methods, evaluation metrics, and ethical considerations."
        specification_prompt = st.text_area("Specification Prompt (optional)", value=default_spec, height=100)

        default_count = 10
        article_count_filter = st.number_input("Number of Articles to include", min_value=1, value=default_count, step=1)

        default_years = "2020-2025"
        article_years_filter_input = st.text_input(
            "Article Years Filter (YYYY-YYYY, optional, 'skip' or leave empty for default)",
            value=default_years
        )
        if not article_years_filter_input.strip() or article_years_filter_input.lower() == 'skip':
            article_years_filter = None if article_years_filter_input.lower() == 'skip' else default_years
        else:
            article_years_filter = article_years_filter_input.strip()

        keywords_input = st.text_input("Keywords (comma-separated, optional, press Enter to auto-generate)")
        keywords_list = [k.strip() for k in keywords_input.split(',') if k.strip()] or None

        max_iterations = st.number_input("Max Refinement Iterations", min_value=1, max_value=10, value=getattr(config, 'REVIEW_MAX_ITERATIONS', 3), step=1)

        submit_button = st.form_submit_button(label="Generate Review")

    # This block executes when the form is submitted and no other process is running
    if submit_button and not st.session_state.running:
        st.session_state.running = True
        st.session_state.results = None
        st.session_state.output_pdf_path = None
        st.session_state.output_json_path = None
        with st.spinner("Generating literature review... This may take several minutes."):
            try:
                logger.info(f"Starting Literature Review Pipeline with params: topic='{research_topic}', spec='{specification_prompt}', count={article_count_filter}, years='{article_years_filter}', keywords={keywords_list}, iterations={max_iterations}")
                # Call the backend review pipeline                
                final_data = run_review_pipeline(
                    research_topic=research_topic,
                    specification_prompt=specification_prompt if specification_prompt.strip() else None,
                    article_count_filter=article_count_filter,
                    article_years_filter=article_years_filter,
                    keywords_list=keywords_list,
                    max_iterations=max_iterations,
                    gemini_model_name=config.GEMINI_API_MODEL_NAME
                )
                st.session_state.results = final_data
                if final_data and final_data.get("error") is None:
                    st.success("Literature Review generated successfully!")
                    if save_review_pdf and final_data.get("final_review_sections"):
                        pdf_filename = f"literature_review_{research_topic.replace(' ','_')[:20]}_{int(time.time())}.pdf"
                        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
                        pdf_filepath = os.path.join(config.OUTPUT_DIR, pdf_filename)
                        if save_review_pdf(
                            title=final_data.get("title"),
                            abstract=final_data.get("abstract"),
                            review_sections=final_data.get("final_review_sections"),
                            references=final_data.get("references_formatted", []),
                            arxiv_ids_ordered=final_data.get("best_run_arxiv_ids", []),
                            article_metadata=final_data.get("best_run_metadata", {}),
                            filename=pdf_filepath
                        ):
                            st.session_state.output_pdf_path = pdf_filepath
                        else:
                            st.warning("Failed to save PDF review.")

                    json_filename = f"final_review_output_{int(time.time())}.json"
                    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
                    json_filepath = os.path.join(config.OUTPUT_DIR, json_filename)
                    with open(json_filepath, "w", encoding='utf-8') as f:
                        json.dump(final_data, f, indent=4, ensure_ascii=False, default=str)
                    st.session_state.output_json_path = json_filepath

                elif final_data and final_data.get("error"):
                    st.error(f"Error generating review: {final_data.get('error')}")
                else:
                    st.error("An unknown error occurred during review generation.")
            except Exception as e:
                logger.error(f"Exception in review pipeline (ui_utils): {e}", exc_info=True)
                st.error(f"An unexpected error occurred: {e}")
        st.session_state.running = False
        st.rerun()

# --- Form for Hypothesis Generation ---
def display_hypothesis_form():
    """
    Displays the Streamlit form for generating a research hypothesis and plan.

    This function handles user input for the research area, specific focus,
    goal, and context. It triggers the backend `run_hypothesis_pipeline` and
    manages the display of progress, success/error messages, and download links
    for the generated PDF and JSON outputs. It also checks for the required
    PubMed email environment variable.
    """
    # Imports necessary modules at function level.
    try:
        import config # Project configuration settings.
        from hypothesis_planning_module.hypothesis_pipeline import run_hypothesis_pipeline # Backend function to run the hypothesis pipeline.
        from utils.pdf_generator import save_hypothesis_pdf # Utility to save the hypothesis as a PDF.
    except ImportError as e:
        st.error(f"Error loading hypothesis modules: {e}. Please ensure `config.py`, `hypothesis_planning_module`, and `utils.pdf_generator` are accessible.")
        logger.critical(f"Failed to load hypothesis-related modules in display_hypothesis_form: {e}", exc_info=True)
        return

    st.subheader("💡 Generate Hypothesis and Research Plan")
    with st.form(key="hypothesis_form"):
        default_topic_hyp = "Glioblastoma multiforme (GBM)"
        research_topic = st.text_input("Research Area/Disease", value=default_topic_hyp)

        default_focus_hyp = "Metformin (SMILES: CN(C)C(=N)N=C(N)N) and metabolic reprogramming"
        user_focus = st.text_area("Specific Focus (molecule/SMILES, gene, pathway)", value=default_focus_hyp, height=100)

        default_goal_hyp = "propose novel hypothesis for repurposing"
        user_goal = st.text_input("Goal for the agent", value=default_goal_hyp)

        user_context = st.text_area("Specific constraints or context (optional)", height=100, placeholder="e.g., 'focus on in vivo', 'exclude epigenetic', 'check toxicity first'")

        submit_button = st.form_submit_button(label="Generate Hypothesis & Plan")

    if submit_button and not st.session_state.running:
        current_pubmed_email = os.getenv("MY_EMAIL_FOR_PUBMED")
        if not current_pubmed_email:
            st.error("MY_EMAIL_FOR_PUBMED is not set. This is required for hypothesis generation.")
            return

        st.session_state.running = True
        st.session_state.results = None
        st.session_state.output_pdf_path = None
        st.session_state.output_json_path = None
        with st.spinner("Generating hypothesis and research plan... This can take some time."):
            try:
                logger.info(f"Starting Hypothesis Pipeline with params: topic='{research_topic}', focus='{user_focus}', goal='{user_goal}', context='{user_context}'")
                final_data = run_hypothesis_pipeline(
                    user_topic=research_topic,
                    user_focus=user_focus,
                    user_goal=user_goal,
                    user_context=user_context if user_context.strip() else None,
                    user_email_for_pubmed=current_pubmed_email
                )
                st.session_state.results = final_data
                if final_data and "error" not in final_data:
                    st.success("Hypothesis and Research Plan generated successfully!")
                    response_text = final_data.get("final_agent_response")
                    if response_text and save_hypothesis_pdf:
                        pdf_filename = f"hypothesis_plan_{research_topic.replace(' ','_')[:20]}_{int(time.time())}.pdf"
                        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
                        pdf_filepath = os.path.join(config.OUTPUT_DIR, pdf_filename)
                        if save_hypothesis_pdf(response_text, pdf_filepath, title=f"Hypothesis & Plan: {research_topic}"):
                           st.session_state.output_pdf_path = pdf_filepath
                        else:
                            st.warning("Failed to save PDF for hypothesis.")

                    json_filename = f"final_hypothesis_output_{int(time.time())}.json"
                    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
                    json_filepath = os.path.join(config.OUTPUT_DIR, json_filename)
                    with open(json_filepath, "w", encoding='utf-8') as f:
                        json.dump(final_data, f, indent=4, ensure_ascii=False, default=str)
                    st.session_state.output_json_path = json_filepath

                elif final_data and "error" in final_data:
                    st.error(f"Error generating hypothesis: {final_data.get('error')}")
                else:
                    st.error("An unknown error occurred during hypothesis generation.")
            except Exception as e:
                logger.error(f"Exception in hypothesis pipeline (ui_utils): {e}", exc_info=True)
                st.error(f"An unexpected error occurred: {e}")
        st.session_state.running = False
        st.rerun()