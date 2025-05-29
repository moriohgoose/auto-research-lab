"""
This script serves as the main entry point for the Auto Research Lab's backend,
orchestrating the execution of either a literature review pipeline or a hypothesis
generation pipeline based on user input.

It handles environment setup, API key configuration, proxy settings, user interaction
for parameter collection, pipeline execution, and saving of results (JSON and PDF).
"""

import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv
import re
import json
import logging
import time 
from typing import Optional, List, Dict, Any

# --- Project Root Path Configuration ---
# This allows modules within the project (e.g., 'config', 'review_module') to be
# imported correctly, regardless of where the script is executed from
project_root_dir = os.path.dirname(os.path.abspath(__file__))
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

# --- Logging Configuration ---
# Sets up the global logging system for the application
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S', 
    encoding='utf-8', 
    handlers=[
        logging.StreamHandler()
    ]
)

# --- Module Imports ---
# Imports custom modules and functions necessary for the application's core logic.
# These imports are placed after the logging setup to ensure any issues are logged.
try:
    import config # Imports the configuration settings for the project (e.g., API model names, output directories).
    from review_module.review_pipeline import run_review_pipeline # Function to execute the literature review process.
    from hypothesis_planning_module.hypothesis_pipeline import run_hypothesis_pipeline # Function to execute the hypothesis generation process.
    from utils.pdf_generator import save_review_pdf, save_hypothesis_pdf # Functions for generating PDF reports from results.
except ImportError as e:
    logging.critical(f"Failed to import core modules. Please ensure all dependencies are installed and paths are correct: {e}", exc_info=True)
    sys.exit(1) # Exit if critical modules cannot be imported.

# --- Global Configuration Variables ---
PROJECT_ROOT_DIR = config.PROJECT_ROOT_DIR
GEMINI_API_MODEL_NAME = config.GEMINI_API_MODEL_NAME

# --- Load Environment Variables ---
load_dotenv()
logging.info("Loaded environment variables.")

# --- Proxy settings ---
proxy_host = os.getenv("PROXY_HOST")
proxy_port = os.getenv("PROXY_PORT")
proxy_user = os.getenv("PROXY_USER")
proxy_password = os.getenv("PROXY_PASSWORD")
use_proxy = all([proxy_host, proxy_port, proxy_user, proxy_password])

if use_proxy:
    proxies = {
        "http": f"http://{proxy_user}:{proxy_password}@{proxy_host}:{proxy_port}",
        "https": f"http://{proxy_user}:{proxy_password}@{proxy_host}:{proxy_port}", 
    }
    os.environ["HTTP_PROXY"] = proxies["http"]
    os.environ["HTTPS_PROXY"] = proxies["https"]
else:
    logging.info("Proxy environment variables not fully set. Proceeding without proxy.")
    # Unset potentially partially set proxy env vars
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)

# --- Configure Gemini API Key ---
gemini_api_key = os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    logging.error("Error: GOOGLE_API_KEY environment variable not set.")
    exit()

genai.configure(api_key=gemini_api_key)
logging.info("Gemini API configured.")

# --- Configure Hugging Face Token ---
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    logging.warning("HF_TOKEN environment variable not set. Downloading gated Hugging Face models might fail.")
else:
    logging.info("HF_TOKEN found.")
    os.environ["HF_TOKEN"] = hf_token 

# --- Extract Pubmed email ---
pubmed_email = os.getenv("MY_EMAIL_FOR_PUBMED")
if not pubmed_email:
    logging.warning("MY_EMAIL_FOR_PUBMED environment variable not set.")
else:
    logging.info("MY_EMAIL_FOR_PUBMED found.")
    os.environ["MY_EMAIL_FOR_PUBMED"] = pubmed_email 

# --- Main Execution Block ---
if __name__ == '__main__':
    
    print("=" * 50)
    print("Agent Runner: Literature Review OR Hypothesis Generation")
    print("=" * 50)

    # --- Mode Selection ---
    # Prompts the user to select an operational mode: Literature Review or Hypothesis Generation
    while True:
        print("\nPlease choose the execution mode:")
        print("  1: Generate Literature Review")
        print("  2: Generate Hypothesis and Research Plan")
        mode_choice = input("Enter your choice (1 or 2): ")
        if mode_choice == '1':
            selected_mode = "review"
            # Basic check to ensure the review pipeline function was imported successfully
            if run_review_pipeline is None:
                 print("Error: Literature review module is not available due to import errors. Cannot proceed.")
                 exit()
            logging.info("Mode selected: Generate Literature Review")
            break
        elif mode_choice == '2':
            selected_mode = "hypothesis"
            # Basic check to ensure the hypothesis pipeline function was imported successfully
            if run_hypothesis_pipeline is None:
                 print("Error: Hypothesis generation module is not available due to import errors. Cannot proceed.")
                 exit()
            logging.info("Mode selected: Generate Hypothesis and Research Plan")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    print("-" * 50)

    # --- Parameter Collection (Common and Mode-Specific) ---
    # Initializes variables to store user-provided parameters for the chosen mode
    research_topic = ""
    specification_prompt = None
    article_count_filter = 10 # Default for review
    article_years_filter = None # Default for review
    keywords_list = None # Default for review
    user_focus = None # For hypothesis
    user_goal = None # For hypothesis
    user_context = None # For hypothesis

    if selected_mode == "review":
        logging.info("Collecting parameters for Literature Review...")
        print("\nPlease provide parameters for the Literature Review:")
        print("-" * 30)

        # 1. Research Topic
        default_topic = "LLMs in Drug Research" 
        research_topic_input = input(f"Enter the Research Topic (default: '{default_topic}'): ")
        research_topic = research_topic_input.strip() if research_topic_input.strip() else default_topic
        logging.info(f"Using Research Topic: '{research_topic}'")

        # 2. Specification Prompt 
        default_spec = "Focus on novel methods, evaluation metrics, and ethical considerations."
        specification_prompt_input = input(f"Enter a Specification Prompt (optional, press Enter to skip or use default: '{default_spec}'): ")
        specification_prompt = specification_prompt_input.strip() if specification_prompt_input.strip() else default_spec
        if specification_prompt:
            logging.info(f"Using Specification Prompt: '{specification_prompt}'")
        else:
            specification_prompt = None 
            logging.info("No Specification Prompt provided.")

        # 3. Article Count Filter
        default_count = 10
        while True:
            article_count_input = input(f"Enter the Number of Articles to include (default: {default_count}): ")
            
            if not article_count_input.strip():
                article_count_filter = default_count
                logging.info(f"Using default Article Count: {article_count_filter}")
                break
            
            try:
                article_count_filter = int(article_count_input)
                if article_count_filter > 0:
                    logging.info(f"Using Article Count: {article_count_filter}")
                    break
                else:
                    print("Please enter a positive integer for the number of articles.")
            
            except ValueError:
                print("Invalid input. Please enter an integer.")
        
        # 4. Article Years Filter 
        default_years = "2020-2025"
        year_pattern = re.compile(r"^\d{4}-\d{4}$") 
        
        while True:
            article_years_input = input(f"Enter the Article Years Filter (YYYY-YYYY, optional, press Enter to skip or use default: '{default_years}'): ")
            article_years_filter_str = article_years_input.strip()
            
            if not article_years_filter_str:
                article_years_filter = default_years
                logging.info(f"Using default Article Years Filter: '{article_years_filter}'")
                break
            
            elif article_years_filter_str.lower() == 'skip':
                article_years_filter = None
                logging.info("Skipping Article Years Filter.")
                break
            
            elif year_pattern.match(article_years_filter_str):
                # Check: start date <= end date
                try:
                    start_y, end_y = map(int, article_years_filter_str.split('-'))
                    if start_y <= end_y:
                        article_years_filter = article_years_filter_str
                        logging.info(f"Using Article Years Filter: '{article_years_filter}'")
                        break
                    else:
                        print("Start year must be less than or equal to end year.")
                except ValueError: 
                    print("Invalid year format within the range.")
            else:
                print("Invalid format. Please use YYYY-YYYY, type 'skip', or press Enter for default.")

        # 5. Keywords 
        keywords_list = None 
        keywords_input = input("Enter keywords separated by commas (optional, press Enter to auto-generate): ")
        keywords_raw = [k.strip() for k in keywords_input.split(',') if k.strip()] 

        if keywords_raw: # User's keywords
            keywords_list = keywords_raw
            logging.info(f"Using manually provided keywords: {keywords_list}")

        else:
            logging.info("No manual keywords provided. Keywords will be generated automatically.")

        print("-" * 30)
        logging.info("Initializing ResearchAgent with user-provided parameters...")

    elif selected_mode == "hypothesis":
        logging.info("Collecting parameters for Hypothesis Generation...")
        print("\nPlease provide information for hypothesis generation:")
        print("-" * 30)
        # Collect hypothesis-specific parameters
        default_topic_hyp = "Glioblastoma multiforme (GBM)"
        user_topic_input = input(f"1. Research Area/Disease (default: '{default_topic_hyp}'): ")
        research_topic = user_topic_input.strip() if user_topic_input.strip() else default_topic_hyp # Use same var for consistency

        default_focus_hyp = "Metformin (SMILES: CN(C)C(=N)N=C(N)N) and metabolic reprogramming"
        user_focus_input = input(f"2. Specific Focus (molecule/SMILES, gene, pathway) (optional, default: '{default_focus_hyp}'): ")
        user_focus = user_focus_input.strip() if user_focus_input.strip() else default_focus_hyp

        print("\n3. What is the goal for the agent?")
        print("   (e.g., 'propose novel hypothesis', 'develop plan for existing hypothesis', 'find repurposing candidate')")
        default_goal_hyp = "propose novel hypothesis for repurposing"
        user_goal_input = input(f"   Goal (default: '{default_goal_hyp}'): ")
        user_goal = user_goal_input.strip() if user_goal_input.strip() else default_goal_hyp

        print("\n4. Any specific constraints or context?")
        print("   (e.g., 'focus on in vivo', 'exclude epigenetic', 'check toxicity first')")
        user_context_input = input("   Constraints/Context (optional, press Enter to skip): ")
        user_context = user_context_input.strip() if user_context_input.strip() else None
        print("-" * 30)

    # --- Pipeline Execution ---
    # Records the start time to calculate total execution duration
    script_start_time = time.time()
    final_data: Optional[Dict[str, Any]] = None # Stores the final results dictionary from the pipeline
    final_pdf_path: Optional[str] = None # Stores the path to the generated PDF report

    if selected_mode == "review":
        logging.info("Starting Literature Review Pipeline...")
        # Calls the literature review pipeline function with the collected parameters
        final_data = run_review_pipeline(
            research_topic=research_topic,
            specification_prompt=specification_prompt,
            article_count_filter=article_count_filter,
            article_years_filter=article_years_filter,
            keywords_list=keywords_list,
            max_iterations=getattr(config, 'REVIEW_MAX_ITERATIONS', 5), # Get from config or default
            gemini_model_name=config.GEMINI_API_MODEL_NAME # Pass model name
        )
        logging.info("Literature Review Pipeline finished.")
        
        # --- Post-processing for Literature Review PDF ---
        # If the pipeline ran successfully and returned data, attempts to save a PDF report
        if final_data and final_data.get("error") is None:
            # Save PDF review
            if save_review_pdf and final_data.get("final_review_sections"):
                pdf_filename_review = f"literature_review_{research_topic.replace(' ','_')[:20]}_{int(time.time())}.pdf"
                pdf_filepath_review = os.path.join(config.OUTPUT_DIR, pdf_filename_review)
                try:
                    # Calls the PDF generation utility function
                    review_pdf_saved = save_review_pdf(
                        title=final_data.get("title"),
                        abstract=final_data.get("abstract"),
                        review_sections=final_data.get("final_review_sections"),
                        references=final_data.get("references_formatted", []),
                        arxiv_ids_ordered=final_data.get("best_run_arxiv_ids", []),
                        article_metadata=final_data.get("best_run_metadata", {}),
                        filename=pdf_filepath_review
                    )
                    if review_pdf_saved:
                        final_pdf_path = pdf_filepath_review # Save path to PDF
                        logging.info(f"Literature review PDF saved to: {final_pdf_path}")
                    else:
                        logging.warning("Literature review PDF generation function returned False.")
                except NameError:
                    logging.error("Function 'save_review_pdf' (aliased from generate_pdf_report) is not defined or imported from review_module.review_utils.")
                except Exception as e_pdf:
                    logging.error(f"Error saving literature review PDF: {e_pdf}", exc_info=True)
            else:
                logging.warning("No review sections found or PDF generation function for reviews is not available.")

    elif selected_mode == "hypothesis":
        logging.info("Starting Hypothesis Generation Pipeline...")
        # Get email for PubMed 
        user_email = pubmed_email

        # Calls the hypothesis generation pipeline function with the collected parameters
        final_data = run_hypothesis_pipeline(
            user_topic=research_topic,
            user_focus=user_focus,
            user_goal=user_goal,
            user_context=user_context,
            user_email_for_pubmed=user_email
        )
        logging.info("Hypothesis Generation Pipeline finished.")

        # --- Post-processing for Hypothesis PDF ---
        # If the pipeline ran successfully and returned data, attempts to save a PDF report
        if final_data and "error" not in final_data and save_hypothesis_pdf:
            response_text = final_data.get("final_agent_response")
            if response_text:
                pdf_filename = f"hypothesis_plan_{research_topic.replace(' ','_')[:20]}_{int(time.time())}.pdf"
                pdf_filepath = os.path.join(getattr(config, 'OUTPUT_DIR', '.'), pdf_filename) # Use output dir from config
                os.makedirs(os.path.dirname(pdf_filepath), exist_ok=True)
                
                # Calls the PDF generation utility function for hypothesis
                try:
                    hyp_pdf_saved = save_hypothesis_pdf(response_text, pdf_filepath, title=f"Hypothesis & Plan: {research_topic}")
                    if hyp_pdf_saved:
                        final_pdf_path = pdf_filepath # Save PDF path
                        logging.info(f"Hypothesis/plan PDF saved to: {final_pdf_path}")
                    else:
                        logging.warning("Hypothesis/plan PDF generation function returned False.")
                except NameError:
                    logging.error("Function 'save_hypothesis_pdf' (aliased from save_agent_response_to_pdf) is not defined or imported from utils.pdf_generator.")
                except Exception as e_pdf:
                    logging.error(f"Error saving hypothesis/plan PDF: {e_pdf}", exc_info=True)
            elif not response_text:
                logging.warning("No agent response text found to generate hypothesis PDF.")
            elif not save_hypothesis_pdf:
                logging.warning("PDF generation function for hypotheses is not available.")                

    # --- Final Output Summary ---
    print("-" * 50)
    logging.info("Processing final results...")
    if final_data and final_data.get("error") is None:
        logging.info("Pipeline completed successfully.")
        print("\n--- Final Agent Output ---")

        # Prints a snippet of the generated text (either review or hypothesis response)
        final_text_key = "final_literature_review_text" if selected_mode == "review" else "final_agent_response"
        final_text = final_data.get(final_text_key)
        if final_text:
            print(final_text[:2000] + "..." if len(final_text) > 2000 else final_text) # Print snippet
        else:
            print("(No final text generated)")

        # Saves the complete final data dictionary to a JSON file.
        # This includes all intermediate steps and final results, useful for debugging or further processing.
        json_filename = f"final_{selected_mode}_output_{int(time.time())}.json"
        json_filepath = os.path.join(getattr(config, 'OUTPUT_DIR', '.'), json_filename) # Save JSON near PDF
        try:
            with open(json_filepath, "w", encoding='utf-8') as f:
                # Convert potential complex objects to string if needed, or handle serialization properly
                json.dump(final_data, f, indent=4, ensure_ascii=False, default=str)
            logging.info(f"Final data saved to JSON: {json_filepath}")
        except Exception as e:
            logging.error(f"Error saving final output JSON: {e}", exc_info=True)

    elif final_data and final_data.get("error"):
        error_msg = final_data.get('error', 'Unknown error')
        logging.error(f"Pipeline failed with error: {error_msg}")
        print(f"\nPipeline execution failed: {error_msg}")
    else:
         # Catches cases where the pipeline returned no data at all (e.g., very early failure)        
        logging.error("Pipeline execution failed critically and returned no data.")
        print("\nPipeline execution failed critically.")

    logging.info("Application finished.")
    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time
    logging.info(f"--- TOTAL SCRIPT TIME: {total_script_time:.2f} SECONDS ---")
    print(f"\nTotal script execution time: {total_script_time:.2f} seconds")