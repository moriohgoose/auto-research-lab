import os
import google.generativeai as genai
from dotenv import load_dotenv
from utils import generate_pdf_report
from review_pipeline import run_review_pipeline
import re
import json
import logging
import time 

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S', 
    encoding='utf-8', 
    filename='review_generator.log', 
    filemode='w' 
)

# Load environment variables
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


# --- Gemini API Key ---
gemini_api_key = os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    logging.error("Error: GOOGLE_API_KEY environment variable not set.")
    exit()

genai.configure(api_key=gemini_api_key)
logging.info("Gemini API configured.")


# --- Agent Pipeline ---
if __name__ == '__main__':
    logging.info("Starting Literature Review Generation Process...")

    # --- User parameters ---
    print("-" * 30)
    print("Please provide the parameters for the literature review:")
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


    # --- Research start ---
    final_review_data = run_review_pipeline(
        research_topic=research_topic,
        specification_prompt=specification_prompt or None, 
        article_count_filter=article_count_filter,
        article_years_filter=article_years_filter,
        keywords_list=keywords_list,
        MAX_ITERATIONS=5
    )
    logging.info("Literature review pipeline execution finished.")

    # --- Pipeline result processing---
    if final_review_data and "error" not in final_review_data:
        logging.info("Pipeline completed successfully. Processing final results...")

        # --- Generate PDF ---
        final_output_pdf = "final_literature_review.pdf"

        if final_review_data.get("final_review_sections"):
            generate_pdf_report(
                title=final_review_data.get("title"),
                abstract=final_review_data.get("abstract"),
                review_sections=final_review_data.get("final_review_sections"),

                references=final_review_data.get("references_formatted", []),
                arxiv_ids_ordered=final_review_data.get("best_run_arxiv_ids", []),

                article_metadata=final_review_data.get("best_run_metadata", {}),
                filename=final_output_pdf
            )
        else:
            logging.warning("No best review sections available in pipeline result to generate PDF.")

        # --- Save final JSON ---
        final_output_json = "final_literature_review.json"
        try:
            with open(final_output_json, "w", encoding='utf-8') as f:
                json.dump(final_review_data, f, indent=4, ensure_ascii=False)
            logging.info(f"Final literature review data saved to {final_output_json}")
        
        except Exception as e:
            logging.error(f"\nError saving final output JSON: {e}", exc_info=True)

        final_text = final_review_data.get("final_literature_review_text")
        if final_text:
            logging.info("\n--- Final Literature Review Text (Best Run) ---")
            logging.info(final_text[:2000] + "..." if len(final_text) > 2000 else final_text)
        else:
             logging.info("\n--- No Final Literature Review Text Available ---")

    elif final_review_data and "error" in final_review_data:
         logging.error(f"Pipeline failed with error: {final_review_data.get('error')}")
    else:
         logging.error("Pipeline execution failed critically and returned no data.")

    logging.info("Application finished.")