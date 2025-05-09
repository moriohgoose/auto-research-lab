import os
import json
import re
import time
import io
import logging
import requests
import arxiv
from datetime import datetime
from typing import Optional, List, Dict
import google.generativeai as genai
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

from utils.llm_clients import generate_text 
from prompts import review_prompts as rp

MODULE_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_OUTPUT_DIR = os.path.join(MODULE_ROOT_DIR, "pipeline_run_outputs")

# --- ResearchAgent ---
class ResearchAgent:
    """
    A research agent that conducts literature reviews using the Gemini model and arXiv API.
    Gathers article metadata, ranks articles, downloads PDFs, extracts text,
    generates individual article reviews, synthesizes these into a literature review,
    and can rewrite the review based on critic feedback.
    Also generates title and abstract for the final review.
    """

    def __init__(
        self,
        research_topic: str = "How to use LLMs for drug research",
        keywords: Optional[List[str]] = None,
        specification_prompt: Optional[str] = None,
        article_count_filter: int = 5,
        article_years_filter: Optional[str] = None,
        gemini_model_name: str = "gemini-2.0-flash"
    ):
        """
        Initializes the ResearchAgent.

        Args:
            research_topic: The topic of the literature review.
            keywords: Optional list of keywords to use for the arXiv search.
            specification_prompt: Optional prompt for specifying the research type or recommendations.
            article_count_filter: The number of articles to include in the final literature review (after ranking).
            article_years_filter: The number of years range to filter the article search by (e.g., "2020-2025"). None for no filter.
            gemini_model_name: The name of the Gemini model to use (e.g., "gemini-2.0-flash").
    
        """
        self.research_topic = research_topic
        self.keywords = keywords
        self.specification_prompt = specification_prompt
        self.article_count_filter = article_count_filter
        self.article_years_filter = article_years_filter
        self.article_metadata = {} # Stores metadata like title, authors, year, summary, urls
        self.ranked_arxiv_ids = [] # List of ArXiv IDs (URL format) in ranked order
        self.relevant_arxiv_ids = [] # List of ArXiv IDs selected after ranking/filtering
        self.pdf_texts = {} # Stores extracted text {arxiv_id: text}
        self.review_sections = {} # Stores generated review sections {section_key: text}
        self.article_reviews = {} # Stores individual article reviews {arxiv_id: review_text}
        self.literature_review = None # Stores the final consolidated literature review text
        self.run_count = 0 # Counter for number of runs
        self.gemini_model_name = gemini_model_name

        logging.info(f"ResearchAgent initialized for topic: {self.research_topic}")

    def generate_keywords(self):
        """
        Generates keywords for the research topic using the Gemini model if keywords are not provided.
        """
        if not self.keywords:
            prompt = rp.GENERATE_KEYWORDS_PROMPT_TEMPLATE.format(research_topic=self.research_topic)
            
            logging.info("Generating keywords...")
            response = generate_text(prompt=prompt, model_name=self.gemini_model_name)

            if response:
                self.keywords = [k.strip() for k in response.split(',') if k.strip()]
            else:
                logging.warning("Failed to generate keywords. Using default keywords.")
                self.keywords = ["LLMs", "drug research", "AI", "cheminformatics"] # Default fallback

        logging.info(f"Using keywords: {self.keywords}")


    def search_arxiv(self):
        """
        Searches arXiv for articles using keywords and year filter.
        Stores detailed metadata for found articles in self.article_metadata.
        """
        self.article_metadata.clear() # Clear previous results
        start_year, end_year = None, None

        if self.article_years_filter:
            try:
                start_year, end_year = map(int, self.article_years_filter.split('-'))
            except ValueError:
                logging.warning("Invalid year filter format. Ignoring year filter.")

        keyword_query = " OR ".join(self.keywords)
        logging.info(f"Searching ArXiv for query: {keyword_query} (Years: {self.article_years_filter or 'Any'})")

        results_found_count = 0
        try:
            search = arxiv.Search(
                query=keyword_query,
                max_results=100, # Adjust max results if needed
                sort_by=arxiv.SortCriterion.Relevance,
            )

            for result in search.results():
                results_found_count += 1
                publish_year = result.published.year
                # Apply year filter
                if start_year and end_year:
                    if not (start_year <= publish_year <= end_year):
                        continue # Skip if outside year range

                # Store metadata using the standard ArXiv ID URL format as key
                arxiv_id_key = result.entry_id # e.g., 'http://arxiv.org/abs/...'
                metadata = {
                    'title': result.title.replace('\n', ' ').strip() if result.title else "No Title Provided",
                    'authors': [str(a) for a in result.authors],
                    'published_date': result.published.strftime('%Y-%m-%d'),
                    'year': result.published.year,
                    'summary': result.summary or "No summary available.",
                    'journal_ref': result.journal_ref,
                    'doi': result.doi,
                    'primary_category': result.primary_category,
                    'pdf_url': result.pdf_url,
                    'arxiv_url': arxiv_id_key # Store the ID/URL itself
                }
                self.article_metadata[arxiv_id_key] = metadata

            logging.info(f"ArXiv search yielded {results_found_count} initial results. Stored metadata for {len(self.article_metadata)} articles matching criteria.")

        except Exception as e:
            logging.error(f"Error during ArXiv search: {e}", exc_info=True)

        # Final check if any articles were actually stored
        if not self.article_metadata:
            logging.warning("No articles matching the criteria were stored after processing ArXiv search results.")


    def rank_abstracts(self):
        """
        Ranks article abstracts based on relevance using Gemini.
        Uses summaries from self.article_metadata. Updates self.ranked_arxiv_ids.
        """
        if not self.article_metadata:
            logging.warning("No abstracts available to rank.")
            self.ranked_arxiv_ids = []
            self.relevant_arxiv_ids = []
            return

        logging.info(f"Ranking {len(self.article_metadata)} abstracts...")
        logging.debug(f"Metadata keys available for ranking: {list(self.article_metadata.keys())}")

        prompt_header = rp.RANK_ABSTRACTS_PROMPT_HEADER_TEMPLATE.format(research_topic=self.research_topic)
        specification_line = ""
        if self.specification_prompt:
            specification_line = rp.RANK_ABSTRACTS_SPECIFICATION_LINE_TEMPLATE.format(specification_prompt=self.specification_prompt)

        abstract_texts_for_prompt = []
        for arxiv_id_key, metadata in self.article_metadata.items():
            abstract = metadata.get('summary', 'No summary available.')
            truncated_abstract = abstract[:500] + "..." if len(abstract) > 500 else abstract
            abstract_texts_for_prompt.append(f"ID: {arxiv_id_key}\nAbstract: {truncated_abstract}\n---")

        ranking_prompt = (
            f"{prompt_header}{specification_line}"
            f"{rp.RANK_ABSTRACTS_INSTRUCTIONS.format(abstract_list_for_prompt= '\n'.join(abstract_texts_for_prompt))}"
        )
        logging.debug(f"Ranking prompt (first 500 chars): {ranking_prompt[:500]}")

        ranking_response = generate_text(prompt=ranking_prompt, model_name=self.gemini_model_name)

        self.ranked_arxiv_ids = [] # Reset ranked list
        if ranking_response:
            logging.debug(f"LLM Ranking Response:\n{ranking_response}\n--------------------")

            # Extract URLs matching the expected format (http://arxiv.org/abs/...)
            found_ids_urls = re.findall(r"http://arxiv\.org/abs/[\w./-]+v?\d*", ranking_response)
            logging.info(f"Found {len(found_ids_urls)} potential ArXiv ABS URLs in LLM response.")
            logging.debug(f"Extracted URLs: {found_ids_urls}")

            if found_ids_urls:
                valid_ranked_ids = []
                seen_ids = set()
                for url in found_ids_urls:
                    arxiv_id_key = url # The URL is the key format we use

                    # Check if this ID exists in our collected metadata and is unique in the ranked list
                    if arxiv_id_key in self.article_metadata:
                        if arxiv_id_key not in seen_ids:
                            valid_ranked_ids.append(arxiv_id_key)
                            seen_ids.add(arxiv_id_key)
                    else:
                         # Log if the LLM returns an ID we didn't find initially
                         logging.warning(f"Ranked URL '{arxiv_id_key}' from LLM not found in initial metadata keys.")

                self.ranked_arxiv_ids = valid_ranked_ids
                logging.info(f"Processed LLM response, found {len(valid_ranked_ids)} valid and unique ranked IDs matching metadata.")
                logging.debug(f"Final ranked IDs (up to 10): {self.ranked_arxiv_ids[:10]}")

                if not self.ranked_arxiv_ids:
                     logging.warning("LLM ranking response parsed, but no valid/matching IDs found after processing.")

            else: # Regex found no matches
                logging.warning("LLM ranking response did not contain recognizable ArXiv ABS URLs.")

        else: # LLM call failed
            logging.warning("LLM API call failed for ranking. Cannot rank articles.")


        # --- Select relevant articles ---
        # If ranking failed or yielded no results, use the original order from metadata
        if not self.ranked_arxiv_ids:
             logging.warning("Ranking failed or yielded no results. Selecting articles based on original ArXiv order.")
             fallback_ids = list(self.article_metadata.keys()) # Get IDs from metadata
             self.relevant_arxiv_ids = fallback_ids[:self.article_count_filter] # Select top N from this list
        else:
            # Select top N from the successfully ranked list
            self.relevant_arxiv_ids = self.ranked_arxiv_ids[:self.article_count_filter]

        logging.info(f"Selected top {len(self.relevant_arxiv_ids)} articles for review based on ranking/filtering.")

        # Warning if fewer articles were selected than requested
        if len(self.relevant_arxiv_ids) < self.article_count_filter and self.article_metadata:
             logging.warning(f"Fewer articles selected ({len(self.relevant_arxiv_ids)}) than requested ({self.article_count_filter}).")
        elif not self.article_metadata:
             logging.info("No articles were available to select from.")


    def download_relevant_pdfs(self):
        """
        Downloads PDFs for the relevant articles (self.relevant_arxiv_ids).
        Skips download if the file already exists locally.
        """
        if not self.relevant_arxiv_ids:
            logging.warning("No relevant articles selected to download PDFs for.")
            return

        logging.info(f"Checking/Downloading PDFs for {len(self.relevant_arxiv_ids)} relevant articles...")
        pdf_dir = "downloaded_pdfs"
        os.makedirs(pdf_dir, exist_ok=True) 

        downloaded_count = 0
        present_count = 0
        failed_count = 0

        for arxiv_id_key in self.relevant_arxiv_ids:

            # Get PDF URL from stored metadata
            metadata = self.article_metadata.get(arxiv_id_key)
            if not metadata or not metadata.get('pdf_url'):
                logging.warning(f"No metadata or PDF URL found for {arxiv_id_key}. Skipping download.")
                failed_count += 1
                continue
            pdf_url = metadata['pdf_url']

            # Create safe filename
            safe_filename_id = re.sub(r'[/:.]', '_', arxiv_id_key)
            pdf_filename = os.path.join(pdf_dir, f"{safe_filename_id}.pdf")

            # Check if file already exists
            if os.path.exists(pdf_filename):
                present_count += 1
                continue # Skip download

            # Attempt download if file doesn't exist
            logging.info(f"Downloading PDF for {arxiv_id_key} from {pdf_url}")
            try:
                response = requests.get(pdf_url, stream=True, timeout=30)
                response.raise_for_status() # Check for HTTP errors

                # Write file content chunk by chunk
                with open(pdf_filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logging.info(f"  PDF saved as {pdf_filename}")
                downloaded_count += 1
                time.sleep(1) 

            except requests.exceptions.RequestException as e:
                logging.error(f"  Error downloading PDF for {arxiv_id_key}: {e}")
                failed_count += 1
                # Clean up partially downloaded file if it exists
                if os.path.exists(pdf_filename):
                    try: os.remove(pdf_filename)
                    except OSError: pass
                continue # Continue to next article

        logging.info(f"Finished PDF check/download. Found existing: {present_count}, Newly downloaded: {downloaded_count}, Failed: {failed_count}.")


    def extract_full_text_from_pdf(self, pdf_file_path: str) -> Optional[str]:
        """
        Extracts the full text from a single PDF file using pdfminer.six.
        """
        if not os.path.exists(pdf_file_path):
            logging.warning(f"PDF file not found for extraction: {pdf_file_path}")
            return None

        logging.info(f"Extracting text from: {pdf_file_path}")

        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        laparams = LAParams() # Use default LAParams
        converter = TextConverter(resource_manager, fake_file_handle, laparams=laparams)
        page_interpreter = PDFPageInterpreter(resource_manager, converter)
        text = None

        try:
            with open(pdf_file_path, 'rb') as fh:
                # Process pages, skipping problematic ones
                for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                    try:
                        page_interpreter.process_page(page)

                    except Exception as page_e:
                        logging.warning(f"Error processing a page in {pdf_file_path}: {page_e}")
                        continue # Skip page

                text = fake_file_handle.getvalue()

        except Exception as file_e:
            logging.error(f"Error opening or reading PDF {pdf_file_path}: {file_e}", exc_info=True)
            text = None # Ensure text is None on error
        
        finally:
            # Ensure resources are closed
            converter.close()
            fake_file_handle.close()

        if text and text.strip(): # Check if text is not empty or just whitespace
            logging.info(f"Successfully extracted text (approx. {len(text)} chars) from {pdf_file_path}.")
            # Basic cleaning: replace multiple whitespaces with single space
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        else:
            logging.warning(f"Failed to extract text or extracted empty text from {pdf_file_path}.")
            return None # Return None if extraction failed or text is empty


    def generate_article_reviews(self, pdf_texts_content: Dict[str, str]) -> Dict[str, str]:
        """
        Generates detailed reviews for each article based on its full text.
        Skips articles with missing or invalid text content.
        """
        self.article_reviews.clear() # Clear previous reviews
        if not pdf_texts_content:
             logging.warning("No PDF texts provided to generate article reviews.")
             return {}

        logging.info(f"Generating detailed article reviews for {len(pdf_texts_content)} provided texts...")
        successful_reviews_count = 0
        for arxiv_id, full_text in pdf_texts_content.items():
            logging.info(f"Processing article for review generation: {arxiv_id}")

            text_for_prompt = full_text

            # Validate text content before proceeding
            if not text_for_prompt or not isinstance(text_for_prompt, str) or text_for_prompt.strip().startswith("Error:"):
                logging.warning(f"  Skipping review generation for {arxiv_id} due to missing, invalid, or error-marked text content.")
                self.article_reviews[arxiv_id] = f"Error: Cannot generate review for {arxiv_id} - text unavailable/invalid."
                continue # Skip to the next article

            # --- Prompt for individual article review ---
            review_prompt = rp.SINGLE_ARTICLE_REVIEW_PROMPT_TEMPLATE.format(
                system_prompt=rp.RESEARCH_AGENT_SYSTEM_PROMPT,
                arxiv_id=arxiv_id,
                research_topic=self.research_topic,
                specification_prompt=self.specification_prompt or 'General Relevance',
                text_for_prompt=text_for_prompt
            )

            logging.info(f"  Sending prompt ({len(review_prompt)} chars) to Gemini for review of {arxiv_id}...")
            review_text = generate_text(prompt=review_prompt)

            if review_text:
                self.article_reviews[arxiv_id] = review_text
                successful_reviews_count += 1
                logging.info(f"  Review generated successfully for {arxiv_id}.")
            else:
                self.article_reviews[arxiv_id] = f"Error: Failed to generate review for {arxiv_id} (LLM error)."
                logging.error(f"  Failed to generate review for article: {arxiv_id} (LLM error or empty response).")          
            time.sleep(1) # Optional delay

        logging.info(f"Finished generating article reviews. Successfully generated {successful_reviews_count}/{len(pdf_texts_content)} reviews.")
        if successful_reviews_count < len(pdf_texts_content):
             # Log which ones failed or were skipped
             failed_or_skipped = [id for id, rev in self.article_reviews.items() if rev.startswith("Error:")]
             logging.warning(f"  Failed/Skipped review generation for IDs: {failed_or_skipped}")

        return self.article_reviews


    def generate_review_sections(self, article_reviews_content: Dict[str, str], critic_feedback: Optional[str] = None):
        """
        Generates literature review sections individually.
        Uses separated sections and instructs LLM on citation format [ID: number].
        Instructs LLM *not* to include markdown headings in the response body.
        """
        self.review_sections.clear()
        valid_reviews = {id: review for id, review in article_reviews_content.items() if not review.startswith("Error:")}
        if not valid_reviews:
             logging.warning("No valid article reviews provided to generate literature review sections.")
             return

        logging.info(f"Generating literature review sections using {len(valid_reviews)} valid article reviews...")

        # Prepare article review content for the prompt
        article_reviews_text_for_prompt = []
        for arxiv_id_key, review in valid_reviews.items():
            arxiv_id_number = arxiv_id_key.split('/abs/')[-1]

            truncated_review = review[:2000] + "... (review truncated)" if len(review) > 2000 else review
            article_reviews_text_for_prompt.append(f"--- Article Review [ID: {arxiv_id_number}] ---\n{truncated_review}\n")
        article_reviews_full_prompt_text = "\n".join(article_reviews_text_for_prompt)

        # Base role and instructions for LLM
        base_system_role = rp.LIT_REVIEW_SECTION_BASE_SYSTEM_ROLE_TEMPLATE.format(
            research_topic=self.research_topic,
            specification_prompt=self.specification_prompt or "General overview"
        )

        # Prepare critic feedback prompt part
        critic_instructions_prompt = ""
        if critic_feedback:
            logging.info("  Generating sections in rewrite mode based on critic feedback.")
            critic_instructions_prompt = rp.CRITICAL_REWRITE_INSTRUCTIONS_TEMPLATE.format(
                critic_feedback=critic_feedback
            )

        successful_sections_count = 0
        
        # Generate sections sequentially
        for section_key, section_config in rp.LIT_REVIEW_SECTIONS_CONFIG.items():
            logging.info(f"  Generating section: {section_key}...")

            template_str = section_config["template"]
            prev_keys = section_config["previous_section_keys"] 

            previous_sections_text = []
            if prev_keys: 
                for prev_key in prev_keys:
                    content = self.review_sections.get(prev_key, 'Not generated yet or generation failed.')
                    previous_sections_text.append(f"--- Context from previously generated section '{prev_key}' (excerpt) ---\n{content[:500]}...\n--- End of context for '{prev_key}' ---")
            
            previous_sections_for_prompt = "\n\n".join(previous_sections_text) if previous_sections_text else "This is the first section being generated."
            
            prompt = template_str.format(
                system_role=base_system_role,
                research_topic=self.research_topic, 
                specification_prompt=self.specification_prompt or "General overview", 
                article_reviews=article_reviews_full_prompt_text,
                previous_sections=previous_sections_for_prompt, 
                critic_instructions=critic_instructions_prompt 
            )
            section_content = generate_text(prompt=prompt, model_name=self.gemini_model_name)
            
            if section_content:
                section_content = re.sub(r"^\s*#+\s*.*\s*", "", section_content).strip()
                section_content = re.sub(r"^\s*\*\*+\s*.*\s*\*\*+\s*", "", section_content).strip()
                self.review_sections[section_key] = section_content
                successful_sections_count += 1
                logging.info(f"    Successfully generated section: {section_key}")
            else:
                self.review_sections[section_key] = f"Error: Failed to generate section '{section_key}'."
                logging.error(f"    Error generating section: {section_key}")
        
        total_sections_to_generate = len(rp.LIT_REVIEW_SECTIONS_CONFIG)
        logging.info(f"Finished generating review sections. Successfully generated {successful_sections_count}/{total_sections_to_generate} sections.")
        if successful_sections_count < total_sections_to_generate:
             failed_sections = [key for key, content in self.review_sections.items() if content.startswith("Error:")]
             logging.warning(f"  Failed to generate sections: {failed_sections}")


    def consolidate_reviews(self):
        """
        Combines generated sections into the final literature review text (Markdown format).
        Uses the new section structure. References section is just a placeholder title.
        """
        logging.info("Consolidating review sections into text format...")
        literature_review_parts = []

        # New section order reflecting the split
        section_order = [
             ("Introduction", "introduction"),
             ("Key Themes and Trends", "key_themes"),
             ("Methods and Technologies", "methods_technologies"),
             ("Resources and Datasets", "resources_datasets"),
             ("Challenges and Limitations", "challenges_limitations"),
             ("Proposed Research Pipeline", "proposed_pipeline"),
             ("Conclusion", "conclusion")
        ]

        for title, key in section_order:
             content = self.review_sections.get(key)
             # Add Markdown H2 heading for structure in the text file
             literature_review_parts.append(f"## {title}\n")
             if content and not content.startswith("Error:"):
                 literature_review_parts.append(content.strip())
             else:
                 literature_review_parts.append("*Section generation failed or skipped.*")
             literature_review_parts.append("\n") # Add a newline after the content

        # Remove the trailing '---' separator if it was added before
        self.literature_review = "\n".join(literature_review_parts).strip()
        logging.info("Literature review sections consolidated into text format.")


    def generate_title_abstract(self, review_text: str) -> tuple[Optional[str], Optional[str]]:
        """
        Generates a title and abstract for the given literature review text using Gemini.
        """
        logging.info("Generating title and abstract for the final review...")

        if not review_text or review_text.strip() == "":
            logging.error("  Error: Cannot generate title/abstract from empty review text.")
            return None, None

        max_len = 200000 
        truncated_review = review_text

        if len(review_text) > max_len:
             truncated_review = review_text[:max_len] + "\n... (review text truncated for title/abstract prompt)"
             logging.warning(f"  Review text truncated to {max_len} chars for title/abstract generation.")

        prompt = rp.TITLE_ABSTRACT_PROMPT_TEMPLATE.format(truncated_review=truncated_review)
        
        response = generate_text(prompt, model_name=self.gemini_model_name)
        if not response:
            logging.error("  Error: Failed to get response from Gemini for title/abstract.")
            return None, None

        title = None
        abstract = None
        try:
            # Extract Title and Abstract using regex, case-insensitive
            title_match = re.search(r"^Title:(.*)$", response, re.MULTILINE | re.IGNORECASE)
            abstract_match = re.search(r"^Abstract:(.*)$", response, re.MULTILINE | re.DOTALL | re.IGNORECASE)

            if title_match:
                title = title_match.group(1).strip()
            else:
                 logging.warning("  Could not parse 'Title:' from Gemini response.")

            if abstract_match:
                abstract = abstract_match.group(1).strip()
                 # Clean up potential extra newlines in abstract
                abstract = re.sub(r'\s+', ' ', abstract).strip()
            else:
                 logging.warning("  Could not parse 'Abstract:' from Gemini response.")

            logging.info(f"  Generated Title: {title}")
            logging.info(f"  Generated Abstract: {abstract[:100]}...")
            return title, abstract

        except Exception as e:
            logging.error(f"  Error parsing title/abstract from Gemini response: {e}", exc_info=True)
            logging.debug(f"  Raw response for title/abstract: {response}")
            return None, None


    def run(self, critic_feedback_and_weaknesses: Optional[str] = None, force_rewrite_from_scratch: bool = False) -> Optional[str]:
        """
        Main execution method for the ResearchAgent.
        Coordinates keyword generation, search, ranking, PDF handling, review generation, and consolidation.
        """
        self.run_count += 1
        logging.info(f"\n--- ResearchAgent Run {self.run_count} ---")

        is_rewrite_mode = bool(critic_feedback_and_weaknesses) and not force_rewrite_from_scratch
        is_initial_run = self.run_count == 1
        current_mode = "Initial" 

        if force_rewrite_from_scratch:
            current_mode = "RewriteFromScratch"
            logging.info("Mode: Forcing Rewrite from Scratch (ignoring previous critic feedback)")
            critic_feedback_to_pass = None

        elif is_rewrite_mode:
            current_mode = "Rewrite"
            logging.info("Mode: Rewriting Literature Review based on Critic Feedback")
            critic_feedback_to_pass = critic_feedback_and_weaknesses

        else:
            logging.info("Mode: Generating Initial Literature Review")
            critic_feedback_to_pass = None

        # --- Core Steps ---
        # 1. Keywords (only if needed)
        if is_initial_run or force_rewrite_from_scratch:
             self.generate_keywords()

        # 2. Search ArXiv and Rank (only if needed)
        if is_initial_run or force_rewrite_from_scratch:
            self.search_arxiv()
            self.rank_abstracts()
            if not self.relevant_arxiv_ids:
                 logging.error(f"Error: {current_mode} run failed to find/select any relevant articles. Cannot proceed.")
                 return None 

        # 3. Download/Check PDFs (only if needed)
        if is_initial_run or force_rewrite_from_scratch:
             if self.relevant_arxiv_ids:
                  self.download_relevant_pdfs()

             else: # Should not happen due to check above, but for safety
                  logging.error("Error: PDF download step reached but no relevant articles selected.")
                  return None
             
        else: # Normal rewrite iteration
            logging.info("Mode: Rewrite - Skipping PDF download/check, reusing existing files.")
            if not self.relevant_arxiv_ids: # Check if list is somehow empty in rewrite
                 logging.error("Error: Relevant articles list is empty in rewrite mode. Cannot proceed.")
                 return None

        # 4. Extract text from PDFs
        pdf_dir = "downloaded_pdfs"

        logging.info("Checking cache and extracting text from relevant PDFs if needed...")
        extracted_newly_count = 0
        cache_hits_count = 0
        missing_pdf_count = 0
        failed_extraction_count = 0

        current_run_pdf_texts = {}

        for arxiv_id_key in self.relevant_arxiv_ids:

            # --- Caching logic ---
            cached_data = self.pdf_texts.get(arxiv_id_key)
            # Check if valid text is already in cache
            if cached_data and not cached_data.startswith("Error:"):
                logging.debug(f"Cache hit for {arxiv_id_key}.")
                current_run_pdf_texts[arxiv_id_key] = cached_data
                cache_hits_count += 1
                continue # to the next ID

            # --- If not in cache or cache contains an error - attempt extraction ---
            logging.debug(f"Cache miss or error for {arxiv_id_key}. Attempting extraction...")

            safe_filename_id = re.sub(r'[/:.]', '_', arxiv_id_key)
            pdf_filename = os.path.join(pdf_dir, f"{safe_filename_id}.pdf")
            
            if os.path.exists(pdf_filename):
                full_text = self.extract_full_text_from_pdf(pdf_filename)
                if full_text:
                    logging.debug(f"Successfully extracted text for {arxiv_id_key}, adding to cache.")
                    self.pdf_texts[arxiv_id_key] = full_text # <<< save to cache >>>
                    current_run_pdf_texts[arxiv_id_key] = full_text
                    extracted_newly_count += 1
            
                else:
                    # Extraction error
                    logging.warning(f"Text extraction failed for {pdf_filename}.")
                    self.pdf_texts[arxiv_id_key] = "Error: PDF text extraction failed." # save error to cache
                    current_run_pdf_texts[arxiv_id_key] = "Error: PDF text extraction failed."
                    failed_extraction_count += 1
            else:
                 # PDF not find
                logging.warning(f"PDF file not found for {arxiv_id_key} at {pdf_filename}. Cannot extract text.")
                self.pdf_texts[arxiv_id_key] = "Error: PDF file not found."
                missing_pdf_count += 1

        logging.info(f"Text extraction complete. Cache hits: {cache_hits_count}, Newly extracted: {extracted_newly_count}, Missing PDFs: {missing_pdf_count}, Failed extractions: {failed_extraction_count}.")                

        # Check after extraction
        valid_texts_count = sum(1 for txt in current_run_pdf_texts.values() if txt and not txt.startswith("Error:"))
        if valid_texts_count == 0 and self.relevant_arxiv_ids:
             logging.error("Error: Could not extract or retrieve from cache any valid text for relevant PDFs. Cannot proceed.")
             return None        
        
        # 5. Generate Article Reviews
        logging.info("Generating article reviews...")

        self.article_reviews = self.generate_article_reviews(current_run_pdf_texts)
        successful_reviews = sum(1 for review in self.article_reviews.values() if not review.startswith("Error:"))
        if successful_reviews == 0 and self.relevant_arxiv_ids:
            logging.error("Error: Failed to generate any usable article reviews. Cannot proceed.")
            return None
        elif successful_reviews < len(self.relevant_arxiv_ids):
             logging.warning(f"Generated reviews for only {successful_reviews}/{len(self.relevant_arxiv_ids)} articles.")

        # 6. Generate Literature Review Sections
        self.generate_review_sections(self.article_reviews, critic_feedback=critic_feedback_to_pass)
        successful_sections = sum(1 for section_key, content in self.review_sections.items() if content and not content.startswith("Error:"))
        total_sections = len(self.review_sections)
        
        if successful_sections == 0 and total_sections > 0:
             logging.error("Error: Failed to generate any usable literature review sections. Cannot proceed.")
             return None
        elif successful_sections < total_sections:
             logging.warning(f"Generated only {successful_sections}/{total_sections} review sections successfully.")

        # 7. Consolidate Review Sections into text
        self.consolidate_reviews()

        # 8. Save Intermediate JSON (optional but useful)
        os.makedirs(RUN_OUTPUT_DIR, exist_ok=True) # Create dir
        output_filename = os.path.join(RUN_OUTPUT_DIR, f"run_{self.run_count}_literature_review.json")       
        try:
            save_data = {
                "run_count": self.run_count, "mode": current_mode,
                "research_topic": self.research_topic, "specification_prompt": self.specification_prompt,
                "keywords": self.keywords, "relevant_arxiv_ids": self.relevant_arxiv_ids,
                "article_reviews": self.article_reviews, "review_sections": self.review_sections,
                "literature_review": self.literature_review,
            }

            with open(output_filename, "w", encoding='utf-8') as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            logging.info(f"Intermediate literature review for run {self.run_count} saved to {output_filename}")
        
        except Exception as e:
            logging.error(f"Error saving intermediate JSON for run {self.run_count}: {e}", exc_info=True)

        # 9. Return the consolidated review text
        return self.literature_review


# --- ReviewCriticAgent ---
class ReviewCriticAgent:
    """
    A critic agent that reviews a literature review generated by the ResearchAgent,
    provides scores based on defined criteria, and suggests improvements.
    """

    def __init__(self, literature_review_json_path: str, gemini_model_name: str = "gemini-2.0-flash"):
        """
        Initializes the ReviewCriticAgent.

        Args:
            literature_review_json_path: Path to the intermediate literature review JSON file for the current run.
            gemini_model_name: The name of the Gemini model to use (e.g., "gemini-2.0-flash").
        """
        self.literature_review_json_path = literature_review_json_path
        try:
            with open(literature_review_json_path, 'r', encoding='utf-8') as f:
                self.literature_review_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Literature review file not found: {literature_review_json_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error decoding JSON in file {literature_review_json_path}: {e.msg}", e.doc, e.pos) from e

        self.gemini_model_name = gemini_model_name

        # Extract relevant data, providing defaults
        self.literature_review_text = self.literature_review_data.get("literature_review", "")
        self.research_topic = self.literature_review_data.get("research_topic", "Unknown Topic")
        self.specification_prompt = self.literature_review_data.get("specification_prompt", "No Specification")
        
        # Load article reviews used *for this specific review*
        self.article_reviews = self.literature_review_data.get("article_reviews", {})
        self.run_count = self.literature_review_data.get("run_count", "N/A")

        if not self.literature_review_text:
            logging.warning(f"Warning: Literature review text is empty in {literature_review_json_path}.")


    def generate_review_prompt(self) -> str:
        """
        Generates the prompt for the Gemini model to review the literature review.
        """
        article_reviews_json_string = json.dumps(self.article_reviews, indent=4, ensure_ascii=False)
        # Truncate if extremely long
        max_len = 15000
        if len(article_reviews_json_string) > max_len:
            article_reviews_json_string = article_reviews_json_string[:max_len] + "\n... (article reviews truncated)\n}"

        literature_review_text_for_prompt = self.literature_review_text
        if len(literature_review_text_for_prompt) > max_len:
             literature_review_text_for_prompt = literature_review_text_for_prompt[:max_len] + "... (literature review text truncated)"

        prompt = rp.CRITIC_REVIEW_PROMPT_TEMPLATE.format(
            run_count=self.run_count,
            research_topic=self.research_topic,
            specification_prompt=self.specification_prompt,
            literature_review_text_for_prompt=literature_review_text_for_prompt,
            article_reviews_json_string=article_reviews_json_string
        )
        return prompt
    

    def literature_review_critics(self) -> dict:
        """
        Reviews the literature review and returns a JSON object with scores, strengths, weaknesses, and recommendations.

        Returns:
            A dictionary containing the review results in JSON format, or an error structure on failure.
            Saves critics JSON to RUN_OUTPUT_DIR.
        """
        os.makedirs(RUN_OUTPUT_DIR, exist_ok=True) 
        output_filename = os.path.join(RUN_OUTPUT_DIR, f"run_{self.run_count}_review_critics.json")

        max_retries = 3
        attempt_num = 0
        default_error_result = {
            "scores": {}, "strengths": [], "weaknesses": [],
            "recommendations": ["Failed to generate a valid JSON review after multiple attempts."]
        }

        logging.info(f"Attempting to get review critique for run {self.run_count}...")

        while attempt_num < max_retries:
            attempt_num += 1
            logging.info(f"  Review critique attempt {attempt_num}/{max_retries}...")
            try:
                # 1. Generate the prompt
                prompt = self.generate_review_prompt()

                # 2. Get the response from the Gemini model
                response = generate_text(prompt=prompt, model_name=self.gemini_model_name)

                if not response:
                    logging.error(f"  Error: Received empty response from LLM (attempt {attempt_num}).")
                    if attempt_num < max_retries:
                        logging.info(f"  Retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                    else:
                        logging.info(f"  Max retries reached after empty LLM response.")
                        default_error_result["recommendations"] = ["Failed to get LLM response after multiple attempts."]
                        review_result = default_error_result
                        break # Exit loop, use default error

                # 3. Clean and Parse the response as JSON
                try:
                    # More robust JSON extraction
                    match = re.search(r"```json\s*({.*?})\s*```", response, re.DOTALL | re.IGNORECASE)
                    if match:
                        json_string = match.group(1)
                    else:
                        # Fallback to finding the first '{' and last '}'
                        start = response.find('{')
                        end = response.rfind('}')
                        if start != -1 and end != -1 and end > start:
                            json_string = response[start:end+1]
                        else:
                            raise json.JSONDecodeError("Could not find JSON object boundaries.", response, 0)

                    review_result = json.loads(json_string)

                    # Basic validation of structure (can be more detailed)
                    if "scores" not in review_result or "recommendations" not in review_result:
                        raise ValueError("Required keys ('scores', 'recommendations') missing in JSON response.")

                    logging.info(f"  Successfully parsed review critique JSON.")
                    
                    # 4. Save the review result to a JSON file
                    try:
                        with open(output_filename, 'w', encoding='utf-8') as f:
                            json.dump(review_result, f, indent=4, ensure_ascii=False)
                        logging.info(f"  Review critique saved to {output_filename}")
                    except Exception as e:
                        logging.warning(f"  Warning: Failed to save review critique to {output_filename}: {e}")

                    return review_result  # Return review result on success

                except (json.JSONDecodeError, ValueError, AttributeError) as e:
                    logging.error(f"  Error decoding/validating JSON from LLM response (attempt {attempt_num}/{max_retries}): {e}")
                    
                    logging.info("  Raw LLM response snippet:", response[:500]) # Show snippet
                    if attempt_num < max_retries:
                        logging.info(f"  Retrying in 5 seconds...")
                        time.sleep(5)
                        continue # Retry API call
                    else:
                        logging.info(f"  Max retries reached after JSON parsing errors.")
                        default_error_result["recommendations"] = [f"Failed to parse valid JSON review after multiple attempts. Error: {e}"]
                        review_result = default_error_result
                        break # Exit loop, use default error

            except Exception as e: # Catch other exceptions (e.g., API errors during generate_text)
                logging.error(f"  An unexpected error occurred during the review process (attempt {attempt_num}/{max_retries}): {e}")
                if attempt_num < max_retries:
                    logging.info(f"  Retrying in 5 seconds...")
                    time.sleep(5)
                    continue # Retry API call
                else:
                    logging.info(f"  Max retries reached after unexpected error.")
                    default_error_result["recommendations"] = [f"An unexpected error occurred during review after multiple attempts: {e}"]
                    review_result = default_error_result
                    break # Exit loop, use default error

        # Save the error result if loop finished due to errors
        if "Failed" in review_result.get("recommendations", [""])[0]:
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(review_result, f, indent=4, ensure_ascii=False)
                logging.error(f"  Error result saved to {output_filename}")
            except Exception as e:
                logging.warning(f"  Warning: Failed to save error result to {output_filename}: {e}")

        return review_result

    
# --- ReviewRefinerAgent ---
class ReviewRefinerAgent:
    """
    An agent that determines the next step based on critic scores and potentially improves
    a literature review (although improvement is now mainly handled by ResearchAgent's rewrite mode).
    Its main role here is analyzing scores to guide the main loop.
    """
    # RELEVANT_CRITERIA defines which scores are used for decision making
    RELEVANT_CRITERIA = [
    "clarity_and_focus",
    "depth_of_analysis",
    "coverage_of_individual_article_reviews",
    "diversity_of_citation_context",
    "proposed_research_pipeline_evaluation"
    ]
        
    def __init__(self, literature_review_json_path: str, review_critics_json_path: str):
        """
        Initializes the ReviewRefinerAgent.

        Args:
            literature_review_json_path: Path to the intermediate literature review JSON file.
            review_critics_json_path: Path to the review critics JSON file for that review.
        """
        try:
            with open(literature_review_json_path, 'r', encoding='utf-8') as f:
                self.literature_review_data = json.load(f)
            with open(review_critics_json_path, 'r', encoding='utf-8') as f:
                self.review_results = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error: Required input file not found: {e}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error decoding JSON input file: {e.msg}", e.doc, e.pos) from e

        self.literature_review_text = self.literature_review_data.get("literature_review", "")
        self.research_topic = self.literature_review_data.get("research_topic", "Unknown Topic")
        self.specification_prompt = self.literature_review_data.get("specification_prompt", "No Specification")
        self.article_reviews = self.literature_review_data.get("article_reviews", {}) 
        self.run_count = self.literature_review_data.get("run_count", "N/A")

        self.strengths = self.review_results.get("strengths", [])
        self.weaknesses = self.review_results.get("weaknesses", [])
        self.recommendations = self.review_results.get("recommendations", [])
        self.current_scores = self.review_results.get("scores", {})

        if not self.current_scores:
            logging.warning(f"Warning: No scores found in review critics file: {review_critics_json_path}")


    @staticmethod
    def _get_score(scores_dict: Optional[Dict], criterion: str) -> Optional[int]:
        """Safely extracts the score for a criterion."""
        if not scores_dict or criterion not in scores_dict:
            return None
        
        score_data = scores_dict.get(criterion, {})

        if isinstance(score_data, dict):
            score_value = score_data.get("score")
            if isinstance(score_value, int):
                return score_value
            elif score_value is not None:
                logging.warning(f"Score for criterion '{criterion}' is not an integer: {score_value}")
        return None

    def determine_next_action(self, previous_scores: Optional[dict] = None) -> str:
        """
        Determines the next action based on current and previous review scores.

        Args:
            previous_scores (Optional[dict]): Scores from the previous iteration.

        Returns:
            str: A status indicating the next action:
                'STOP_ALL_5' - All relevant scores are 5. Stop successfully.
                'STOP_PLATEAU_OR_WORSE_HIGH' - All scores >= 4, but plateaued or worsened. Stop, use previous best.
                'CONTINUE_IMPROVE' - Scores are improving or stable but haven't met stopping criteria. Continue refinement.
                'REWRITE_FROM_SCRATCH' - Scores < 4 and worsened or plateaued. Trigger a full rewrite.
                'ERROR_NO_SCORES' - Cannot determine action due to missing scores.
        """
        logging.info(f"Determining next action for run {self.run_count}...")
        if not self.current_scores:
            logging.error("  Error: Current scores are missing.")
            return 'ERROR_NO_SCORES'

        # --- Rule 1: Check if all relevant scores are 5 ---
        all_5 = True
        for criterion in self.RELEVANT_CRITERIA:
            score = self._get_score(self.current_scores, criterion)
            if score is None:
                logging.warning(f"  Warning: Score missing for relevant criterion '{criterion}'. Cannot confirm all 5s.")
                all_5 = False
                break
            if score != 5:
                all_5 = False
                break
        if all_5:
            logging.info("  Action: STOP_ALL_5 (All relevant scores are 5).")
            return 'STOP_ALL_5'

        # --- Rule 2: Analyze based on score levels and changes ---
        min_score = 5
        all_above_or_equal_4 = True
        improvement_detected = False
        worsening_detected = False

        for criterion in self.RELEVANT_CRITERIA:
            current_score = self._get_score(self.current_scores, criterion)

            if current_score is None:
                logging.warning(f"  Warning: Score missing for relevant criterion '{criterion}'. Cannot reliably determine status.")
                current_score = 0 
                all_above_or_equal_4 = False

            if current_score < 4:
                all_above_or_equal_4 = False
            min_score = min(min_score, current_score) # Track the actual minimum score

            # Compare with previous scores if available
            if previous_scores:
                previous_score = self._get_score(previous_scores, criterion)
                if previous_score is None:
                    # If previous was missing but current exists, ignore change.
                    logging.info(f"  Previous score missing for {criterion}, cannot detect change.")
                    continue

                if current_score > previous_score:
                    improvement_detected = True
                    logging.info(f"  Improvement detected: {criterion} ({previous_score} -> {current_score})")
                elif current_score < previous_score:
                    worsening_detected = True
                    logging.info(f"  Worsening detected: {criterion} ({previous_score} -> {current_score})")


        plateau_detected = not improvement_detected and not worsening_detected and previous_scores is not None

        if plateau_detected:
            logging.info("  Plateau detected: No scores improved or worsened compared to previous run.")

        # --- Apply Decision Logic ---
        if all_above_or_equal_4:
            # All scores >= 4
            if improvement_detected:
                logging.info("  Action: CONTINUE_IMPROVE (Scores >= 4 and improving).")
                return 'CONTINUE_IMPROVE'
            
            elif worsening_detected or plateau_detected:
                logging.info("  Action: STOP_PLATEAU_OR_WORSE_HIGH (Scores >= 4, but plateaued or worsened. Use previous best).")
                return 'STOP_PLATEAU_OR_WORSE_HIGH'
            
            else: # First evaluation and all >= 4
                logging.info("  Action: CONTINUE_IMPROVE (First evaluation, scores >= 4).")
                return 'CONTINUE_IMPROVE'
        else:
            # At least one score < 4
            if improvement_detected:
                logging.info("  Action: CONTINUE_IMPROVE (Scores < 4 but improving).")
                return 'CONTINUE_IMPROVE'
            
            elif worsening_detected or plateau_detected:
                logging.info("  Action: REWRITE_FROM_SCRATCH (Scores < 4 and plateaued or worsened).")
                return 'REWRITE_FROM_SCRATCH'
            
            else: # First evaluation and some < 4
                logging.info("  Action: CONTINUE_IMPROVE (First evaluation, scores < 4).")
                return 'CONTINUE_IMPROVE'


    def get_critic_feedback_for_rewrite(self) -> str:
        """Combines weaknesses and recommendations into a feedback string."""
        feedback_lines = ["**Weaknesses Identified:**"]
        if self.weaknesses:
            feedback_lines.extend([f"- {w}" for w in self.weaknesses])
        else:
            feedback_lines.append("- None specified.")

        feedback_lines.append("\n**Recommendations for Improvement:**")
        if self.recommendations:
            feedback_lines.extend([f"- {r}" for r in self.recommendations])
        else:
            feedback_lines.append("- None specified.")

        return "\n".join(feedback_lines)

    
