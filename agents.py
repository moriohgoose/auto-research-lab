import os
import json
import re
import time
import io
from datetime import datetime
from typing import Optional, List, Dict
import logging
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import requests
import arxiv
import google.generativeai as genai
from utils import generate_text 

RUN_OUTPUT_DIR = "review_runs"

# --- ResearchAgent ---
class ResearchAgent:
    """
    A research agent that conducts literature reviews using the Gemini model and arXiv API.
    Gathers article metadata, ranks articles, downloads PDFs, extracts text,
    generates individual article reviews, synthesizes these into a literature review,
    and can rewrite the review based on critic feedback.
    Also generates title and abstract for the final review.
    """

    system_prompt = """
        You are an experienced researcher specializing in the in-depth analysis of scientific literature.
        Your task is to conduct a literature review for a given research topic.
        Pay close attention to the methodology used in the articles, and ensure your summaries include comprehensive details on each of the below:
        - The **methods** they used to come to their conclusions. Be specific and include details about experimental or simulation procedures.
        - The **resources** used to support the analysis, for example any special datasets used, equipment or computing power.
        - A summary of each of the **steps** that were used in the paper's experiments to come to its conclusions.
        """

    rewrite_prompt_template = """
        You are an expert researcher and scientific writer, tasked with **rewriting and significantly improving** a literature review based on the provided feedback.
        Research Topic: {research_topic}
        Specification: {specification_prompt}

        **Critique and Recommendations:**
        {critic_recommendations_and_weaknesses}

        **Article Reviews (MUST be used for the rewrite):**
        {article_reviews_content}

        **Instructions:**
        1.  **Thoroughly address ALL weaknesses and recommendations** from the critique.
        2.  **Integrate insights** from the provided Article Reviews more deeply and effectively, especially regarding methods, resources, results, and limitations. Ensure diverse citation context.
        3.  **Strengthen the analysis**, going beyond summaries to synthesize findings, identify patterns/contradictions, and evaluate methods critically.
        4.  **Refine the proposed research pipeline** to be more concrete, detailed, and actionable, with clear steps, methods, resources, and success criteria, directly addressing the research topic and insights from the review.
        5.  **Maintain the original strengths** identified in the review while making these improvements.
        6.  **Ensure clarity, focus, and coherence** throughout the rewritten review.
        7.  **Rewrite the ENTIRE literature review** incorporating all necessary changes. Do not just patch sections.
        """

    def __init__(
        self,
        research_topic: str = "How to use LLMs for drug research",
        keywords: Optional[List[str]] = None,
        specification_prompt: Optional[str] = None,
        article_count_filter: int = 5,
        article_years_filter: Optional[str] = None,
    ):
        """
        Initializes the ResearchAgent.

        Args:
            research_topic: The topic of the literature review.
            keywords: Optional list of keywords to use for the arXiv search.
            specification_prompt: Optional prompt for specifying the research type or recommendations.
            article_count_filter: The number of articles to include in the final literature review (after ranking).
            article_years_filter: The number of years range to filter the article search by (e.g., "2020-2025"). None for no filter.
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
        self.content_type_used = "Article Reviews (from PDF text)"

        logging.info(f"ResearchAgent initialized for topic: '{self.research_topic}'")

    def generate_keywords(self):
        """
        Generates keywords for the research topic using the Gemini model if keywords are not provided.
        """
        if not self.keywords:
            prompt = f"""Please suggest at least 4 keywords related to the research topic: {self.research_topic}.
            Write ONLY keywords IN ONE LINE, separated by commas. Do not include any extra text.
            Example: LLMs, drug discovery, AI, cheminformatics"""
            
            logging.info("Generating keywords...")
            response = generate_text(prompt=prompt)

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

        prompt_lines = [
            "You are an expert researcher. Rank the following abstracts based ONLY on their relevance to the research topic and specification provided below.",
            f"Research Topic: {self.research_topic}",
        ]
        if self.specification_prompt:
            prompt_lines.append(f"Specification: {self.specification_prompt}")
        prompt_lines.append("\nAbstracts to Rank (ID and Summary):")

        abstract_texts_for_prompt = []
        for arxiv_id_key, metadata in self.article_metadata.items():
            abstract = metadata.get('summary', 'No summary available.')
            truncated_abstract = abstract[:500] + "..." if len(abstract) > 500 else abstract
            abstract_texts_for_prompt.append(f"ID: {arxiv_id_key}\nAbstract: {truncated_abstract}\n---")

        prompt_lines.append("\n".join(abstract_texts_for_prompt))
        prompt_lines.append(
            "\n**Instructions:**"
            "\n1. Evaluate each abstract's relevance to the *specific* research topic and specification."
            "\n2. Output ONLY a ranked list of the ArXiv IDs (full URL starting with http://arxiv.org/abs/), starting with the most relevant."
            "\n3. List one ID per line."
            "\n4. Do NOT include any other text, explanations, or scores."
            "\nExample Output:\nhttp://arxiv.org/abs/2305.12345\nhttp://arxiv.org/abs/2401.67890\n..."
        )

        ranking_prompt = "\n".join(prompt_lines)
        logging.debug(f"Ranking prompt (first 500 chars): {ranking_prompt[:500]}")

        ranking_response = generate_text(prompt=ranking_prompt)

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
                        # else: logging.debug(f"Skipping duplicate ID from LLM: {arxiv_id_key}")
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
            review_prompt = ("""
                {self.system_prompt}
                You are now focusing on reviewing a SINGLE research paper in extreme detail.
                **Paper Context:**
                - ArXiv ID: {arxiv_id}
                - Research Topic Context: {self.research_topic}
                - Specification Focus: {self.specification_prompt or 'General Relevance'}
                
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
                     - How relevant and novel do you assess this paper to be concerning the specific topic: '{self.research_topic}'? Justify briefly.
                     - Are there indications of data/code availability for reproducibility?
                 
                 **Output Format:**
                 Produce ONLY the structured review text. Do not add any introductory or concluding remarks outside of the specified sections.
            """)

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
        base_system_role = ("""
                    You are an expert research synthesizer and scientific writer. Your task is to write sections of a **critically analytical, highly informative, and DETAILED** literature review on:
                    **Research Topic:** {self.research_topic}
                    **Specification:** {self.specification_prompt}
                    You MUST base your writing **strictly** on the provided Article Reviews, identified by [ID: number] at the start of each review. Synthesize, compare, contrast, and critique the information **in depth**.
                    **Citation Instruction:** When referencing findings or methods from a specific article review, cite it using ONLY the format `[ID: number]`, for example: `[ID: 2410.11550v1]`. Do NOT use full URLs or author names for citations within the text.
                    **Formatting Instruction:** Do NOT include Markdown headings (like ## Introduction or **Introduction**) in your response text. **Do NOT use markdown tables.** Describe all information using prose sentences and paragraphs.
                    **Detail Requirement:** Ensure each section provides **substantial detail, thorough analysis, and sufficient examples** drawn directly from the provided reviews to justify claims and comparisons. Avoid superficial summaries." 
                """)

        # Define sections, prompts, and dependencies
        sections_to_generate = {
            "introduction": {
                "prompt_template": (
                    "{system_role}\n\n"
                    "{critic_instructions}"
                    "**Task:** Write ONLY the **Introduction** section body text.\n\n"
                    "**Instructions for Introduction (minimum 30-35 sentences):**\n"
                    "- Briefly introduce the research topic '{research_topic}' and its significance.\n"
                    "- State the specific scope or question this review addresses, guided by the specification '{specification_prompt}'.\n"
                    "- Briefly mention the types of studies included (based on the reviews [ID: number]) and the overall goal of this review (e.g., synthesize methods, identify gaps).\n"
                    "- Highlight the importance of this research area.\n\n"
                    "**Article Reviews Provided:**\n{article_reviews}\n\n"
                    "**Output:** Write ONLY the Introduction section body text (no heading)."
                ),
                "previous_section_keys": []
            },
            "key_themes": {
                "prompt_template": (
                    "{system_role}\n\n"
                    "{critic_instructions}"
                    "**Task:** Write ONLY the **Key Themes and Trends** section body text, building upon the (unseen) introduction.\n\n"
                    "**Instructions for Key Themes and Trends (minimum 20-25 sentences):**\n"
                    "- **Critically analyze and synthesize** the provided Article Reviews [ID: number].\n"
                    "- Identify and discuss 2-4 major recurring themes, trends, or common approaches found across the papers.\n"
                    "- For each theme, explain it clearly and provide specific supporting examples **directly** from the article reviews, citing using `[ID: number]` format.\n"
                    "- Compare and contrast how different papers address these themes.\n"
                    "- Discuss the significance or implications of these trends.\n\n"
                    "- Identify and discuss any **emerging or less common (but potentially important) trends** observed in the reviewed articles.\n"
                    "**Article Reviews Provided:**\n{article_reviews}\n\n"
                    "**Output:** Write ONLY the Key Themes and Trends section body text (no heading)."
                ),
                "previous_section_keys": ["introduction"]
            },
            "methods_technologies": {
                 "prompt_template": (
                     "{system_role}\n\n"
                     "{critic_instructions}"
                     "**Task:** Write ONLY the **Methods and Technologies** section body text.\n\n"
                     "**Previously Written Sections (Context Only):**\n{previous_sections}\n\n"
                     "**Instructions for Methods and Technologies (minimum 30-35 sentences):**\n"
                     "- **Critically analyze and compare** the different methods and technologies described in the Article Reviews [ID: number].\n"
                     "- Group similar methods. Discuss the strengths and weaknesses of key approaches *as suggested by the reviews or your analysis of them*.\n"
                     "- Provide specific details about prominent algorithms, techniques, or experimental procedures mentioned.\n"
                     "- Analyze and compare the **evaluation metrics** used in the different studies [ID: number]. Are they consistent? Are they appropriate for the tasks? Discuss any limitations noted regarding evaluation.\n"
                     "- Are there notable differences in methodology for similar problems? Discuss them.\n"
                     "- Cite specific examples using `[ID: number]` format.\n\n"
                     "**Article Reviews Provided:**\n{article_reviews}\n\n"
                     "**Output:** Write ONLY the Methods and Technologies section body text (no heading)."
                 ),
                 "previous_section_keys": ["introduction", "key_themes"]
            },
            "resources_datasets": {
                 "prompt_template": (
                    "{system_role}\n\n"
                    "{critic_instructions}"
                    "**Task:** Write ONLY the **Resources and Datasets** section body text.\n\n"
                    "**Previously Written Sections (Context Only):**\n{previous_sections}\n\n"
                    "**Instructions for Resources and Datasets (minimum 20-25 sentences):**\n"
                    "- **Synthesize** information about the datasets, software, hardware, or other resources mentioned in the Article Reviews [ID: number].\n"
                    "- Identify commonly used resources or types of resources.\n"
                    "- Discuss the characteristics of key datasets mentioned (e.g., size, type, source).\n"
                    "- Are there mentions of resource limitations or the importance of specific resources? Include these insights.\n"
                    "- Cite specific examples using `[ID: number]` format.\n\n"
                    "**Article Reviews Provided:**\n{article_reviews}\n\n"
                    "**Output:** Write ONLY the Resources and Datasets section body text (no heading)."
                 ),
                 "previous_section_keys": ["introduction", "key_themes", "methods_technologies"]
            },
            "challenges_limitations": {
                 "prompt_template": (
                     "{system_role}\n\n"
                     "{critic_instructions}"
                     "**Task:** Write ONLY the **Challenges and Limitations** section body text.\n\n"
                     "**Previously Written Sections (Context Only):**\n{previous_sections}\n\n"
                     "**Instructions for Challenges and Limitations (minimum 20-25 sentences):**\n"
                     "- **Synthesize** the challenges, limitations, and weaknesses identified **within the provided Article Reviews** [ID: number] (both self-reported by authors and noted in the review itself).\n"
                     "- **Categorize** the identified challenges (e.g., data-related, methodological, computational, ethical).\n"
                     "- Discuss the **potential impact** of each major challenge category on the field's progress.\n"
                     "- Are there specific **mitigation strategies** proposed or implied in the reviewed papers for these challenges? Discuss them.\n"
                     "- What are the recurring obstacles or gaps mentioned across the papers?\n"
                     "- Discuss the implications of these limitations for the field and for the research topic '{research_topic}'.\n"
                     "- Cite specific examples using `[ID: number]` format.\n\n"
                     "**Article Reviews Provided:**\n{article_reviews}\n\n"
                     "**Output:** Write ONLY the Challenges and Limitations section body text (no heading)."
                 ),
                 "previous_section_keys": ["introduction", "key_themes", "methods_technologies", "resources_datasets"]
             },
             "proposed_pipeline": {
                 "prompt_template": (
                     "{system_role}\n\n"
                     "{critic_instructions}"
                     "**Task:** Write ONLY the **Proposed Research Pipeline** section body text.\n\n"
                     "**Previously Written Sections (Context Only):**\n{previous_sections}\n\n"
                     "**Instructions for Proposed Research Pipeline (minimum 30-35 sentences):**\n"
                     "1.  **Introduction Paragraph(s):** Begin with 1-2 paragraphs outlining the overall goal and core principles of the proposed pipeline. Explain *why* this pipeline structure is proposed, based **strictly** on the analysis of the provided Article Reviews (methods, resources, challenges synthesized in previous sections). Briefly mention the key limitations it aims to address.\n\n"
                     "2.  **Pipeline Stages (Describe each stage in a separate paragraph, clearly numbering or naming them, e.g., 'Stage 1: Data Curation', 'Stage 2: Model Adaptation', etc.):**\n"
                     "    *   Propose **5-7 distinct, sequential stages** for a comprehensive research pipeline addressing the topic '{research_topic}'.\n"
                     "    *   For **EACH stage**: \n"
                     "        *   Clearly state the **purpose** of the stage.\n"
                     "        *   Suggest **specific, named methods, tools, or dataset types** inspired by the reviewed papers [ID: number].\n"
                     "        *   Briefly **justify** the choice of these components for this stage, linking back to the synthesized knowledge (e.g., 'Inspired by the RAG approach in [ID: X], this stage will...').\n"
                     "        *   Mention potential **evaluation metrics or intermediate validation points** for this stage.\n\n"
                     "3.  **Challenges and Mitigation Paragraph(s):** After describing the stages, dedicate 1-2 paragraphs to discuss the **potential challenges** specifically for *this proposed pipeline* (e.g., data integration, computational cost, validation hurdles) and suggest **concrete mitigation strategies**, referencing insights from the reviews [ID: number] where applicable.\n\n"
                     "4.  **Concluding Summary Paragraph:** End with a brief paragraph summarizing the expected outcomes and significance of implementing this proposed pipeline for advancing research on '{research_topic}'.\n\n"
                     "**Article Reviews Provided:**\n{article_reviews}\n\n"
                     "**Output:** Write ONLY the Proposed Research Pipeline section body text, following the 4-part structure described above (Intro, Stages, Challenges, Conclusion). Use clear paragraph breaks between parts and between stages."
                 ),
                 "previous_section_keys": ["introduction", "key_themes", "methods_technologies", "resources_datasets", "challenges_limitations"]
             },
             "conclusion": {
                 "prompt_template": (
                     "{system_role}\n\n"
                     "{critic_instructions}"
                     "**Task:** Write ONLY the **Conclusion** section body text.\n\n"
                     "**Previously Written Sections (Context Only):**\n{previous_sections}\n\n"
                     "**Instructions for Conclusion (minimum 25-30 sentences):**\n"
                     "- Briefly summarize the key themes, dominant methods, and major challenges identified in the review.\n"
                     "- Provide an overall critical assessment of the state of research on '{research_topic}' based **only** on the reviewed articles.\n"
                     "- Reiterate the main insights regarding effective research approaches or pipelines derived from the synthesis.\n"
                     "- Briefly suggest the most promising directions for future research **emerging directly from the analysis of the reviewed papers and the proposed pipeline**.\n"
                     "- Conclude with the potential impact or significance of advancing research in this area.\n"
                     "- Cite key supporting points using `[ID: number]` format.\n\n"
                     "**Article Reviews Provided:**\n{article_reviews}\n\n"
                     "**Output:** Write ONLY the Conclusion section body text (no heading)."
                 ),
                 "previous_section_keys": ["introduction", "key_themes", "methods_technologies", "resources_datasets", "challenges_limitations", "proposed_pipeline"]
            }
        }

        # Prepare critic feedback prompt part
        critic_instructions_prompt = ""
        if critic_feedback:
             logging.info("  Generating sections in rewrite mode based on critic feedback.")
             critic_instructions_prompt = (
                 "**CRITICAL REWRITE INSTRUCTIONS:**\nYou MUST rewrite the sections based on the following critique. Focus on addressing all weaknesses and recommendations:\n"
                 f"{critic_feedback}\n"
                 "Ensure the rewritten sections are significantly improved and integrate insights from the Article Reviews more effectively.\n---\n"
             )

        successful_sections_count = 0
        
        # Generate sections sequentially
        for section_key, config in sections_to_generate.items():
            logging.info(f"  Generating section: {section_key}...")
            previous_sections_text = "\n\n".join(
                [f"--- Context from {prev_key} ---:\n{self.review_sections.get(prev_key, 'Not generated yet.')[:500]}..." # Send only snippet of previous for context
                 for prev_key in config["previous_section_keys"]]
            )

            prompt = config["prompt_template"].format(
                system_role=base_system_role,
                research_topic=self.research_topic,
                specification_prompt=self.specification_prompt or "General overview",
                article_reviews=article_reviews_full_prompt_text,
                previous_sections=previous_sections_text or "This is the first section.",
                critic_instructions=critic_instructions_prompt
            )

            section_content = generate_text(prompt=prompt)
            
            if section_content:
                # Remove potential leading/trailing markdown, extra spaces
                section_content = re.sub(r"^\s*#+\s*.*\s*", "", section_content).strip() # Remove leading # headings
                section_content = re.sub(r"^\s*\*\*+\s*.*\s*\*\*+\s*", "", section_content).strip() # Remove leading ** headings
                self.review_sections[section_key] = section_content
                successful_sections_count += 1
                logging.info(f"    Successfully generated section: {section_key}")
            else:
                self.review_sections[section_key] = f"Error: Failed to generate section '{section_key}'."
                logging.error(f"    Error generating section: {section_key}")

        logging.info(f"Finished generating review sections. Successfully generated {successful_sections_count}/{len(sections_to_generate)} sections.")
        if successful_sections_count < len(sections_to_generate):
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

        prompt = f"""
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
                Abstract: This review synthesizes recent advancements in leveraging Large Language Models (LLMs) for drug discovery. Key themes identified include the development of domain-specific models and multi-modal approaches. Common methods involve fine-tuning pre-trained LLMs and integrating knowledge graphs. Significant challenges remain, such as data scarcity, model interpretability, and ethical considerations. The review concludes by highlighting the potential of LLMs while proposing a research pipeline focused on improved benchmarking and explainability to further advance the field.

                Your Turn:
                Title:
                Abstract:
                """
        
        response = generate_text(prompt)
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
        self.pdf_texts.clear()

        logging.info("Extracting text from relevant PDFs...")

        extracted_texts_count = 0
        missing_pdf_count = 0

        for arxiv_id_key in self.relevant_arxiv_ids:
            safe_filename_id = re.sub(r'[/:.]', '_', arxiv_id_key)
            pdf_filename = os.path.join(pdf_dir, f"{safe_filename_id}.pdf")
            
            if os.path.exists(pdf_filename):
                full_text = self.extract_full_text_from_pdf(pdf_filename)
                self.pdf_texts[arxiv_id_key] = full_text if full_text else "Error: PDF text extraction failed."
                if full_text: extracted_texts_count += 1
            
            else:
                 logging.warning(f"PDF file not found for {arxiv_id_key} at {pdf_filename}. Cannot extract text.")
                 self.pdf_texts[arxiv_id_key] = "Error: PDF file not found."
                 missing_pdf_count += 1

        if extracted_texts_count == 0 and self.relevant_arxiv_ids:
             logging.error("Error: Could not extract text from ANY relevant PDF. Cannot proceed.")
             return None
        
        if missing_pdf_count > 0:
             logging.warning(f"Could not find PDF files for {missing_pdf_count}/{len(self.relevant_arxiv_ids)} articles.")

        if extracted_texts_count < len(self.relevant_arxiv_ids) - missing_pdf_count:
             logging.warning(f"Successfully extracted text from only {extracted_texts_count} out of {len(self.relevant_arxiv_ids) - missing_pdf_count} PDFs found.")

        # 5. Generate Article Reviews
        self.article_reviews = self.generate_article_reviews(self.pdf_texts)
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

    def __init__(self, literature_review_json_path: str):
        """
        Initializes the ReviewCriticAgent.

        Args:
            literature_review_json_path: Path to the intermediate literature review JSON file for the current run.
        """
        self.literature_review_json_path = literature_review_json_path
        try:
            with open(literature_review_json_path, 'r', encoding='utf-8') as f:
                self.literature_review_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Literature review file not found: {literature_review_json_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error decoding JSON in file {literature_review_json_path}: {e.msg}", e.doc, e.pos) from e

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


        prompt = f"""You are a highly experienced and critical academic reviewer specializing in literature reviews.
                    Your task is to evaluate the following literature review (run {self.run_count}) and provide constructive feedback for improvement, following the structured JSON format requested.
                    The review focuses on:
                    Research Topic: "{self.research_topic}"
                    Specification: "{self.specification_prompt}"

                    **Literature Review Text to Evaluate:** {literature_review_text_for_prompt}
                    **Article Reviews Used to Generate This Literature Review (potentially truncated):** {article_reviews_json_string}
                    Evaluation Criteria & JSON Output Structure:

                    Evaluate the literature review based on the following criteria. For each criterion:

                        Provide a concise "Assessment" explaining your reasoning with specific examples from the review text.

                        Assign a "Score" from 1 (Very Poor) to 5 (Excellent).

                    Output your entire response as a SINGLE, VALID JSON object. Do NOT include ANY text outside this JSON object.

                    JSON Structure:
                    {{
                    "scores": {{
                    "clarity_and_focus": {{
                    "assessment": "Your assessment text here...",
                    "score": <INT 1-5>
                    }},
                    "depth_of_analysis": {{
                    "assessment": "Your assessment text here... Does it synthesize, compare, contrast, critique? Or just summarize?",
                    "score": <INT 1-5>
                    }},
                    "coverage_of_individual_article_reviews": {{
                    "assessment": "Your assessment text here... Are key methods, results, limitations from the provided article reviews reflected accurately and sufficiently? Estimate coverage %.",
                    "score": <INT 1-5>
                    }},
                    "diversity_of_citation_context": {{
                    "assessment": "Your assessment text here... Are articles cited across different sections (Intro, Methods, Challenges, Conclusion) to support various points, or confined to one section?",
                    "score": <INT 1-5>
                    }},
                    "proposed_research_pipeline_evaluation": {{
                    "assessment": "Your assessment text here... How concrete, actionable, and well-justified is the proposed pipeline? Does it address the topic/gaps? Are steps/methods/metrics clear?",
                    "score": <INT 1-5>
                    }}
                    // Note: Removed "citation_quality" as it's hard to judge format from text alone and covered by other criteria.
                    }},
                    "strengths": [
                    "List the main strengths concisely.",
                    "Example: Good synthesis of methods in section X.",
                    "..."
                    ],
                    "weaknesses": [
                    "List the main weaknesses concisely.",
                    "Example: Proposed pipeline lacks specific metrics.",
                    "Example: Analysis doesn't sufficiently compare approaches from paper A and B.",
                    "..."
                    ],
                    "recommendations": [
                    "Provide specific, actionable recommendations for improvement.",
                    "Example: 'In the pipeline section, define specific target metrics for evaluating method X.'",
                    "Example: 'Expand the 'Methods' section to explicitly compare the algorithms used in ArXiv:1234 and ArXiv:5678.'",
                    "..."
                    ]
                    }}

                    Detailed Criteria Levels (for your reference during evaluation):

                        Clarity and Focus (1-5):

                            5: Excellent clarity, sharp focus on topic/specification, strong central argument woven throughout.

                            4: Good clarity and focus, mostly consistent, clear purpose.

                            3: Generally clear, but some minor deviations or weaker focus.

                            2: Struggles with clarity/focus, inconsistent, weak argument.

                            1: Lacks clarity/focus, deviates significantly, no clear argument.

                        Depth of Analysis (1-5):

                            5: Highly insightful, goes beyond summary, synthesizes, critiques, identifies nuances, offers original perspectives.

                            4: Good analysis, identifies patterns/trends/gaps, synthesizes well.

                            3: Some analysis, identifies basic patterns, but somewhat superficial.

                            2: Minimal analysis, mostly summary with limited connections.

                            1: No real analysis, just lists findings.

                        Coverage of Individual Article Reviews (1-5): (Based on the provided article review texts)

                            5: Comprehensive integration (>80%) of key findings, methods, limitations from article reviews; accurate reflection.

                            4: Good coverage (60-80%), most key aspects included.

                            3: Moderate coverage (40-60%), some important details missed.

                            2: Minimal coverage (20-40%), many key aspects ignored.

                            1: Almost no coverage (<20%), ignores/misrepresents article reviews.

                        Diversity of Citation Context (1-5):

                            5: Key articles strategically cited/discussed in 4-5+ different sections, enriching multiple facets.

                            4: Most articles cited in 3+ different sections, showing good integration.

                            3: Mix of citation contexts, but some key articles confined to 1-2 sections.

                            2: Most articles cited in only 1-2 sections, superficial integration.

                            1: Articles almost exclusively cited in one section, lack of integration.

                        Proposed Research Pipeline Evaluation (1-5):

                            5: Exceptionally well-defined, actionable, specific methods/tools/datasets/metrics for every step, directly addresses gaps.

                            4: Well-defined pipeline, most steps have specifics, but some minor details lacking.

                            3: Some concrete steps, but lacks detail or clear success criteria in many areas.

                            2: Poorly defined, lacks essential details (methods, data, criteria).

                            1: No pipeline proposed, or completely abstract/vague.

                    Ensure your final output is ONLY the JSON object.
                    """
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
                response = generate_text(prompt=prompt)

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

    
