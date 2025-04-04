import google.generativeai as genai
import time
import requests
import re
from typing import Optional, List, Dict
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import black
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import logging
from reportlab.lib.pagesizes import A4


def generate_text(prompt):
    """
    Generates text using the Gemini model. Includes basic retry logic.
    """
    logging.info(f"Sending prompt to Gemini ({len(prompt)} chars): {prompt[:150]}...")
    max_retries = 2
    attempt = 0
    delay = 5

    while attempt < max_retries:
        attempt += 1
        try:
            model = genai.GenerativeModel('gemini-2.0-flash') 
            response = model.generate_content(prompt)

            response_text = None
            try:
                 response_text = response.text
            except ValueError:
                 try:
                      if response.candidates:
                           candidate = response.candidates[0]
                           if candidate.content and candidate.content.parts:
                                response_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                      else:
                           logging.error(f"  Gemini response has no text and no candidates. Response: {response}")
                 except (AttributeError, IndexError, Exception) as e_cand:
                      logging.error(f"  Error accessing candidates: {e_cand}. Original response: {response}")

            if response_text:
                logging.info(f"  Gemini API response received ({len(response_text)} chars): {response_text[:150]}...")
                return response_text
            else:
                 logging.warning(f"  Gemini API returned empty or blocked content (Attempt {attempt}/{max_retries}).")
                 return None

        except Exception as e:
            logging.error(f"  Gemini API error (Attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                logging.info(f"  Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                logging.error("  Max retries reached after API errors.")
                return None
    return None


def markdown_to_reportlab(text: str) -> str:
    """Converts basic Markdown (bold, italics) to ReportLab inline tags."""
    if not text:
        return ""
    # **bold** to <b>bold</b> 
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # *italic* to <i>italic</i> 
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    return text


def replace_ids_with_numbers(text: str, arxiv_ids_ordered: list, metadata: dict) -> str:
    """
    Replaces [ID: xxx], [ID: xxx, yyy], etc. citations in text with numbered citations [1], [1, 5], etc.
    based on the order provided in arxiv_ids_ordered. More robust parsing.
    """
    if not text or not arxiv_ids_ordered or not metadata:
        return text

    id_num_to_citation_num_map = {}
    for i, arxiv_id_key in enumerate(arxiv_ids_ordered):
        match = re.search(r'([\d.]+v?\d*)$', arxiv_id_key) 
        if match:
            arxiv_num_only = match.group(1)
            id_num_to_citation_num_map[arxiv_num_only] = i + 1
        else:
            logging.warning(f"Could not extract ID number from key '{arxiv_id_key}' for citation mapping.")

    if not id_num_to_citation_num_map:
        logging.warning("Citation map is empty, cannot replace IDs with numbers.")
        return text

    logging.info("Replacing text citations [ID: number(s)] with numbered citations [1], [2, 3]...")


    def get_citation_numbers(ids_string: str) -> Optional[str]:
        """Takes the content inside [ID: ...], finds numbers, returns comma-separated citation numbers."""
        # All IDs 
        extracted_id_nums = re.findall(r'([\d.]+v?\d*)', ids_string)
        if not extracted_id_nums:
            return None 

        citation_numbers = []
        all_found = True
        for id_num in extracted_id_nums:
            citation_num = id_num_to_citation_num_map.get(id_num)
            if citation_num is not None:
                citation_numbers.append(str(citation_num))
            else:
                logging.warning(f"Citation ID '{id_num}' found in text, but not in reference list map.")
                all_found = False
                break 

        if all_found and citation_numbers:
            return ", ".join(citation_numbers) 
        else:
            return None 


    def replacer(match):
        ids_content = match.group(1) 
        replacement_nums = get_citation_numbers(ids_content)
        if replacement_nums:
            return f"[{replacement_nums}]"
        else:
            return match.group(0)
        
    pattern = r"\[ID:\s*([^\]]+?)\s*\]"
    processed_text = re.sub(pattern, replacer, text)

    original_matches_count = len(re.findall(pattern, text))
    processed_matches_count = len(re.findall(r"\[(\d+(?:,\s*\d+)*)\]", processed_text))
    logging.info(f"Citation replacement: Processed {original_matches_count} potential [ID:...] groups, resulted in {processed_matches_count} numbered citation groups.")
    if original_matches_count > processed_matches_count:
        logging.warning(f"  {original_matches_count - processed_matches_count} citation groups were not replaced (likely due to missing IDs in map).")

    return processed_text


def generate_pdf_report(title: Optional[str], abstract: Optional[str],
                        review_sections: Dict[str, str],
                        references: List[str],
                        arxiv_ids_ordered: List[str],
                        article_metadata: Dict[str, dict],
                        filename: str = "literature_review.pdf"):
    """
    Generates a PDF report from review sections dictionary.
    No section numbering, converts markdown, numbered citations.
    Page break ONLY before References.
    """
    logging.info(f"Generating PDF report: {filename}...")
    if not review_sections:
         logging.error("Cannot generate PDF: review_sections dictionary is empty.")
         return False

    try:
        doc = SimpleDocTemplate(filename, pagesize=A4,
                                leftMargin=1.5*cm, rightMargin=1.5*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()


        FONT_NAME = 'Times-Roman'; 
        FONT_BOLD = 'Times-Bold'

        doc_title_style = styles['h1']; doc_title_style.fontName = FONT_BOLD; doc_title_style.fontSize = 16; doc_title_style.alignment = TA_CENTER; doc_title_style.spaceAfter = 0.3*inch
        section_heading_style = ParagraphStyle(name='SectionHeading', parent=styles['h2'], fontName=FONT_BOLD, fontSize=13, alignment=TA_LEFT, spaceBefore=18, spaceAfter=8, keepWithNext=1); styles.add(section_heading_style)
        abstract_style = ParagraphStyle(name='AbstractStyle', parent=styles['Normal'], fontName=FONT_NAME, fontSize=10, alignment=TA_JUSTIFY, leftIndent=0, rightIndent=0, spaceBefore=6, spaceAfter=0.2*inch, leading=12); styles.add(abstract_style)
        body_style = styles['Normal']; body_style.fontName = FONT_NAME; body_style.fontSize = 11; body_style.alignment = TA_JUSTIFY; body_style.firstLineIndent = 0.25*inch; body_style.spaceAfter = 6; body_style.leading = 14
        ref_style = ParagraphStyle(name='ReferenceStyle', parent=styles['Normal'], fontSize=10, leftIndent=0.25*inch, firstLineIndent=-0.15*inch, spaceAfter=3, leading=12); styles.add(ref_style)

        story = []

        # 1. Title
        story.append(Paragraph(markdown_to_reportlab(title or "Literature Review"), doc_title_style))

        # 2. Abstract
        story.append(Paragraph("Abstract", styles['SectionHeading']))
        abstract_cleaned = re.sub(r'\s+', ' ', abstract or "No abstract available.").strip()
        abstract_cited = replace_ids_with_numbers(abstract_cleaned, arxiv_ids_ordered, article_metadata)
        abstract_formatted = markdown_to_reportlab(abstract_cited)
        story.append(Paragraph(abstract_formatted, abstract_style))

        # 3. Main body
        section_order = [ 
             ("Introduction", "introduction"),
             ("Key Themes and Trends", "key_themes"),
             ("Methods and Technologies", "methods_technologies"),
             ("Resources and Datasets", "resources_datasets"),
             ("Challenges and Limitations", "challenges_limitations"),
             ("Proposed Research Pipeline", "proposed_pipeline"),
             ("Conclusion", "conclusion")
        ]

        for title_str, section_key in section_order:
            section_content = review_sections.get(section_key)
            if section_content and not section_content.startswith("Error:"):
                story.append(Paragraph(markdown_to_reportlab(title_str), styles['SectionHeading']))
                story.append(Spacer(1, 2)) 

                section_cited = replace_ids_with_numbers(section_content, arxiv_ids_ordered, article_metadata)

                paragraphs_text = re.split(r'\n\n+', section_cited.strip())
                for para_text in paragraphs_text:
                    formatted_para_text = markdown_to_reportlab(para_text.replace('\n', ' ').strip())
                    if formatted_para_text:
                        story.append(Paragraph(formatted_para_text, body_style))
                story.append(Spacer(1, 6)) 
            else:
                 story.append(Paragraph(title_str, styles['SectionHeading']))
                 story.append(Paragraph("<i>Section not generated or contains errors.</i>", body_style))
                 story.append(Spacer(1, 6))

        # 4. References
        story.append(PageBreak()) 
        story.append(Paragraph("References", styles['SectionHeading']))
        if references:
            for i, ref_text in enumerate(references):
                ref_formatted = markdown_to_reportlab(ref_text)
                story.append(Paragraph(f"{i + 1}. {ref_formatted}", ref_style))
        else:
            story.append(Paragraph("No references were processed for the final review.", body_style))

        doc.build(story)
        logging.info(f"PDF report '{filename}' generated successfully.")
        return True

    except Exception as e:
        logging.error(f"Error generating PDF report: {e}", exc_info=True)
        return False


def format_references_list(arxiv_ids: List[str], article_metadata: Dict[str, dict]) -> List[str]:
    """
    Formats references based on collected metadata. Uses arXiv ID number.
    """
    formatted_refs = []
    if not arxiv_ids or not article_metadata:
        return formatted_refs

    logging.info(f"Formatting references for {len(arxiv_ids)} articles...")

    id_to_number_map = {arxiv_id: i + 1 for i, arxiv_id in enumerate(arxiv_ids)} # for fast ID search

    for i, arxiv_id_key in enumerate(arxiv_ids):
        metadata = article_metadata.get(arxiv_id_key)
        if not metadata:
            logging.warning(f"Metadata not found for {arxiv_id_key} during reference formatting.")
            formatted_refs.append(f"[Metadata missing for {arxiv_id_key}]")
            continue

        authors_list = metadata.get('authors', [])
        if len(authors_list) > 3:
            authors_str = f"{authors_list[0]} et al."
        elif authors_list:
            authors_str = ", ".join(authors_list)
        else:
            authors_str = "N.A."

        year = metadata.get('year', 'N.A.')
        title = metadata.get('title', 'No Title Provided')
        title = title.replace('\n', ' ').strip()

        journal_ref = metadata.get('journal_ref')
        doi = metadata.get('doi')
        arxiv_match = re.search(r'([\d.]+v?\d*)$', arxiv_id_key)
        arxiv_num_only = arxiv_match.group(1) if arxiv_match else arxiv_id_key

        source_str = ""
        if journal_ref:
            source_str = f"*{journal_ref}*"
        elif doi:
             source_str = f"DOI: {doi}"
        elif arxiv_num_only:
             source_str = f"arXiv preprint arXiv:{arxiv_num_only}" 
        else:
             source_str = "[Source not specified]"

        ref_string = f"{authors_str} ({year}). {title}. {source_str}."
        formatted_refs.append(ref_string)

    logging.info("Reference formatting complete.")

    return formatted_refs