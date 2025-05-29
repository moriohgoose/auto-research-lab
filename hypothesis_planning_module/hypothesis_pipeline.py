import logging
import os
import time
from typing import Dict, Optional, Any
from dotenv import load_dotenv

from .hypothesis_agent import GeneratorAgent
from .orchestrator import inference_gemini_orchestrator

from tools.tool_manager import ToolManager
from tools.pubmed_tool import PubMedSearch
from tools.txgemma_predict_tool import ClinicalTox, BBBPredictTool 

from utils.txgemma_loader import load_predict_model_and_tokenizer, load_tdc_prompts
from utils.pdf_generator import save_hypothesis_pdf

import config

load_dotenv()
pubmed_email = os.getenv("MY_EMAIL_FOR_PUBMED")

def run_hypothesis_pipeline(
    user_topic: str,
    user_focus: Optional[str],
    user_goal: str,
    user_context: Optional[str] = None,
    user_email_for_pubmed: str = pubmed_email
) -> Dict[str, Any]:
    """
    Executes the full pipeline for hypothesis generation based on user input.

    Args:
        user_topic: The main research area/disease.
        user_focus: Specific molecule, pathway, etc. (optional).
        user_goal: The objective for the agent (e.g., "propose novel hypothesis").
        user_context: Additional constraints or context (optional).
        user_email_for_pubmed: User's email required for PubMed API calls.

    Returns:
        A dictionary containing the results:
        {
            "final_agent_response": str, # The final text output from the agent
            "generated_pdf_path": Optional[str], # Path to the saved PDF report, or None
            "error": Optional[str] # Error message if the pipeline failed, else None
        }
    """
    pipeline_start_time = time.time()
    logging.info("--- Starting Hypothesis Generation Pipeline ---")
    pipeline_result = {
        "final_agent_response": None,
        "generated_pdf_path": None,
        "error": None
    }

    try:
        # --- 1. Load Local LLM Model (Predictor) ---
        predict_model, predict_tokenizer = load_predict_model_and_tokenizer(
            model_id=config.TXGEMMA_PREDICT_MODEL_ID,
            use_quantization=config.USE_QUANTIZATION
        )
        if predict_model is None or predict_tokenizer is None:
             pipeline_result["error"] = "Failed to load local predictor model."
             raise RuntimeError(pipeline_result["error"]) # Stop process

        # --- 2. Load TDC Prompts ---
        tdc_prompts = load_tdc_prompts(
            model_id_for_repo=config.TXGEMMA_PREDICT_MODEL_ID,
            local_prompts_path=config.TDC_PROMPTS_FILENAME
        )
        if tdc_prompts is None:
            pipeline_result["error"] = "Failed to load TDC prompts."
            raise RuntimeError(pipeline_result["error"])        

        # --- 3. Define Local Predictor Function ---
        def _local_predict_func(prompt: str) -> str:
            if predict_model is None or predict_tokenizer is None: return "Error: Predictor not loaded."
            if tdc_prompts is None: return "Error: TDC prompts not loaded."
            try:
                inputs = predict_tokenizer(prompt, return_tensors="pt").to(predict_model.device)
                outputs = predict_model.generate(**inputs, max_new_tokens=10, temperature=0.1, do_sample=False)
                input_len = inputs['input_ids'].shape[1]
                return predict_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
            except Exception as e: logging.error(f"Error in _local_predict_func: {e}"); return "Prediction error."

        # --- 4. Initialize Tools ---
        logging.info("Initializing tools...")
        active_tools = []
        # PubMed
        try:
            tool = PubMedSearch(email=user_email_for_pubmed)
            active_tools.append(tool); logging.info(f"'{tool.tool_call_name}' tool created.")
        except Exception as e: logging.error(f"Failed to create PubMedSearch: {e}")
        # ClinTox
        try:
            tool = ClinicalTox(_local_predict_func, tdc_prompts)
            active_tools.append(tool); logging.info(f"'{tool.tool_call_name}' tool created.")
        except KeyError as ke: logging.warning(f"ClinicalTox not created: Template '{ke}' not found.")
        except Exception as e: logging.error(f"Failed to create ClinicalTox: {e}")
        # BBB
        try:
            tool = BBBPredictTool(_local_predict_func, tdc_prompts)
            active_tools.append(tool); logging.info(f"'{tool.tool_call_name}' tool created.")
        except KeyError as ke: logging.warning(f"BBBPredictTool not created: Template '{ke}' not found.")
        except Exception as e: logging.error(f"Failed to create BBBPredictTool: {e}")
        # --- Can add other tools here ---

        if not active_tools:
            pipeline_result["error"] = "Failed to initialize any tools."
            raise RuntimeError(pipeline_result["error"])

        tool_manager = ToolManager(active_tools)

        # --- 5. Format Agent Question ---
        logging.info("Formatting agent question...")
        agent_question_parts = [ f"Research Area: {user_topic}." ]
        if user_focus: agent_question_parts.append(f"Specific Focus: {user_focus}.")
        agent_question_parts.append(f"Task Goal: {user_goal}.")
        if user_context: agent_question_parts.append(f"Additional Context/Constraints: {user_context}.")
        # Append standard instructions
        agent_question_parts.extend([
            "\nPlease perform the requested task step-by-step.",
            "If proposing a hypothesis, ensure it is novel and testable.",
            "If outlining a research plan, include key steps like model systems and assays.",
            "Use available tools to gather necessary information or check assumptions."
        ])
        agent_question = "\n".join(agent_question_parts)
        logging.info(f"Formatted Agent Question:\n{agent_question}\n" + "-"*60)

        # --- 6. Initialize and Run Agent ---
        logging.info("Initializing GeneratorAgent...")
        generator_agent = GeneratorAgent(
            tool_manager=tool_manager,
            orchestrator_func=inference_gemini_orchestrator, 
            num_steps=10 
        )
        logging.info("Running agent steps...")
        final_response = generator_agent.step(agent_question)
        pipeline_result["final_agent_response"] = final_response
        logging.info("Agent finished execution.")

        # --- 7. Save PDF Output ---
        if final_response and "error" not in final_response:
            # Create dir
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            pdf_filename_base = f"agent_response_{user_topic.replace(' ','_')[:20]}_{int(time.time())}.pdf"
            pdf_filepath = os.path.join(config.OUTPUT_DIR, pdf_filename_base)
            try:
                pdf_saved = save_hypothesis_pdf(final_response, pdf_filepath)
                if pdf_saved:
                    pipeline_result["generated_pdf_path"] = pdf_filepath
                else:
                    pipeline_result["error"] = (pipeline_result.get("error") or "") + " Failed to save PDF."
            except NameError:
                logging.error("Function 'save_hypothesis_pdf' not found/imported.")
                pipeline_result["error"] = (pipeline_result.get("error") or "") + " PDF saving function missing."
            except Exception as e_pdf:
                logging.error(f"Error saving PDF: {e_pdf}")
                pipeline_result["error"] = (pipeline_result.get("error") or "") + f" PDF saving error: {e_pdf}"
        else:
            logging.warning("Agent response contains error or is empty, PDF will not be generated.")
            if not pipeline_result["error"]: 
                 pipeline_result["error"] = "Agent response contained an error."


    except Exception as e:
        logging.error(f"Critical error in hypothesis pipeline: {e}", exc_info=True)
        pipeline_result["error"] = f"Critical pipeline error: {e}"

    pipeline_end_time = time.time()
    logging.info(f"--- Hypothesis Generation Pipeline Finished. Duration: {pipeline_end_time - pipeline_start_time:.2f} sec. ---")
    return pipeline_result