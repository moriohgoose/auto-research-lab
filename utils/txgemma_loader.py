import os
import torch
import json
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import hf_hub_download
from typing import Optional, Tuple, Dict, Any


def load_predict_model_and_tokenizer(model_id: str, use_quantization: bool = True, force_reload: bool = False) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
    """
    Loads the specified TxGemma predictor model and its tokenizer from Hugging Face.
    Applies quantization if requested and GPU is available. Caches the loaded model
    and tokenizer globally within this module to avoid redundant loading.

    Args:
        model_id (str): The Hugging Face model ID (e.g., "google/txgemma-2b-predict").
        use_quantization (bool): Whether to attempt loading with 4-bit quantization.
        force_reload (bool): If True, forces reloading even if model/tokenizer are cached.

    Returns:
        Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
            A tuple containing the loaded model and tokenizer, or (None, None) if loading fails.
    """
    logging.info(f"Attempting to load predictor model and tokenizer: {model_id} (Quantization: {use_quantization})")

    # --- Model Loading Logic ---
    quantization_cfg = None
    if use_quantization:
        if torch.cuda.is_available():
            logging.info("GPU available. Configuring 4-bit quantization.")
            quantization_cfg = BitsAndBytesConfig(load_in_4bit=True)
        else:
            logging.warning("GPU not available! Quantization disabled.")
            use_quantization = False # Ensure flag reflects reality

    model = None
    tokenizer = None
    try:
        logging.info(f"Loading tokenizer for {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info("Tokenizer loaded.")

        logging.info(f"Loading model {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_cfg,
            device_map="auto" # Automatically use GPU if available
        )
        logging.info(f"Model {model_id} loaded successfully onto device: {model.device}")
        return model, tokenizer

    except Exception as e:
        logging.error(f"CRITICAL ERROR loading predictor model '{model_id}': {e}", exc_info=True)
        return None, None


def load_tdc_prompts(model_id_for_repo: str, local_prompts_path: str = "tdc_prompts.json", force_reload: bool = False) -> Optional[Dict]:
    """
    Loads the TDC prompt templates JSON file, downloading if necessary.
    Caches the loaded prompts globally.

    Args:
        model_id_for_repo (str): The HF repo ID used to locate the file (e.g., "google/txgemma-2b-predict").
        filename (str): The name of the JSON file.
        force_reload (bool): If True, forces reloading even if prompts are cached.

    Returns:
        Optional[Dict]: The loaded dictionary of prompts, or None if loading fails.
    """
    logging.info(f"Attempting to load TDC prompts file (path hint: {local_prompts_path})...")

    tdc_prompts = None
    # --- Attempt 1: Load Locally ---
    try:
        if os.path.exists(local_prompts_path):
            with open(local_prompts_path, "r", encoding='utf-8') as f:
                tdc_prompts = json.load(f)
            logging.info(f"TDC prompts file loaded successfully from local path: {local_prompts_path}")
            return tdc_prompts
        else:
            logging.warning(f"Local TDC prompts file not found at: {local_prompts_path}. Attempting download...")
    except Exception as e_local:
        logging.warning(f"Error loading local TDC prompts file '{local_prompts_path}': {e_local}. Attempting download...")

    # --- Attempt 2: Download from Hugging Face Hub ---
    try:
        logging.info(f"Attempting to download {os.path.basename(local_prompts_path)} from repo '{model_id_for_repo}'...")
        # Extract just the filename for hf_hub_download
        filename_only = os.path.basename(local_prompts_path)
        tdc_prompts_filepath = hf_hub_download(repo_id=model_id_for_repo, filename=filename_only)
        with open(tdc_prompts_filepath, "r", encoding='utf-8') as f:
            tdc_prompts = json.load(f)
        logging.info(f"TDC prompts file '{filename_only}' downloaded and loaded successfully.")
        return tdc_prompts
    except Exception as e_hf:
        logging.error(f"Failed to download {filename_only} from Hugging Face Hub after local attempt failed: {e_hf}")
        return None


def local_txgemma_predict(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer
    ) -> str:
    """
    Performs prediction using the PASSED predictor model and tokenizer.

    Args:
        prompt: The fully formatted prompt ready for the model.
        model: The loaded predictor model instance.
        tokenizer: The loaded tokenizer instance.

    Returns:
        The model's prediction or an error message.
    """
    if model is None or tokenizer is None:
        logging.error("local_txgemma_predict called with None model or tokenizer.")
        return "Error: Predictor model/tokenizer not provided."

    try:
        logging.debug(f"local_txgemma_predict: Sending prompt (start): {prompt[:100]}...")
        # Ensure tensors are on the same device as the model
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        # Generate response
        outputs = model.generate(**input_ids, max_new_tokens=10, temperature=0.1, do_sample=False)
        # Decode only the generated part
        input_length = input_ids['input_ids'].shape[1]
        output_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        logging.debug(f"local_txgemma_predict: Raw output: '{output_text}'")
        return output_text.strip()
    except Exception as e:
        logging.error(f"Error during local_txgemma_predict inference: {e}", exc_info=True)
        return f"Error during prediction: {e}"