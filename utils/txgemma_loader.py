import os
import json
import logging
from huggingface_hub import hf_hub_download, login as hf_login
from typing import Optional, Tuple, Dict, Any
import streamlit as st

logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner="Loading TxGemma prediction model...")
def load_predict_model_and_tokenizer_cached(model_id: str, use_quantization: bool = True) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Loads the specified TxGemma predictor model and its tokenizer from Hugging Face.
    Applies quantization if requested and GPU is available.
    This function is cached by Streamlit to avoid redundant loading.
    Heavy imports (torch, transformers) are done inside this function.

    Args:
        model_id (str): The Hugging Face model ID (e.g., "google/txgemma-2b-predict").
        use_quantization (bool): Whether to attempt loading with 4-bit quantization.

    Returns:
        Tuple[Optional[Any], Optional[Any]]:
            A tuple containing the loaded model and tokenizer, or (None, None) if loading fails.
            The types are 'Any' before the actual import of transformer classes.
    """
    # --- 1. Deferred Heavy Imports ---
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    logger.info(f"Attempting to load (or retrieve from cache) predictor model and tokenizer: {model_id} (Quantization: {use_quantization})")

    # --- 2. Hugging Face Hub Login (Optional, based on HF_TOKEN) ---
    # This will only be attempted during the first actual execution of this cached function.
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            hf_login(token=hf_token, add_to_git_credential=False)
            logger.info("Successfully logged in to Hugging Face Hub using HF_TOKEN for model/tokenizer loading.")
        except Exception as e_login:
            logger.warning(f"Failed to login to Hugging Face Hub with HF_TOKEN: {e_login}. Assuming model is public or accessible without login.")
    # else: # No need to log if token is not set, from_pretrained will try anonymous access
        # logger.info("HF_TOKEN not set for model loading. Assuming model is public or accessible without login.")

    # --- 3. Model Loading Logic ---
    quantization_cfg = None
    if use_quantization:
        if torch.cuda.is_available():
            logger.info("GPU available. Configuring 4-bit quantization for the model.")
            quantization_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                # bnb_4bit_quant_type="nf4", # Optional: for more precise quantization
                # bnb_4bit_compute_dtype=torch.bfloat16, # Optional: for faster computation
                # bnb_4bit_use_double_quant=True, # Optional: for nested quantization
            )
        else:
            logger.warning("GPU not available! Quantization will be disabled for the model.")
            # Quantization is implicitly disabled as quantization_cfg remains None

    model = None
    tokenizer = None
    try:
        logger.info(f"Loading tokenizer for {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token) # Pass token if available
        logger.info(f"Tokenizer for {model_id} loaded successfully.")

        logger.info(f"Loading model {model_id} with quantization_config={quantization_cfg is not None}...")
        model_load_kwargs = {
            "quantization_config": quantization_cfg,
            "device_map": "auto",  # Handles multi-GPU or CPU fallback if GPU is not enough/available
            "torch_dtype": torch.bfloat16, # Recommended for Gemma models, esp. with quantization
            "token": hf_token # Pass token if available
        }
        # Remove kwargs that are None to avoid passing them to from_pretrained
        model_load_kwargs = {k: v for k, v in model_load_kwargs.items() if v is not None}

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_load_kwargs
        )
        logger.info(f"Model {model_id} loaded successfully. Current device: {model.device}")
        return model, tokenizer

    except ImportError as e_imp:
        logger.error(f"ImportError during model loading. Ensure torch, transformers, bitsandbytes, and accelerate are installed: {e_imp}", exc_info=True)
        st.error(f"Library import error: {e_imp}. Please check your environment.") # Show error in Streamlit UI
        return None, None
    except Exception as e:
        logger.error(f"CRITICAL ERROR loading predictor model '{model_id}': {e}", exc_info=True)
        st.error(f"Failed to load model {model_id}: {e}") # Show error in Streamlit UI
        return None, None

# Wrapper function to handle force_reload for the cached model/tokenizer loader
def load_predict_model_and_tokenizer(model_id: str, use_quantization: bool = True, force_reload: bool = False) -> Tuple[Optional[Any], Optional[Any]]:
    if force_reload:
        logger.warning("`force_reload=True` was called for `load_predict_model_and_tokenizer`.")
        logger.warning("For resources cached with `@st.cache_resource`, explicit clearing of specific arguments is not straightforward.")
        logger.warning("To truly force a reload of the model, you might need to restart the Streamlit server or implement custom cache invalidation.")
        # For now, we'll just call the cached function. If it's already cached, it will return the cached version.
        # A more advanced solution would involve a custom caching mechanism or modifying Streamlit's cache.
    return load_predict_model_and_tokenizer_cached(model_id, use_quantization)


@st.cache_data(show_spinner="Loading TDC prompts...")
def load_tdc_prompts_cached(model_id_for_repo: str, local_prompts_path: str) -> Optional[Dict]:
    """
    Loads the TDC prompt templates JSON file, trying local path first, then Hugging Face Hub.
    This function is cached by Streamlit.
    """
    logger.info(f"Attempting to load (or retrieve from cache) TDC prompts. Local path hint: '{local_prompts_path}'")
    
    # --- Attempt 1: Load Locally ---
    if os.path.exists(local_prompts_path):
        try:
            with open(local_prompts_path, "r", encoding='utf-8') as f:
                prompts = json.load(f)
            logger.info(f"TDC prompts loaded successfully from local file: {local_prompts_path}")
            return prompts
        except Exception as e_local:
            logger.warning(f"Error loading local TDC prompts file '{local_prompts_path}': {e_local}. Will attempt download.")
    else:
        logger.warning(f"Local TDC prompts file not found at: {local_prompts_path}. Attempting download...")

    # --- Attempt 2: Download from Hugging Face Hub ---
    try:
        filename_only = os.path.basename(local_prompts_path)
        logger.info(f"Attempting to download '{filename_only}' from Hugging Face Hub repo '{model_id_for_repo}'...")
        
        hf_token = os.getenv("HF_TOKEN") # Get token for download if needed
        # hf_login is not strictly necessary here if token is passed to hf_hub_download
        # and permissions are for reading.

        downloaded_filepath = hf_hub_download(
            repo_id=model_id_for_repo,
            filename=filename_only,
            token=hf_token if hf_token else None # Pass token if available
        )
        with open(downloaded_filepath, "r", encoding='utf-8') as f:
            prompts = json.load(f)
        logger.info(f"TDC prompts file '{filename_only}' downloaded from repo '{model_id_for_repo}' and loaded successfully.")
        
        # Optional: Save the downloaded file locally for future use
        try:
            os.makedirs(os.path.dirname(local_prompts_path), exist_ok=True)
            with open(local_prompts_path, 'w', encoding='utf-8') as f_save:
                json.dump(prompts, f_save, indent=4)
            logger.info(f"Saved downloaded TDC prompts to local path: {local_prompts_path}")
        except Exception as e_save:
            logger.warning(f"Could not save downloaded TDC prompts locally to '{local_prompts_path}': {e_save}")
            
        return prompts
    except Exception as e_hf:
        logger.error(f"Failed to download or load '{filename_only}' from Hugging Face Hub (repo: '{model_id_for_repo}'): {e_hf}", exc_info=True)
        st.error(f"Failed to load TDC prompts: {e_hf}") # Show error in Streamlit UI
        return None
    

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


def local_txgemma_predict(prompt: str, model: Any, tokenizer: Any) -> str:
    """
    Performs prediction using the PASSED (and already loaded) predictor model and tokenizer.

    Args:
        prompt: The fully formatted prompt ready for the model.
        model: The loaded predictor model instance (type Any due to deferred import).
        tokenizer: The loaded tokenizer instance (type Any due to deferred import).

    Returns:
        The model's prediction (str) or an error message (str).
    """
    if model is None or tokenizer is None:
        logger.error("local_txgemma_predict called with None model or tokenizer.")
        return "Error: Predictor model or tokenizer not provided to prediction function."

    try:
        logger.debug(f"local_txgemma_predict: Sending prompt (first 100 chars): '{prompt[:100]}...'")
        
        # Tokenize the input prompt
        # No explicit torch import needed here as tokenizer and model objects handle tensor operations internally
        input_data = tokenizer(prompt, return_tensors="pt")
        
        # Ensure input tensors are on the same device as the model
        # This is usually handled by device_map="auto" during model loading,
        # but this ensures compatibility if inputs were somehow on CPU.
        if hasattr(model, 'device') and model.device is not None: # Check if model has a device attribute
            if hasattr(input_data, 'to'):
                input_data = input_data.to(model.device)
            elif isinstance(input_data, dict): # If tokenizer returns a dict of tensors
                for key in input_data:
                    if hasattr(input_data[key], 'to'):
                        input_data[key] = input_data[key].to(model.device)
        else:
            logger.warning("Model device not found or is None, cannot move input tensors to model device.")


        # Generate response from the model
        outputs = model.generate(
            **input_data, 
            max_new_tokens=10, # Keep it small for simple property predictions
            temperature=0.1,   # Low temperature for more deterministic output
            do_sample=False    # Disable sampling for deterministic output
        )
        
        # Decode only the newly generated tokens, skipping special tokens
        input_length = 0
        if 'input_ids' in input_data and hasattr(input_data['input_ids'], 'shape'):
            input_length = input_data['input_ids'].shape[1]
        elif hasattr(input_data, 'shape'): # if tokenizer returned a single tensor
            input_length = input_data.shape[1]
            
        output_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        logger.debug(f"local_txgemma_predict: Raw model output: '{output_text}'")
        return output_text.strip()

    except Exception as e:
        logger.error(f"Error during local_txgemma_predict inference: {e}", exc_info=True)
        return f"Error during prediction: {e}"