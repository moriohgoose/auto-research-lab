import os

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "output")
GEMINI_API_MODEL_NAME = "gemini-2.0-flash" 

REVIEW_MAX_ITERATIONS = 5

TXGEMMA_PREDICT_MODEL_ID = "google/txgemma-2b-predict"
USE_QUANTIZATION = True
TDC_PROMPTS_FILENAME = "prompts/tdc_prompts.json"