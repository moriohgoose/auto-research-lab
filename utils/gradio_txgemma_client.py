import os
import logging
from gradio_client import Client
from dotenv import load_dotenv
import time
from typing import Optional, Tuple, Dict, Any

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN") 
GRADIO_SPACE_ID = "joshuaberkowitzus/txgemma-2b-predict-demo" 

logger = logging.getLogger(__name__)


def predict_with_gradio_client(prompt_text: str, 
                               max_new_tokens_val: int = 100, 
                               temperature_val: float = 0.7,
                               max_retries: int = 3,
                               wait_time_sec: int = 5) -> Optional[str]:
    """
    Uses gradio_client to make a prediction to the specified TxGemma Gradio Space.

    Args:
        prompt_text (str): The prompt to send to the model.
        max_new_tokens_val (int): Max new tokens for generation.
        temperature_val (float): Temperature for generation.
        max_retries (int): Number of retries in case of issues.
        wait_time_sec (int): Wait time between retries.

    Returns:
        Optional[str]: The generated text from the model or None if an error occurs.
    """
    # global _gradio_client_instance # Если бы использовали глобальный клиент
    # if _gradio_client_instance is None:
    #     try:
    #         logger.info(f"Initializing Gradio Client for Space: {GRADIO_SPACE_ID}")
    #         _gradio_client_instance = Client(GRADIO_SPACE_ID, hf_token=HF_TOKEN) # Передаем токен, если нужно
    #         logger.info("Gradio Client initialized.")
    #     except Exception as e_init:
    #         logger.error(f"Failed to initialize Gradio Client for {GRADIO_SPACE_ID}: {e_init}", exc_info=True)
    #         return None # Не удалось инициализировать клиент

    client = None # Инициализируем здесь для области видимости
    for attempt in range(max_retries):
        try:
            if client is None: # Инициализируем клиент при первой попытке или если предыдущая не удалась
                logger.info(f"Attempt {attempt + 1}/{max_retries}: Initializing Gradio Client for Space: {GRADIO_SPACE_ID}")
                client = Client(GRADIO_SPACE_ID, hf_token=HF_TOKEN) # Попробуйте без hf_token сначала, если Space публичный
                logger.info("Gradio Client initialized.")

            logger.info(f"Attempt {attempt + 1}/{max_retries}: Sending prediction request to Gradio Space '{GRADIO_SPACE_ID}' with api_name='/predict'")
            logger.debug(f"Prompt: {prompt_text[:100]}..., max_new_tokens: {max_new_tokens_val}, temperature: {temperature_val}")
            
            result = client.predict(
                prompt=prompt_text,	            # str in 'Your Prompt' Textbox component
                max_new_tokens=float(max_new_tokens_val), # float in 'Max New Tokens' Slider component - передаем как float, как в док.
                temperature=float(temperature_val),	    # float in 'Temperature' Slider component - передаем как float
                api_name="/predict"
            )
            # Результат должен быть строкой
            if isinstance(result, str):
                logger.info(f"Prediction successful. Result: {result[:100]}...")
                return result.strip()
            else:
                logger.warning(f"Received unexpected result type from Gradio client: {type(result)}. Result: {result}")
                # Попробуем извлечь, если это кортеж/список с одним элементом (иногда Gradio так делает)
                if isinstance(result, (list, tuple)) and len(result) > 0 and isinstance(result[0], str):
                    logger.info(f"Extracted string from result sequence: {result[0][:100]}...")
                    return result[0].strip()
                # Если не строка, считаем ошибкой для данного случая
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {wait_time_sec}s before retry due to unexpected result type...")
                    time.sleep(wait_time_sec)
                    client = None # Пересоздадим клиент на следующей итерации
                    continue
                else:
                    logger.error("Max retries reached with unexpected result type.")
                    return None # Или вернуть сообщение об ошибке

        except Exception as e: # Общее исключение
            logger.warning(f"Error predicting with Gradio client (Attempt {attempt + 1}/{max_retries}): {e}", exc_info=True)
        
        if attempt < max_retries - 1:
            logger.info(f"Waiting {wait_time_sec}s before retry...")
            time.sleep(wait_time_sec)
            client = None # Попробуем пересоздать клиент на следующей итерации
        else:
            logger.error(f"Max retries reached for Gradio prediction to {GRADIO_SPACE_ID}.")
            return None
            
    return None