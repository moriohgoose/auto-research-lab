import logging
import re
from typing import Dict, Callable, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.text_parser import extract_prompt

PredictorFunc = Callable[[str, AutoModelForCausalLM, AutoTokenizer], str]

class ClinicalTox:
    """
    Tool to predict potential clinical trial toxicity for a given SMILES string
    using a pre-loaded TxGemma Predict model.
    """
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompts_json: Dict):
        """Initializes the tool with loaded model, tokenizer, and prompts."""
        self.tool_call_name = "ClinicalToxTool"
        self.task_tdc_name = "ClinTox"
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts_json
        if self.task_tdc_name not in self.prompts: raise KeyError(f"Template '{self.task_tdc_name}' not found.")
        logging.debug(f"{self.tool_call_name} initialized.")

    def use_tool(self, smiles_string: str) -> str:
        """
        Predicts clinical toxicity for the given SMILES string.

        Args:
            smiles_string (str): The SMILES string of the molecule to evaluate.

        Returns:
            str: A human-readable string indicating the toxicity prediction
                 (e.g., "non-toxic", "toxic") or an error message.
        """
        if not smiles_string or not isinstance(smiles_string, str):
            logging.warning(f"{self.tool_call_name} received invalid SMILES input.")
            return "Error: Invalid or empty SMILES string provided."
        try:
            # Format the prompt using the template and input SMILES
            prompt = self.prompts[self.task_tdc_name].replace("{Drug SMILES}", smiles_string)
            # Call the predictor function (which handles the local LLM inference)
            raw_prediction = self.predictor(prompt)
            logging.debug(f"{self.tool_call_name} raw prediction for '{smiles_string}': '{raw_prediction}'")

            # Interpret the raw prediction (usually '(A)' or '(B)')
            if "(A)" in raw_prediction:
                # According to TDC format, (A) usually means negative/no/non-toxic for ClinTox
                return f"Result for {smiles_string}: Predicted to be non-toxic in clinical trials."
            elif "(B)" in raw_prediction:
                # (B) usually means positive/yes/toxic
                return f"Result for {smiles_string}: Predicted to be potentially toxic in clinical trials."
            else:
                # Handle cases where the model output is unexpected
                logging.warning(f"{self.tool_call_name} received unexpected output: {raw_prediction}")
                return f"Prediction ({self.task_tdc_name}): Could not determine toxicity. Raw output: {raw_prediction}"
        except KeyError:
             # Should not happen due to check in __init__, but as safeguard
             logging.error(f"Error using {self.tool_call_name}: Prompt template '{self.task_tdc_name}' missing.")
             return f"Error: Configuration error for {self.tool_name}."
        except Exception as e:
            logging.error(f"Error during {self.tool_call_name} execution for '{smiles_string}': {e}", exc_info=True)
            return f"Error: Exception during {self.tool_name} prediction."

    def tool_is_used(self, query: str) -> bool:
        """Checks if the query string contains the specific call sign for this tool."""
        return f"```{self.tool_call_name}" in query

    def process_query(self, query: str) -> str:
        """Extracts the SMILES string input from the formatted agent query."""
        return extract_prompt(query, word=self.tool_call_name)

    def instructions(self) -> str:
        """Provides instructions for the agent on how to use this tool."""
        return (
            f"=== {self.tool_name} ({self.tool_call_name}) ===\n"
            f"Predicts potential clinical trial toxicity for a given molecule.\n"
            f"Format: ```{self.tool_call_name}\n<SMILES string here>\n```\n"
            f"Returns: Prediction whether the compound is likely 'toxic' or 'non-toxic' in clinical trials."
        )


class BBBPredictTool:
    """
    Tool to predict Blood-Brain Barrier (BBB) permeability for a given SMILES string
    using a pre-loaded TxGemma Predict model.
    """
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompts_json: Dict):
        """Initializes the tool with loaded model, tokenizer, and prompts."""
        self.tool_call_name = "BBBPredictTool"
        self.task_tdc_name = "BBB_Martins"
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts_json
        if self.task_tdc_name not in self.prompts: raise KeyError(f"Template '{self.task_tdc_name}' not found.")
        logging.debug(f"{self.tool_call_name} initialized.")

    def use_tool(self, smiles_string: str) -> str:
        """
        Predicts if the molecule crosses the BBB.

        Args:
            smiles_string (str): The SMILES string of the molecule.

        Returns:
            str: Prediction ("CROSS BBB" or "NOT cross BBB") or an error message.
        """
        if not smiles_string or not isinstance(smiles_string, str):
             logging.warning(f"{self.tool_call_name} received invalid SMILES input.")
             return "Error: Invalid or empty SMILES string provided."
        try:
            prompt = self.prompts[self.task_tdc_name].replace("{Drug SMILES}", smiles_string)
            raw_prediction = self.predictor(prompt)
            logging.debug(f"{self.tool_call_name} raw prediction for '{smiles_string}': '{raw_prediction}'")

            # Interpret the result: (A) does not cross, (B) crosses
            if "(A)" in raw_prediction:
                return f"Result for {smiles_string}: Predicted NOT to cross the BBB."
            elif "(B)" in raw_prediction:
                return f"Result for {smiles_string}: Predicted to cross the BBB."
            else:
                logging.warning(f"{self.tool_call_name} received unexpected output: {raw_prediction}")
                return f"Prediction ({self.task_tdc_name}): Could not determine BBB permeability. Raw output: {raw_prediction}"
        except KeyError:
             logging.error(f"Error using {self.tool_call_name}: Prompt template '{self.task_tdc_name}' missing.")
             return f"Error: Configuration error for {self.tool_name}."
        except Exception as e:
            logging.error(f"Error during {self.tool_call_name} execution for '{smiles_string}': {e}", exc_info=True)
            return f"Error: Exception during {self.tool_name} prediction."

    def tool_is_used(self, query: str) -> bool:
        """Checks if the query string contains the call sign for this tool."""
        return f"```{self.tool_call_name}" in query

    def process_query(self, query: str) -> str:
        """Extracts the SMILES string input."""
        return extract_prompt(query, word=self.tool_call_name)

    def instructions(self) -> str:
        """Provides instructions for the agent."""
        return (
            f"=== {self.tool_name} ({self.tool_call_name}) ===\n"
            f"Predicts if a molecule is likely to cross the Blood-Brain Barrier (BBB).\n"
            f"Format: ```{self.tool_call_name}\n<SMILES string here>\n```\n"
            f"Returns: Prediction 'CROSS BBB' or 'NOT cross BBB'."
        )