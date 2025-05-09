import os
import torch
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import re
import logging
import time
from typing import List, Dict, Optional, Tuple
from huggingface_hub import hf_hub_download
from Bio import Medline, Entrez

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4


from tools.tool_manager import ToolManager 
from utils.text_parser import extract_prompt
from prompts import hypothesis_prompts as hp


STATUS_READY = "READY_TO_ANSWER"
STATUS_NOT_READY = "NOT_READY"
ACTION_REFINE_INTERNALLY = "Action: Refine internally"

class GeneratorAgent:
    """
    An agent that uses a large language model (LLM) as an orchestrator
    to perform complex tasks by iteratively thinking, choosing actions (tools),
    and observing results. It is designed for hypothesis generation and
    research planning in the therapeutics domain.

    Attributes:
        num_steps (int): Maximum number of Thought-Action-Observation cycles.
        orchestrator_func (callable): Function to call the orchestrator LLM API.
        tool_manager (ToolManager): Manager for available tools.
        thoughts (List[str]): History of agent's thoughts.
        actions (List[str]): History of agent's actions (tool calls or internal refinement).
        observations (List[str]): History of observations received from tools or actions.
        curr_steps (int): Current step number in the cycle.
    """
    def __init__(self, tool_manager: ToolManager, orchestrator_func, num_steps=8):
        """
        Initializes the GeneratorAgent.

        Args:
            tool_manager (ToolManager): An instance of ToolManager that handles available tools.
            orchestrator_func (callable): A function that takes a prompt and system_prompt (optional)
                                          and returns the LLM's text response. This is typically
                                          a call to an LLM API like Gemini.
            num_steps (int): The maximum number of Thought-Action-Observation cycles the agent
                             can perform before generating a final answer.
        """
        self.num_steps: int = num_steps
        self.orchestrator_func: callable = orchestrator_func
        self.tool_manager: 'ToolManager' = tool_manager # Forward reference for type hint if ToolManager is in another file
        self.thoughts: List[str] = []
        self.actions: List[str] = []
        self.observations: List[str] = []
        self.curr_steps: int = 0
        logging.info(f"GeneratorAgent initialized with {num_steps} steps.")
        self.reset() # Initial reset of state

    def reset(self):
        """Resets the agent's internal state (thoughts, actions, observations, step count)."""
        self.curr_steps = 0
        self.thoughts = []
        self.actions = []
        self.observations = []
        logging.info("Agent state has been reset.")

    def _system_prompt(self, use_tools: bool = True, is_final_answer_step: bool = False) -> str:
        """
        Generates the system prompt for the orchestrator LLM based on the current state.

        Args:
            use_tools (bool): If True, includes instructions about available tools.
                              If False, instructs the LLM to focus on reasoning or final answer.
            is_final_answer_step (bool): If True, indicates this is the final step to generate
                                         the conclusive answer.

        Returns:
            str: The system prompt string.
        """
        base_role = hp.BASE_SYSTEM_ROLE
        available_tool_names = [t.tool_call_name for t in self.tool_manager.toolset if hasattr(t, 'tool_call_name')]
        available_tool_names_str = ", ".join(available_tool_names) if available_tool_names else "None"

        if is_final_answer_step:
            return hp.FINAL_ANSWER_SYSTEM_PROMPT_TEMPLATE.format(
                base_role=base_role,
                current_step=self.curr_steps # Note: curr_steps is 0-indexed, so it represents completed steps
            )
        else:
            step_info_params = {
                "max_steps": self.num_steps,
                "current_step": self.curr_steps + 1,
                "cycles_left": self.num_steps - self.curr_steps - 1,
            }
            if use_tools:
                # Use the action system prompt template
                return hp.ACTION_SYSTEM_PROMPT_TEMPLATE.format(
                    base_role=base_role,
                    tool_info=self.tool_manager.tool_instructions(), # Get full instructions
                    **step_info_params
                )
            else: # Thought step
                # Use the think system prompt template
                return hp.THINK_SYSTEM_PROMPT_TEMPLATE.format(
                    base_role=base_role,
                    available_tool_names_str=available_tool_names_str,
                    **step_info_params
                )
            

    def _prior_information(self, original_query: str) -> str:
        """
        Constructs a string representing the history of the agent's interaction so far,
        including the original user query.

        Args:
            original_query (str): The initial question posed by the user.

        Returns:
            str: A formatted string of the interaction history.
        """
        history_str = f"Original User Question: {original_query}\n\n--- History ---\n"
        if not self.thoughts:
            history_str += "(No history yet)\n"
        else:
            # Iterate up to current step to build history
            for i in range(self.curr_steps):
                history_str += f"Cycle {i + 1}:\nThought: {self.thoughts[i]}\n"
                if i < len(self.actions): # Action might not exist for the current thought yet
                    history_str += f"Action: {self.actions[i]}\n"
                if i < len(self.observations): # Observation might not exist yet
                    history_str += f"Observation: {self.observations[i]}\n"
                history_str += "---\n"
        return history_str

    def _extract_structured_thought(self, thought_text: str) -> Dict[str, str]:
        """
        Parses the raw thought text from the LLM to separate the
        analysis/plan from the readiness status.

        Args:
            thought_text (str): The raw text generated by the LLM for its thought.

        Returns:
            Dict[str, str]: A dictionary with "analysis_plan" and "readiness" status
                            (either "READY_TO_ANSWER" or "NOT_READY").
        """
        readiness_match = re.search(r"Readiness:\s*(READY_TO_ANSWER|NOT_READY)", thought_text, re.IGNORECASE)
        readiness = readiness_match.group(1).upper() if readiness_match else STATUS_NOT_READY
        # Remove the "Readiness: ..." part from the thought text
        analysis_plan = re.sub(r"Readiness:\s*(READY_TO_ANSWER|NOT_READY)", "", thought_text, flags=re.IGNORECASE).strip()
        return {"analysis_plan": analysis_plan, "readiness": readiness}

    def _generate_final_answer(self, original_query: str) -> str:
        """
        Generates the final comprehensive answer once the agent is ready or
        has reached the maximum number of steps.

        Args:
            original_query (str): The initial question posed by the user.

        Returns:
            str: The final answer formulated by the orchestrator LLM.
        """
        logging.info("Generating final answer...")
        final_sys_prompt = self._system_prompt(is_final_answer_step=True)
        full_history = self._prior_information(original_query) # Get history up to the current completed step
        if self.thoughts: # Add the very last thought that triggered finalization
            full_history += f"Final Thought Before Answer: {self.thoughts[-1]}\n---\n"

        # Prompt for the orchestrator to synthesize everything
        final_user_prompt = hp.FINAL_ANSWER_USER_PROMPT_TEMPLATE.format(
            history=full_history,
            original_query=original_query
        )
        final_answer_text = self.orchestrator_func(prompt=final_user_prompt, system_prompt=final_sys_prompt)
        logging.info("Final answer generated.")
        
        return final_answer_text

    def step(self, original_query: str) -> str:
        """
        Executes one full Thought-Action-Observation cycle or proceeds to final answer.

        Args:
            original_query (str): The user's question for the agent to address.

        Returns:
            str: The agent's final answer after completing its cycles or an error message.
        """
        self.reset() # Ensure a clean state for each new query
        logging.info(f"GeneratorAgent starting work on query: {original_query[:150]}...")

        for i in range(self.num_steps):
            self.curr_steps = i # curr_steps is 0-indexed for internal logic
            current_history = self._prior_information(original_query)

            # --- 1. Thought Generation ---
            # System prompt for the thought step (no tools allowed for thought generation)
            think_sys_prompt = self._system_prompt(use_tools=False, is_final_answer_step=False)
            think_user_prompt = hp.THOUGHT_USER_PROMPT_TEMPLATE.format(history=current_history)
            raw_thought = self.orchestrator_func(prompt=think_user_prompt, system_prompt=think_sys_prompt)

            structured_thought = self._extract_structured_thought(raw_thought)
            thought_analysis_plan = structured_thought["analysis_plan"]
            readiness_status = structured_thought["readiness"]

            self.thoughts.append(raw_thought) # Store the full raw thought for context
            logging.info(f"Cycle {i+1}/{self.num_steps} - Thought: {thought_analysis_plan}")
            logging.info(f"Cycle {i+1}/{self.num_steps} - Readiness: {readiness_status}")

            # --- Check for Readiness or Final Step ---
            if readiness_status == STATUS_READY or i == self.num_steps - 1:
                if i == self.num_steps - 1 and readiness_status != STATUS_READY:
                    logging.warning("Max steps reached, proceeding to final answer without READY signal.")
                else:
                    logging.info("Agent signaled ready or last step reached. Proceeding to finalization...")
                return self._generate_final_answer(original_query)

            # --- 2. Action Generation ---
            # System prompt for the action step (tools are allowed)
            action_sys_prompt = self._system_prompt(use_tools=True)

            # Get available tool names for the prompt
            available_tool_names_list = [t.tool_call_name for t in self.tool_manager.toolset if hasattr(t, 'tool_call_name')]

            action_user_prompt = hp.ACTION_USER_PROMPT_TEMPLATE.format(
                history=current_history, # Provide history again for action context
                thought_analysis_plan=thought_analysis_plan, # Provide the plan part of the thought
                available_tool_names_list=available_tool_names_list # Provide list of tools
            )
            action_text = self.orchestrator_func(prompt=action_user_prompt, system_prompt=action_sys_prompt)
            self.actions.append(action_text) # Store the raw action
            logging.info(f"Cycle {i+1}/{self.num_steps} - Action: {action_text}")

            # --- Intercept if LLM tries to give final answer as an action ---
            if action_text.strip().startswith("Final Answer:"):
                logging.warning("Agent attempted to give final answer as an action. Proceeding to finalization.")
                return self._generate_final_answer(original_query)

            # --- 3. Observation (Tool Execution) ---
            if action_text.strip().lower() == ACTION_REFINE_INTERNALLY.lower(): 
                observation_text = "Internal refinement based on previous observations/thoughts."
                logging.info(f"Cycle {i+1}/{self.num_steps} - Observation: {observation_text}")            
            else:
                # Execute the tool call
                tool_start_time = time.time()
                observation_text = self.tool_manager.use_tool(action_text.strip())
                tool_end_time = time.time()
                log_obs = observation_text[:500] + ('...' if len(observation_text) > 500 else '')
                logging.info(f"Cycle {i+1}/{self.num_steps} - Observation ({tool_end_time - tool_start_time:.2f} sec): {log_obs}")
                if observation_text.startswith("Error"):
                    logging.error(f"Tool returned an error: {observation_text}")
                    # Error will be available in the history for the next thought cycle
            
            self.observations.append(observation_text) # Store observation (or error message)

        # Fallback if loop finishes unexpectedly (shouldn't happen with the check above)
        logging.error("Agent cycle completed without reaching a final answer state.");
        return "Error: Agent exceeded maximum steps without providing a final answer."