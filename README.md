# Auto Research Lab: Literature Review & Hypothesis Generation

## 1. Overview

This project provides an AI-powered pipeline utilizing Large Language Models (LLMs) to assist researchers in two key areas:

1.  **Automated Literature Review Generation:** Discovers relevant academic papers (currently via arXiv), analyzes their content, synthesizes findings into a structured review, and iteratively refines the draft based on AI critique.
2.  **Hypothesis & Research Plan Generation:** Leverages an agentic system (using Google Gemini as an orchestrator and local TxGemma models as tools) to propose novel, testable hypotheses based on user-defined topics/focus areas and generate detailed preclinical research plans.

The system uses Google's Gemini API (via `google-generativeai`) as the primary orchestrator/generator and specialized models like TxGemma Predict locally for specific tasks. It aims to streamline research workflows by automating information gathering, synthesis, and ideation.

The project now includes an interactive web interface built with **Streamlit**, allowing users to easily select modes, input parameters, and view results directly in their browser.

## 2. Key Features

*   **Dual Modes:** Choose between generating a comprehensive literature review or a novel hypothesis with a research plan.
*   **Interactive Web Interface (Streamlit):** User-friendly GUI for running pipelines and viewing outputs.
*   **Modular Design:** Code is organized into distinct modules for configuration, utilities, tools, agent logic, and pipelines.
*   **LLM-Powered:**
    *   **Literature Review:** Uses Gemini for keyword generation, abstract ranking, article review generation, section synthesis, critique, and refinement. Uses `arxiv` library for paper discovery.
    *   **Hypothesis Generation:** Employs an agentic system (`GeneratorAgent`) orchestrated by Gemini, which uses specialized tools.
*   **Tool Integration (Hypothesis Mode):**
    *   **TxGemma Predict (Local):** Runs `google/txgemma-2b-predict` locally (with 4-bit quantization for efficiency) to predict properties like clinical toxicity (`ClinicalToxTool`), BBB permeability (`BBBPredictTool`), etc., acting as tools for the agent.
    *   **PubMed Search:** Queries PubMed via the Entrez API (`BioPython`) to retrieve relevant article metadata based on agent requests.
*   **Iterative Refinement (Review Mode):** Includes an AI-driven critique and refinement loop to improve the quality of the generated literature review over multiple iterations.
*   **User Configuration:** 
    *   **Streamlit UI:** Allows setting the primary mode, research topic, specific focus, goals, article filters (for review mode), and keywords via interactive widgets.
*   **Output Formatting:**
    *   Saves intermediate results (for review mode) and final structured data in JSON format.
    *   Produces a final, formatted report in PDF using `reportlab`.
*   **Environment Management:** Uses `.env` for secure API key management and optional proxy configuration (`python-dotenv`).
*   **Dependency Management:** Includes a `requirements.txt` file.

## 3. Technology Stack

*   **Python:** (version 3.9+ recommended)
*   **Web Interface:**
    *   `streamlit`: For building the interactive user interface.
*   **Core Libraries:**
    *   `google-generativeai`: For interacting with the Gemini LLM API (Orchestrator).
    *   `transformers`: For loading and running local TxGemma Predict models.
    *   `torch`: Required by Transformers.
    *   `accelerate`: For efficient model loading across devices.
    *   `bitsandbytes`: For 4-bit quantization of local models.
    *   `huggingface_hub`: For downloading model files and prompt templates.
    *   `python-dotenv`: For managing environment variables.
    *   `requests`: (Potentially used by HF Hub or for downloads).
    *   `arxiv`: (Used only in Literature Review mode).
    *   `pdfminer.six`: (Used only in Literature Review mode).
    *   `BioPython`: For PubMed Search tool (Entrez API).
    *   `reportlab`: For generating PDF reports.
*   **Services / Models:**
    *   Google AI Studio / Vertex AI (Gemini API Key required).
    *   TxGemma Models (e.g., `google/txgemma-2b-predict` from Hugging Face).
    *   arXiv API (for Literature Review mode).
    *   PubMed Entrez API (for Hypothesis Generation mode).

## 4. Installation

Follow these steps to set up the project environment:

1.  **Prerequisites:**
    *   Python (3.9+ recommended) and `pip`.
    *   Git (optional, for cloning).
    *   **(For Local TxGemma/GPU usage):** An NVIDIA GPU with appropriate drivers and CUDA Toolkit installed (check compatibility with PyTorch/bitsandbytes).

2.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd <project-directory-name>
    ```

3.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows: venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Note: Installing PyTorch with CUDA support might require a specific command
    # from the PyTorch website (pytorch.org) depending on your CUDA version.
    # Ensure bitsandbytes is installed correctly for your OS and CUDA version.
    ```

5.  **Set up API Keys and Configuration:**
    *   Create a file named `.env` in the project root directory (`llmagents/`).
    *   Add your keys:
        ```dotenv
        GOOGLE_API_KEY='YOUR_GEMINI_API_KEY_HERE'
        HF_TOKEN='YOUR_HUGGINGFACE_TOKEN_HERE'
        # Required for PubMed tool - replace with your actual email
        MY_EMAIL_FOR_PUBMED='your.email@example.com'
        # Optional Proxy Settings
        # PROXY_HOST=...
        # PROXY_PORT=...
        # PROXY_USER=...
        # PROXY_PASSWORD=...
        ```
    *   **Crucially:** Add `.env` to your `.gitignore` file.
    *   Review and adjust settings in `config.py` (e.g., model IDs, default paths, number of agent steps).

6.  **Download TDC Prompts JSON:**
    *   Ensure the `tdc_prompts.json` file exists in the location specified by `config.TDC_PROMPTS_FILENAME`. The script will attempt to download it from Hugging Face Hub if not found locally, but having it locally is more reliable.

## 5. Usage
There are now two main ways to run the application:

### 5.1. Using the Streamlit Web Interface (Recommended for Users)

1.  **Activate Virtual Environment:**
    ```bash
    # Windows: venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```
2.  **Navigate to the project root directory** (e.g., `llmagents/`) in your terminal.
3.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    This will typically open the application 자동으로 in your default web browser.
4.  **Interact with the UI:**
    *   Select the desired mode: "Generate Literature Review" or "Generate Hypothesis & Plan".
    *   Fill in the input parameters specific to the selected mode.
    *   Click the "Generate" button.
    *   Monitor progress (spinners will be shown) and view the results directly on the page.
    *   Download generated PDF reports using the provided buttons.

    **Note for TxGemma (Hypothesis Mode):** If you encounter issues with local model loading (especially on repeated runs without Streamlit's watcher enabled via `--server.fileWatcherType none` for development), and see errors related to GPU memory or `torch._classes`, try running Streamlit with the file watcher disabled for more stability during local model use:
    ```bash
    streamlit run app.py --server.fileWatcherType none
    ```
    This is particularly relevant if you are actively developing and `torch`-related errors appear due to how Streamlit monitors file changes. For regular use after development, the simple `streamlit run app.py` should suffice if model caching is implemented correctly.

### 5.2. Using the Command-Line Interface (Legacy/Backend Testing)

1.  **Activate Virtual Environment** (as above).
2.  **Run the Main Script from Project Root:**
    ```bash
    python main.py
    ```
3.  **Select Mode:** Choose '1' for Literature Review or '2' for Hypothesis Generation when prompted.
4.  **Provide Input Parameters:** Answer the prompts specific to the selected mode.
5.  **Execution & Output:** The selected pipeline will run. Monitor the console and `agent_runner.log` for progress. Final results (PDFs, JSONs) will be saved to the directory specified in `config.OUTPUT_DIR` (e.g., `output/`).

## 6. Project Structure (Example)

llmagents/

├── .env # API keys, proxy (user-created, gitignored)

├── .gitignore

├── app.py                       # Streamlit application entry point

├── main.py                      # Legacy command-line entry point

├── config.py                    # Configuration constants

├── requirements.txt             # Python dependencies

├── hypothesis_planning_module/  # Module for hypothesis generation

│ ├── init.py

│ ├── hypothesis_agent.py        # GeneratorAgent class 

│ ├── hypothesis_pipeline.py     # run_hypothesis_pipeline function

│ └── orchestrator.py            # Gemini API call function

├── review_module/               # Module for literature review generation

│ ├── init.py

│ ├── review_agents.py           # ResearchAgent, ReviewCriticAgent, ReviewRefinerAgent

│ └── review_pipeline.py         # run_review_pipeline function

├── tools/                       # Reusable agent tools

│ ├── init.py

│ ├── tool_manager.py            # ToolManager class

│ ├── pubmed_tool.py             # PubMedSearch class

│ └── txgemma_predict_tool.py    # ClinicalTox, BBBPredictTool classes

├── utils/                       # General utility functions

│ ├── init.py

│ ├── llm_clients.py             # Functions to load local models/prompts

│ ├── pdf_generator.py           # PDF generation functions (shared or specific)

│ ├── txgemma_loader.py          # TxGemma using 

│ ├── ui_utils.py                # Helper functions for Streamlit UI components

│ └── text_parser.py             # Text parsing functions (e.g., extract_prompt)

├── prompts/ # Directory for shared/other prompts

│ ├── init.py

│ ├── tdc_prompts.json # TDC task prompts (downloaded or placed by user)

│ ├── hypothesis_prompts.py

│ ├── review_prompts.py

└── output/ # Directory for final PDF/JSON outputs (created by script)

│ └── ...
    # Streamlit UI might also trigger saves here or in a sub-directory based on config

## 7. Potential Improvements / Future Work

*   Implement a chat interface (especially for Hypothesis mode) to allow users to iteratively refine the AI's output or ask follow-up questions.
*   Refine LLM prompts for more consistent tool call formatting.
*   Implement more robust error handling and recovery within agent loops.
*   Add more sophisticated tools (e.g., chemical database search by structure, gene/protein info tools, web search).
*   Implement the iterative critique/refinement loop for the hypothesis generation mode.
*   Allow configuration of specific LLM models (e.g., different Gemini versions, different local models) via `config.py`.
*   Improve PDF generation formatting and parsing of complex LLM outputs.
*   Add unit and integration tests.