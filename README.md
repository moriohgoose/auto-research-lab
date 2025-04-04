# Auto Research Lab

## 1. Overview

This project implements an automated pipeline for generating and iteratively refining academic literature reviews. Given a research topic, optional keywords, and other parameters, the system leverages Large Language Models (specifically Google's Gemini via `google-generativeai`) and the `arxiv` library to:

1.  **Discover:** Find relevant research papers on arXiv.
2.  **Analyze:** Download papers, extract text, and generate detailed summaries (reviews) for each selected paper using an LLM.
3.  **Synthesize:** Combine the individual paper reviews into a coherent, structured literature review draft covering key themes, methods, datasets, challenges, and conclusions, using an LLM.
4.  **Critique:** Evaluate the synthesized review draft using an LLM based on predefined criteria (clarity, depth, coverage, etc.).
5.  **Refine:** Based on the critique scores and feedback, decide whether the review meets quality standards, requires further refinement (triggering another LLM rewrite based on feedback), or needs a complete restart (if quality degrades significantly). This iterative process continues until stopping criteria are met or a maximum number of iterations is reached.
6.  **Format:** Generate a final report including a title, abstract (both AI-generated), the main review body (with numbered in-text citations), and a formatted reference list in both JSON and user-friendly PDF formats using `reportlab`.

The pipeline is designed to be configurable via user input and utilizes environment variables for secure API key management and optional proxy configuration. Logging provides detailed insight into the execution flow.

## 2. Key Features

*   **Automated Paper Discovery:** Searches arXiv based on user-defined or LLM-generated keywords and filters.
*   **LLM-Powered Ranking:** Ranks search results by relevance using Gemini.
*   **PDF Processing:** Downloads relevant papers and extracts full text using `pdfminer.six`.
*   **Detailed Article Reviews:** Generates structured summaries for each selected paper via LLM.
*   **Intelligent Synthesis:** Creates a structured literature review draft by synthesizing individual article reviews using LLM, following specific section prompts.
*   **AI-Driven Critique:** Evaluates the generated draft using LLM against defined quality metrics.
*   **Iterative Refinement:** Automatically decides whether to stop, refine, or restart the review generation based on critique scores, aiming for quality improvement.
*   **Content Generation:** Automatically generates a suitable Title and Abstract for the final review.
*   **Citation Management:** Creates a formatted reference list and replaces in-text markers with corresponding numbers.
*   **PDF Report Generation:** Produces a formatted PDF document of the final literature review using `reportlab`.
*   **User Configuration:** Allows setting the topic, focus, article count, year range, and keywords via command-line input.
*   **Proxy Support:** Configurable HTTP/HTTPS proxy via environment variables.
*   **Logging:** Detailed execution logs saved to `review_generator.log`.

## 3. Technology Stack

*   **Python:** (version 3.13)
*   **Core Libraries:**
    *   `google-generativeai`: For interacting with the Gemini LLM.
    *   `arxiv`: For searching and retrieving papers from arXiv.
    *   `requests`: For downloading PDFs.
    *   `pdfminer.six`: For extracting text from PDFs.
    *   `reportlab`: For generating the final PDF report.
    *   `python-dotenv`: For managing environment variables (API keys, proxy).
*   **Services:**
    *   Google AI Studio (Gemini API)
    *   arXiv API

## 4. Installation

Follow these steps to set up the project environment:

1.  **Prerequisites:**
    *   Ensure you have Python (version 3.9 or higher recommended) and `pip` installed.
    *   Git (optional, for cloning).

2.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd <project-directory-name>
    ```

3.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    The required Python packages are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up Google AI (Gemini) API Key:**
    *   Obtain an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   Create a file named `.env` in the root directory of the project.
    *   Add the following line to the `.env` file, replacing `YOUR_API_KEY_HERE` with your actual key:
        ```dotenv
        GOOGLE_API_KEY='YOUR_API_KEY_HERE'
        ```
    *   **Important:** Add `.env` to your `.gitignore` file to avoid committing your API key.

6.  **Configure Proxy (Optional):**
    *   If you need to use a proxy to access the internet (for Gemini API, arXiv, PDF downloads), add the following lines to your `.env` file, replacing the placeholders with your actual proxy details:
        ```dotenv
        PROXY_HOST=your_proxy_host
        PROXY_PORT=your_proxy_port
        PROXY_USER=your_proxy_user
        PROXY_PASSWORD=your_proxy_password
        ```
    *   If these variables are not set or incomplete, the script will attempt to run without a proxy.

## 5. Usage

1.  **Activate Virtual Environment:**
    ```bash
    # Windows: venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```
2.  **Run the Main Script:**
    ```bash
    python main.py
    ```
3.  **Provide Input Parameters:** The script will prompt you to enter the following:
    *   **Research Topic:** The main subject of the review (e.g., "LLMs in Drug Research").
    *   **Specification Prompt:** Optional specific focus (e.g., "Focus on novel methods..."). Press Enter to use default or skip if default is empty.
    *   **Number of Articles:** How many top-ranked articles to include (e.g., 10).
    *   **Article Years Filter:** Year range like "YYYY-YYYY" (e.g., "2020-2025"), or type "skip", or press Enter for default.
    *   **Keywords:** Comma-separated keywords. Press Enter to let the script auto-generate them based on the topic.

4.  **Execution:** The pipeline will start, logging its progress to the console and `review_generator.log`. This may take a significant amount of time depending on the number of articles, iterations, and LLM response times.

5.  **Output Files:** Upon completion (or failure), check the project directory for:
    *   `review_generator.log`: Detailed logs of the entire process.
    *   `downloaded_pdfs/`: Folder containing downloaded PDF articles.
    *   `run_*.json`: Intermediate JSON files for each research/critique cycle.
    *   `final_literature_review.json`: A JSON file containing the final results, including the best review text, sections, generated title/abstract, formatted references, final scores, metadata, etc.
    *   `final_literature_review.pdf`: The final, formatted literature review document.

## 6. Project Structure
.
├── .env # Stores API keys and proxy settings (user-created, gitignored)
├── agents.py # Defines ResearchAgent, ReviewCriticAgent, ReviewRefinerAgent classes
├── downloaded_pdfs/ # Directory for downloaded papers (created by script)
├── literature_review_pipeline.py # Contains the main iterative pipeline function (run_literature_review_pipeline)
├── main.py # Main script: handles user input, runs the pipeline, processes final output
├── requirements.txt # Project dependencies (user-creates from list above)
├── utils.py # Helper functions (API calls, PDF generation, text processing)
├── review_generator.log # Log file (created by script)
├── review_runs/ # Directory for intermediate run outputs (created by script)
├── final_literature_review.json # Final structured results (created by script)
└── final_literature_review.pdf # Final PDF report (created by script)

      
*   **`main.py`:** Entry point, user interaction, pipeline invocation, final output handling.
*   **`review_pipeline.py`:** Contains the core `run_review_pipeline` function orchestrating the iterative generation and refinement process.
*   **`agents.py`:** Defines the logic for the different AI agents involved (research, critique, refinement decision-making).
*   **`utils.py`:** Provides reusable utility functions used by agents and the pipeline.


## 7. Potential Improvements / Future Work

*   Implement more sophisticated error handling and recovery within the pipeline.
*   Allow configuration of LLM models and parameters (e.g., temperature) via config file or command line.
*   Integrate more advanced PDF text extraction techniques if `pdfminer.six` struggles with certain layouts.
*   Implement caching for API calls and intermediate results to speed up repeated runs.
*   Explore different strategies for synthesizing sections (e.g., multi-step generation).
*   Add support for other academic search engines besides arXiv.
*   Improve the reliability of parsing LLM outputs (e.g., JSON critique, ranked lists).