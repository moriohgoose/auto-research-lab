import re

def extract_prompt(text: str, word: str) -> str:
    """
    Extracts the content from within fenced code blocks marked by a specific keyword.

    For example, if `word` is "PubMedSearch", it finds blocks like:
    ```PubMedSearch
    content to extract
    ```
    and returns "content to extract".

    Args:
        text (str): The input string potentially containing the fenced block.
        word (str): The keyword immediately following the opening triple backticks.

    Returns:
        str: The extracted content, stripped of leading/trailing whitespace.
             Returns an empty string if no matching block is found or if the
             block is empty.
    """
    if not text or not word:
        return "" # Return empty string for invalid input
    
    code_block_pattern = rf"```{word}(.*?)```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
    extracted_code = "\n".join(code_blocks).strip()
    return extracted_code