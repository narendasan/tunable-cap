import re
from typing import List

def extract_code_blocks(markdown_string: str) -> List[str]:
    """
    Extracts all code blocks from a markdown-formatted string.

    This function uses a regular expression to find all instances of
    fenced code blocks, which are enclosed by triple backticks (```).
    It captures the content of the block and returns a list of the
    captured code strings.

    Args:
        markdown_string: The input string formatted with markdown.

    Returns:
        A list of strings, where each string is the content of a code block.
        The backticks and language identifiers are not included.
    """
    # The regular expression pattern to match fenced code blocks.
    # ```: Matches the literal opening backticks.
    # (?:[a-z]+\n)?: This is a non-capturing group that matches an optional
    #                language specifier followed by a newline (e.g., 'python\n').
    # (.*?): This is the key part that captures the code content. It's a non-greedy
    #        match for any character (including newlines due to re.DOTALL).
    # ```: Matches the literal closing backticks.
    # re.DOTALL: A flag that allows '.' to match any character, including a newline.
    #            This is essential for matching multi-line code blocks.
    pattern = r"```python(?:[a-z]+\n)?(.*?)```"

    # Use re.findall to find all non-overlapping matches of the pattern.
    # findall returns a list of all captured groups.
    code_blocks = re.findall(pattern, markdown_string, re.DOTALL)

    # Strip any leading or trailing whitespace from each extracted block.
    return [block.strip() for block in code_blocks]

if __name__ == "__main__":
    # Example markdown string with multiple code blocks
    markdown_content = """
    # My Awesome Markdown Document

    Here is a paragraph of text.

    ```python
    # This is a Python code block
    def hello_world():
        print("Hello, World!")

    hello_world()
    ```

    Here is some more text in between the code blocks.

    ```
    // This is an untagged code block (like a JavaScript snippet)
    function greet(name) {
      console.log(`Hello, ${name}!`);
    }
    ```

    And here is one more code block, this one for a shell script.

    ```bash
    # Run a command
    npm install
    git add .
    git commit -m "Add new feature"
    ```
    """

    # Extract the code blocks
    extracted_blocks = extract_code_blocks(markdown_content)

    # Print the extracted code blocks
    print("Found and extracted the following code blocks:")
    for i, block in enumerate(extracted_blocks):
        print("-" * 30)
        print(f"Code Block #{i+1}:\n")
        print(block)
        print("-" * 30)

    # You can now save these to files, or use them in your program.
    # For example, to save the first block to a file:
    # with open("my_python_code.py", "w") as f:
    #     f.write(extracted_blocks[0])
