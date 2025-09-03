import re
import logging

_LOGGER = logging.getLogger(__name__)

def get_code_from_markdown(markdown_string: str) -> list[str]:
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
    _LOGGER.info(markdown_string)
    # The regular expression pattern to match fenced code blocks.
    # ```: Matches the literal opening backticks.
    # (?:[a-z]+\n)?: This is a non-capturing group that matches an optional
    #                language specifier followed by a newline (e.g., 'python\n').
    # (.*?): This is the key part that captures the code content. It's a non-greedy
    #        match for any character (including newlines due to re.DOTALL).
    # ```: Matches the literal closing backticks.
    # re.DOTALL: A flag that allows '.' to match any character, including a newline.
    #            This is essential for matching multi-line code blocks.
    pattern = r"```(?:[a-z]+\n)?(.*?)```"

    # Use re.findall to find all non-overlapping matches of the pattern.
    # findall returns a list of all captured groups.
    code_blocks = re.findall(pattern, markdown_string, re.DOTALL)
    [re.sub("\\n", "\n", block) for block in code_blocks]

    # Strip any leading or trailing whitespace from each extracted block.
    _LOGGER.info(code_blocks)
    return [block.strip() for block in code_blocks]
