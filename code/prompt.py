from typing import Union, List, Optional, Dict, Any


def lowercase_first_char(text: str) -> str:
    """Lowercases the first character of a string.

    Args:
        text: Input string.

    Returns:
        The input string with the first character lowercased.
    """
    return text[0].lower() + text[1:] if text else text


def format_prompt_section(lead_in: str, value: Union[str, List[str]]) -> str:
    """Formats a prompt section by joining a lead-in with content.

    Args:
        lead_in: Introduction sentence for the section.
        value: Section content, as a string or list of strings.

    Returns:
        A formatted string with the lead-in followed by the content.
    """
    if isinstance(value, list):
        formatted_value = "\n".join(f"- {item}" for item in value)
    else:
        formatted_value = value
    return f"{lead_in}\n{formatted_value}"

def build_prompt_from_config(
    config: Dict[str, Any],
    input_data: str = "",
    app_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Builds a complete prompt string based on a config dictionary.

    Args:
        config: Dictionary specifying prompt components.
        input_data: Content to be processed.
        app_config: Optional app-wide configuration (e.g., reasoning strategies).

    Returns:
        A fully constructed prompt as a string.

    Raises:
        ValueError: If the required 'instruction' field is missing.
    """
    prompt_parts = []

    # Goal
    if goal := config.get("goal"):
        prompt_parts.append(
            f"Goal:\n{goal.strip()}"
        )

    # Role
    if role := config.get("role"):
        prompt_parts.append(f"You are {role.strip()}.")

    # Instruction
    instruction = config.get("instruction")
    if not instruction:
        raise ValueError("Missing required field: 'instruction'")
    prompt_parts.append(
        f"Instruction:\n{instruction.strip()}"
    )

    # Output constraints
    if constraints := config.get("output_constraints"):
        prompt_parts.append(
            format_prompt_section(
                "Output constraints:", constraints
            )
        )

    # Style / tone
    if tone := config.get("style_or_tone"):
        prompt_parts.append(
            format_prompt_section(
                "Style and tone guidelines:", tone
            )
        )

    # Output format
    if format_ := config.get("output_format"):
        prompt_parts.append(
            format_prompt_section("Response format:", format_)
        )


    # Debugging / self-describe
    if debug_instr := config.get("meta_instruction_for_debugging"):
        prompt_parts.append(f"Debug instructions:\n{debug_instr.strip()}")

    # Input data
    if input_data:
        prompt_parts.append(
            "Content to process:\n"
            "<<<BEGIN CONTENT>>>\n"
            "```\n" + input_data.strip() + "\n```\n<<<END CONTENT>>>"
        )

    # Reasoning strategy
    reasoning_strategy = config.get("reasoning_strategy")
    if reasoning_strategy and reasoning_strategy != "None" and app_config:
        strategies = app_config.get("reasoning_strategies", {})
        if strategy_text := strategies.get(reasoning_strategy):
            prompt_parts.append(strategy_text.strip())

    # Final instruction
    prompt_parts.append("Now perform the task as instructed above.")

    return "\n\n".join(prompt_parts)
