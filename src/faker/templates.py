"""Template handling for generating prompts."""

import random
import re
from typing import Any, Dict, List, Optional, Union

# JSON format instruction templates
JSON_FORMAT_INSTRUCTION = """
IMPORTANT: Your response MUST be formatted as a VALID JSON ARRAY with NO TEXT before or after.
Each array element MUST have ALL of these fields: "role", "content", "sentiment", "intent", "entities", "topics", "formality".
DO NOT include markdown code blocks, backticks, or any other formatting. ONLY VALID JSON.
"""

CONVERSATION_JSON_SCHEMA = """
The response MUST follow this exact schema:
[
  {
    "role": "user or support_agent",
    "content": "the message text",
    "sentiment": "positive, neutral, or negative",
    "intent": "greeting, question, clarification, solution, or farewell",
    "entities": ["array", "of", "named", "entities"],
    "topics": ["array", "of", "topics"],
    "formality": "formal, casual, or technical"
  },
  ... more messages ...
]
"""

VALIDATION_REMINDER = """
Before responding, validate that:
1. Your response is a syntactically valid JSON array
2. Each object has ALL required fields
3. No fields have null values
4. The array is properly terminated with ]
5. There is NO text before or after the JSON array
"""


class TemplateEngine:
    """Simple template engine for generating prompts."""

    def __init__(self, templates: Dict[str, str]):
        """Initialize the template engine.

        Args:
            templates: Dictionary of template name to template string
        """
        self.templates = templates
        
        # Add formatting instructions to the templates if not already present
        if "json_format_instruction" not in self.templates:
            self.templates["json_format_instruction"] = JSON_FORMAT_INSTRUCTION
            
        if "json_schema" not in self.templates:
            self.templates["json_schema"] = CONVERSATION_JSON_SCHEMA
            
        if "validation_reminder" not in self.templates:
            self.templates["validation_reminder"] = VALIDATION_REMINDER

    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a template with the given context.

        Args:
            template_name: Name of the template to render
            context: Dictionary of variables to use in the template

        Returns:
            The rendered template string

        Raises:
            ValueError: If the template doesn't exist
        """
        if template_name not in self.templates:
            raise ValueError(f"Template not found: {template_name}")

        template = self.templates[template_name]
        return self._render_string(template, context)

    def _render_string(self, template: str, context: Dict[str, Any]) -> str:
        """Render a template string with the given context.

        Args:
            template: The template string
            context: Dictionary of variables to use in the template

        Returns:
            The rendered template string
        """

        # Replace {{variable}} with the value from context
        def replace_var(match):
            var_name = match.group(1).strip()

            # Handle nested properties with dot notation
            if "." in var_name:
                parts = var_name.split(".")
                value = context
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return match.group(0)  # Keep original if property not found
                return str(value)

            # Handle simple variable
            if var_name in context:
                return str(context[var_name])

            # Handle random selection from a list
            if var_name.startswith("random(") and var_name.endswith(")"):
                list_name = var_name[7:-1].strip()
                if list_name in context and isinstance(context[list_name], list):
                    return random.choice(context[list_name])

            return match.group(0)  # Keep original if variable not found

        # Replace variables in the template
        result = re.sub(r"{{([^}]+)}}", replace_var, template)
        return result
        
    def add_formatting_instructions(self, prompt: str, include_schema: bool = True) -> str:
        """Add JSON formatting instructions to a prompt.
        
        Args:
            prompt: Original prompt
            include_schema: Whether to include the detailed schema
            
        Returns:
            Enhanced prompt with formatting instructions
        """
        formatted_prompt = f"{prompt}\n\n{self.templates['json_format_instruction']}"
        
        if include_schema:
            formatted_prompt += f"\n\n{self.templates['json_schema']}"
            
        formatted_prompt += f"\n\n{self.templates['validation_reminder']}"
        
        return formatted_prompt
