"""Template handling for generating prompts."""

import random
import re
from typing import Any, Dict, List, Optional, Union

from src.faker.logging_service import PerformanceTimer, timer

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

# Few-shot examples for variable substitution
FEW_SHOT_EXAMPLES = """
CRITICAL: You MUST directly use the provided names in your output.

For example, if given:
- advisor_name: "John Smith"
- client_name: "Michael Brown" 

You MUST use these EXACT names in messages, like:

✅ CORRECT: "Hello Michael Brown, I'm John Smith. How can I help with your portfolio today?"
❌ INCORRECT: "Hello {{client_name}}, I'm {{advisor_name}}. How can I help with your portfolio today?"

Similarly, for:
- agent_name: "David Lee"
- user_name: "Alex Thompson"

✅ CORRECT: "Hi Alex Thompson, this is David Lee from customer support. How may I assist you?"
❌ INCORRECT: "Hi {{user_name}}, this is {{agent_name}} from customer support. How may I assist you?"

Names to use in this conversation:
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
            
        if "few_shot_examples" not in self.templates:
            self.templates["few_shot_examples"] = FEW_SHOT_EXAMPLES

    @timer("template_render")
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
        
        # Start timer for the main template rendering
        PerformanceTimer.start_timer("template_render_string")
        rendered = self._render_string(template, context)
        PerformanceTimer.end_timer("template_render_string")
        
        # After initial rendering, check if there are still template markers in the result
        # This can happen if nested templates are used or if the template contains examples
        import re
        import logging
        logger = logging.getLogger(__name__)
        
        # Find all remaining {{var}} patterns
        remaining_vars = re.findall(r'{{(.+?)}}', rendered)
        if remaining_vars:
            # Filter out variables from examples like {{name}} which are intentionally there
            filtered_vars = [var.strip() for var in remaining_vars if var.strip() != 'name']
            
            # Also filter out variables in examples section which are part of instructions
            filtered_vars = [var for var in filtered_vars if 
                           not (var == 'advisor_name' and 'EXAMPLES OF CORRECT NAME USAGE' in rendered) and
                           not (var == 'client_name' and 'EXAMPLES OF CORRECT NAME USAGE' in rendered) and
                           not (var == 'agent_name' and 'EXAMPLES OF CORRECT NAME USAGE' in rendered) and
                           not (var == 'user_name' and 'EXAMPLES OF CORRECT NAME USAGE' in rendered) and
                           not (var == 'company_name' and 'EXAMPLES OF CORRECT NAME USAGE' in rendered) and
                           not (var == 'advisor_firm' and 'EXAMPLES OF CORRECT NAME USAGE' in rendered)]
            
            # For conversation template, enhance with explicit values
            if template_name == "conversation" and filtered_vars:
                logger.info(f"Adding explicit values for template variables in rendered prompt: {filtered_vars}")
                
                # Start timer for variable substitution post-processing
                PerformanceTimer.start_timer("template_post_process")
                
                # Add explicit notes about name variables at the end of the prompt
                for var in filtered_vars:
                    var = var.strip()
                    # Only process name variables that have actual values in context
                    if var in context and isinstance(context[var], str) and var.endswith('_name'):
                        # Find the pattern and add the value in quotes
                        pattern = f"{{{{{var}}}}}"
                        if pattern in rendered:
                            rendered = rendered.replace(pattern, f'"{context[var]}"')
                
                PerformanceTimer.end_timer("template_post_process")
        
        # Record template size for performance analysis
        template_length = len(template)
        rendered_length = len(rendered)
        PerformanceTimer.record_tokens(
            "template_rendering", 
            template_length // 4,  # Rough estimate of input token count
            rendered_length // 4   # Rough estimate of output token count
        )
        
        return rendered

    def _render_string(self, template: str, context: Dict[str, Any]) -> str:
        """Render a template string with the given context.

        Args:
            template: The template string
            context: Dictionary of variables to use in the template

        Returns:
            The rendered template string
        """
        import logging
        logger = logging.getLogger(__name__)
        
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
                        logger.warning(f"Nested property not found: {var_name}")
                        return match.group(0)  # Keep original if property not found
                return str(value)

            # Handle simple variable
            if var_name in context:
                # For name variables, add quotes if not already present
                if '_name' in var_name and isinstance(context[var_name], str):
                    # For special handling of name variables, we want them to be properly quoted
                    # if they appear directly in prompts (but not in content)
                    val = context[var_name]
                    # If it's being used in a prompt (standalone variable), quote it
                    # if not already quoted
                    if not (val.startswith('"') and val.endswith('"')):
                        logger.debug(f"Adding quotes to name variable: {var_name}")
                        return f'"{val}"'
                return str(context[var_name])

            # Handle random selection from a list
            if var_name.startswith("random(") and var_name.endswith(")"):
                list_name = var_name[7:-1].strip()
                if list_name in context and isinstance(context[list_name], list) and context[list_name]:
                    return str(random.choice(context[list_name]))
                else:
                    logger.warning(f"Random list not found or empty: {list_name}")
                    return "[value not found]"
                    
            # Handle conditional expressions (simple if-then)
            if var_name.startswith("if(") and "then" in var_name and "else" in var_name:
                try:
                    # Extract condition, then, and else parts using regex
                    condition_match = re.match(r"if\((.*?)\)\s*then\s*(.*?)\s*else\s*(.*)", var_name)
                    if condition_match:
                        condition = condition_match.group(1).strip()
                        then_part = condition_match.group(2).strip()
                        else_part = condition_match.group(3).strip()
                        
                        # Evaluate condition
                        condition_value = False
                        if condition in context:
                            condition_value = bool(context[condition])
                        
                        # Return then or else part
                        return str(context.get(then_part, then_part) if condition_value else context.get(else_part, else_part))
                except Exception as e:
                    logger.warning(f"Error in conditional expression: {var_name}, {str(e)}")
                    
            # Handle formatting and concatenation (var | format)
            if "|" in var_name:
                parts = var_name.split("|")
                base_var = parts[0].strip()
                format_op = parts[1].strip()
                
                if base_var in context:
                    base_value = context[base_var]
                    try:
                        if format_op.startswith("format(") and format_op.endswith(")"):
                            # Extract format string
                            format_str = format_op[7:-1].strip()
                            # Apply format
                            formatted = format_str.format(base_value)
                            return formatted
                    except Exception as e:
                        logger.warning(f"Error in format operation: {var_name}, {str(e)}")

            # Only log a warning for non-example variables
            # (The examples in the template often use {{name}} format deliberately to show what to replace)
            if not (var_name.endswith('_name') and match.group(0) in ["{{advisor_name}}", "{{client_name}}", "{{agent_name}}", "{{user_name}}"]):
                logger.debug(f"Variable not found in context: {var_name}")
            return match.group(0)  # Keep original if variable not found or if we can't process it

        # First pass: Replace variables in the template with a marker for unresolved vars
        result = re.sub(r"{{([^}]+)}}", replace_var, template)
        
        # Second pass: Try to resolve any variables that depend on other variables
        if "{{" in result and "}}" in result:
            result = re.sub(r"{{([^}]+)}}", replace_var, result)
        
        return result
        
    def add_formatting_instructions(self, prompt: str, include_schema: bool = True, context: Optional[Dict[str, Any]] = None) -> str:
        """Add JSON formatting instructions to a prompt.
        
        Args:
            prompt: Original prompt
            include_schema: Whether to include the detailed schema
            context: Optional context with variables to add to the few-shot examples
            
        Returns:
            Enhanced prompt with formatting instructions
        """
        formatted_prompt = prompt
        
        # Add few-shot examples with explicit name variables if context is provided
        if context:
            name_variables = {}
            
            # Extract all name variables from context
            for key, value in context.items():
                if isinstance(value, str) and '_name' in key:
                    name_variables[key] = value
            
            # If we found name variables, add few-shot examples
            if name_variables:
                few_shot = self.templates['few_shot_examples']
                
                # Add the list of specific names to use
                for name_key, name_value in name_variables.items():
                    few_shot += f"- {name_key}: \"{name_value}\"\n"
                
                # Add explicit examples using the actual names
                # For advisor-client scenario
                if 'advisor_name' in name_variables and 'client_name' in name_variables:
                    advisor = name_variables['advisor_name']
                    client = name_variables['client_name']
                    few_shot += f"\nExample conversation using these names:\n"
                    few_shot += f"- \"{client}: Hello, I'd like to discuss my investment portfolio.\"\n"
                    few_shot += f"- \"{advisor}: Good morning {client}, I'd be happy to review your portfolio. What specific concerns do you have?\"\n"
                
                # For support agent-user scenario
                elif 'agent_name' in name_variables and 'user_name' in name_variables:
                    agent = name_variables['agent_name']
                    user = name_variables['user_name']
                    few_shot += f"\nExample conversation using these names:\n"
                    few_shot += f"- \"{user}: I'm having trouble logging into my account.\"\n"
                    few_shot += f"- \"{agent}: I understand, {user}. This is {agent} from technical support. Let me help you resolve this login issue.\"\n"
                
                formatted_prompt = f"{few_shot}\n\n{formatted_prompt}"
        
        # Add standard formatting instructions
        formatted_prompt += f"\n\n{self.templates['json_format_instruction']}"
        
        if include_schema:
            formatted_prompt += f"\n\n{self.templates['json_schema']}"
            
        formatted_prompt += f"\n\n{self.templates['validation_reminder']}"
        
        return formatted_prompt
