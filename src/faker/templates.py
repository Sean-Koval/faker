"""Template handling for generating prompts."""

import re
import random
from typing import Any, Dict, List, Optional, Union


class TemplateEngine:
    """Simple template engine for generating prompts."""
    
    def __init__(self, templates: Dict[str, str]):
        """Initialize the template engine.
        
        Args:
            templates: Dictionary of template name to template string
        """
        self.templates = templates
    
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