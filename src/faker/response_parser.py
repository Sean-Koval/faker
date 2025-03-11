"""Response parser for handling LLM responses.

This module handles parsing and validation of LLM responses, 
with specific focus on conversation data in JSON format.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

# Import timer for performance logging
from src.faker.logging_service import PerformanceTimer, timer


logger = logging.getLogger(__name__)


def parse_llm_response(response_text: str) -> Union[List[Dict], Dict]:
    """Parse LLM response into structured data.
    
    Attempts multiple parsing strategies to extract valid JSON data
    from LLM responses, handling common formatting issues.
    
    Args:
        response_text: The raw text response from the LLM
        
    Returns:
        Parsed JSON data as a list or dictionary
        
    Raises:
        ValueError: If the response cannot be parsed as valid JSON after all attempts
    """
    # Strip any markdown code blocks or backticks
    cleaned_text = strip_code_blocks(response_text)
    
    # First parsing attempt: direct JSON loading
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        logger.debug("Initial JSON parsing failed, trying fallback methods")
    
    # Second attempt: find JSON array in text
    array_match = re.search(r'\[(.*)\]', cleaned_text, re.DOTALL)
    if array_match:
        try:
            return json.loads(f"[{array_match.group(1)}]")
        except json.JSONDecodeError:
            logger.debug("JSON array extraction failed")
    
    # Third attempt: find JSON object in text
    object_match = re.search(r'\{(.*)\}', cleaned_text, re.DOTALL)
    if object_match:
        try:
            return json.loads(f"{{{object_match.group(1)}}}")
        except json.JSONDecodeError:
            logger.debug("JSON object extraction failed")
    
    # Fourth attempt: try to repair common JSON issues
    repaired_json = repair_json(cleaned_text)
    try:
        return json.loads(repaired_json)
    except json.JSONDecodeError:
        logger.warning("All JSON parsing attempts failed")
        raise ValueError(f"Failed to parse response as JSON: {response_text[:500]}...")


def strip_code_blocks(text: str) -> str:
    """Remove markdown code blocks from text.
    
    Args:
        text: Text that may contain markdown code blocks
        
    Returns:
        Text with code blocks removed
    """
    # Remove ```json and ``` markers
    text = re.sub(r'```(?:json)?', '', text)
    text = re.sub(r'```', '', text)
    # Remove leading/trailing whitespace
    return text.strip()


def repair_json(text: str) -> str:
    """Attempt to repair common JSON formatting issues.
    
    Args:
        text: Potentially malformed JSON text
        
    Returns:
        Repaired JSON string
    """
    # Replace single quotes with double quotes (common LLM mistake)
    text = re.sub(r"(?<![\\'])(')((?:.(?!(?<![\\'])')))(')", r'"\2"', text)
    
    # Fix trailing commas in arrays and objects
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    # Ensure property names are quoted
    text = re.sub(r'(\s*)([a-zA-Z0-9_]+)(\s*):(\s*)', r'\1"\2"\3:\4', text)
    
    # Convert Python None to JSON null
    text = re.sub(r'\bNone\b', 'null', text)
    
    # Convert Python True/False to JSON true/false
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    
    return text


@timer("validate_messages")
def validate_conversation_messages(
    messages: List[Dict], 
    required_roles: Optional[List[str]] = None,
    context_vars: Optional[Dict[str, Any]] = None
) -> Tuple[List[Dict], bool]:
    """Validate and fix conversation messages against schema.
    
    Args:
        messages: List of message dictionaries to validate
        required_roles: List of roles that should be present
        context_vars: Optional context variables to replace placeholders in content
        
    Returns:
        Tuple of (fixed_messages, is_fully_valid)
    """
    if not messages:
        return [], False
    
    required_fields = ['role', 'content']
    metadata_fields = ['sentiment', 'intent', 'entities', 'topics', 'formality']
    allowed_roles = required_roles or ['user', 'assistant', 'support_agent', 'agent', 'customer']
    
    valid_messages = []
    is_fully_valid = True
    
    # Start timer for placeholder detection
    PerformanceTimer.start_timer("placeholder_detection")
    
    for i, message in enumerate(messages):
        # Check if message is a dictionary
        if not isinstance(message, dict):
            logger.warning(f"Message {i} is not a dictionary: {message}")
            is_fully_valid = False
            continue
            
        # Validate required fields
        missing_fields = [field for field in required_fields if field not in message]
        if missing_fields:
            logger.warning(f"Message {i} is missing required fields: {missing_fields}")
            is_fully_valid = False
            
            # Skip messages without core fields
            if 'content' not in message or 'role' not in message:
                continue
        
        # Ensure role is valid
        role = message.get('role', '')
        if role not in allowed_roles:
            logger.warning(f"Message {i} has invalid role: {role}")
            is_fully_valid = False
            # Attempt to fix by assigning alternating roles
            message['role'] = allowed_roles[i % len(allowed_roles)]
        
        # Create a clean message with all fields
        clean_message = {
            'role': message.get('role', allowed_roles[i % len(allowed_roles)]),
            'content': message.get('content', ''),
        }
        
        # Process content to replace any remaining placeholders if context is provided
        if 'content' in clean_message and isinstance(clean_message['content'], str):
            content = clean_message['content']
            
            # If context is provided, use it to replace placeholders
            if context_vars:
                # Look for template placeholders like {{var_name}}
                placeholder_pattern = r'\{\{([^}]+)\}\}'
                placeholders = re.findall(placeholder_pattern, content)
                
                if placeholders:
                    logger.warning(f"Found template placeholders in message {i}: {placeholders}")
                    
                    # Replace placeholders with actual values
                    for placeholder in placeholders:
                        placeholder_name = placeholder.strip()
                        if placeholder_name in context_vars and isinstance(context_vars[placeholder_name], str):
                            placeholder_pattern = f'{{{{{placeholder_name}}}}}'
                            content = content.replace(placeholder_pattern, context_vars[placeholder_name])
                            logger.info(f"Replaced placeholder {placeholder_name} with {context_vars[placeholder_name]}")
            
            # Extra check for name patterns even without explicit placeholders
            if context_vars:
                # Check for typical placeholder patterns that might not be in {{}} format
                name_patterns = [
                    (r'\[([A-Za-z_]+_name)\]', r'\1'),         # [advisor_name]
                    (r'\<([A-Za-z_]+_name)\>', r'\1'),         # <advisor_name>
                    (r'__([A-Za-z_]+_name)__', r'\1'),         # __advisor_name__
                    (r'ADVISOR_NAME', 'advisor_name'),         # ADVISOR_NAME
                    (r'CLIENT_NAME', 'client_name'),           # CLIENT_NAME
                    (r'AGENT_NAME', 'agent_name'),             # AGENT_NAME
                    (r'USER_NAME', 'user_name'),               # USER_NAME
                    (r'CUSTOMER_NAME', 'user_name'),           # CUSTOMER_NAME
                    (r'SUPPORT_AGENT_NAME', 'agent_name'),     # SUPPORT_AGENT_NAME
                ]
                
                for pattern, replacement_key in name_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        for match in matches:
                            key = re.sub(pattern, replacement_key, match)
                            if key in context_vars and isinstance(context_vars[key], str):
                                content = re.sub(pattern, context_vars[key], content)
                                logger.info(f"Replaced variant placeholder format {match} with {context_vars[key]}")
            
            # Update the message with fixed content
            clean_message['content'] = content
        
        # Add metadata fields, using empty lists for array fields
        for field in metadata_fields:
            if field in ['entities', 'topics']:
                # These should be lists - ensure they are
                if field not in message or message[field] is None:
                    clean_message[field] = []
                elif not isinstance(message[field], list):
                    # If the field exists but isn't a list, convert it
                    logger.warning(f"Message {i} has {field} that is not a list, converting: {message[field]}")
                    try:
                        # Try to convert to list if possible, otherwise use empty list
                        clean_message[field] = list(message[field]) if hasattr(message[field], '__iter__') else []
                    except (TypeError, ValueError):
                        clean_message[field] = []
                else:
                    # It's a proper list, use it directly
                    clean_message[field] = message[field]
            else:
                # For other fields, just use the value or None
                clean_message[field] = message.get(field, None)
                
        valid_messages.append(clean_message)
    
    PerformanceTimer.end_timer("placeholder_detection")
    
    # Check if there's at least one message from each required role
    if required_roles:
        # Convert required_roles to a set of strings to ensure we don't have unhashable types
        required_roles_set = set()
        for role in required_roles:
            if isinstance(role, str):
                required_roles_set.add(role)
            else:
                # If it's not a string, convert it to a string to avoid unhashable types
                logger.warning(f"Non-string role detected: {role}, converting to string")
                required_roles_set.add(str(role))
                
        roles_present = {msg['role'] for msg in valid_messages}
        missing_roles = required_roles_set - roles_present
        if missing_roles:
            logger.warning(f"Conversation is missing required roles: {missing_roles}")
            is_fully_valid = False
    
    return valid_messages, is_fully_valid


def extract_conversation_from_text(
    text: str, roles: List[str], max_messages: int = 12
) -> List[Dict]:
    """Fallback method to extract conversation from text when JSON parsing fails.
    
    Args:
        text: Text containing conversation
        roles: List of roles to alternate between
        max_messages: Maximum number of messages to extract
        
    Returns:
        List of message dictionaries
    """
    # Try to extract messages using role prefixes
    messages = []
    
    # Look for role prefixes like "User:" or "Agent:"
    for role in roles:
        pattern = rf"{role}(?::|\s*-)\s*(.*?)(?=\n\s*(?:{('|').join(roles)}):|$)"
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            content = match.group(1).strip()
            if content:
                messages.append({
                    "role": role.lower(),
                    "content": content,
                    "sentiment": None,
                    "intent": None,
                    "entities": [],
                    "topics": [],
                    "formality": None
                })
    
    # If no messages found with role prefixes, fall back to simple line splitting
    if not messages:
        lines = re.split(r"\n+", text)
        lines = [line.strip() for line in lines if line.strip()]
        
        for i, line in enumerate(lines[:max_messages]):
            role = roles[i % len(roles)]
            messages.append({
                "role": role,
                "content": line,
                "sentiment": None,
                "intent": None,
                "entities": [],
                "topics": [],
                "formality": None
            })
    
    return messages