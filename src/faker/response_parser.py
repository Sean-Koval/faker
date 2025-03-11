"""Response parser for handling LLM responses.

This module handles parsing and validation of LLM responses, 
with specific focus on conversation data in JSON format.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union


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


def validate_conversation_messages(
    messages: List[Dict], required_roles: Optional[List[str]] = None
) -> Tuple[List[Dict], bool]:
    """Validate and fix conversation messages against schema.
    
    Args:
        messages: List of message dictionaries to validate
        required_roles: List of roles that should be present
        
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
        
        # Add metadata fields, using empty lists for array fields
        for field in metadata_fields:
            if field in ['entities', 'topics'] and (field not in message or message[field] is None):
                clean_message[field] = []
            else:
                clean_message[field] = message.get(field, None)
                
        valid_messages.append(clean_message)
    
    # Check if there's at least one message from each required role
    if required_roles:
        roles_present = {msg['role'] for msg in valid_messages}
        missing_roles = set(required_roles) - roles_present
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