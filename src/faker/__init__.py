"""Faker - Modular synthetic chat data generation system."""

from src.faker.generator import ChatGenerator
from src.faker.models import Conversation, Dataset, Message, Speaker
from src.faker.templates import TemplateEngine
from src.faker.response_parser import parse_llm_response, validate_conversation_messages

__version__ = "0.1.1"
__all__ = [
    "ChatGenerator", 
    "Conversation", 
    "Message", 
    "Dataset", 
    "Speaker",
    "TemplateEngine",
    "parse_llm_response",
    "validate_conversation_messages"
]
