"""Faker - Modular synthetic chat data generation system."""

from faker.generator import ChatGenerator
from faker.models import Conversation, Message, Dataset

__version__ = "0.1.0"
__all__ = ["ChatGenerator", "Conversation", "Message", "Dataset"]