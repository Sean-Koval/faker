"""Tests for the data models module."""

import json
import os
import tempfile
from datetime import datetime
from faker.models import Message, Conversation, Dataset


def test_message_to_dict():
    """Test Message.to_dict() method."""
    # Create a message
    timestamp = datetime.now()
    msg = Message(
        role="user",
        content="Hello, world!",
        timestamp=timestamp,
        metadata={"emotion": "neutral"}
    )
    
    # Convert to dict
    result = msg.to_dict()
    
    # Check result
    assert result["role"] == "user"
    assert result["content"] == "Hello, world!"
    assert result["timestamp"] == timestamp.isoformat()
    assert result["metadata"] == {"emotion": "neutral"}


def test_conversation_to_dict():
    """Test Conversation.to_dict() method."""
    # Create a conversation
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!")
    ]
    conv = Conversation(
        id="test-123",
        messages=messages,
        metadata={"domain": "greeting"}
    )
    
    # Convert to dict
    result = conv.to_dict()
    
    # Check result
    assert result["id"] == "test-123"
    assert result["metadata"] == {"domain": "greeting"}
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == "Hello"
    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][1]["content"] == "Hi there!"


def test_dataset_export_import():
    """Test Dataset export and import functionality."""
    # Create a dataset
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!")
    ]
    conv = Conversation(
        id="test-123",
        messages=messages,
        metadata={"domain": "greeting"}
    )
    dataset = Dataset(conversations=[conv])
    
    # Export to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        temp_path = f.name
    
    try:
        # Export
        dataset.export(temp_path, format="jsonl")
        
        # Import
        imported_dataset = Dataset.from_file(temp_path)
        
        # Check imported dataset
        assert len(imported_dataset.conversations) == 1
        assert imported_dataset.conversations[0].id == "test-123"
        assert len(imported_dataset.conversations[0].messages) == 2
        assert imported_dataset.conversations[0].messages[0].role == "user"
        assert imported_dataset.conversations[0].messages[0].content == "Hello"
        assert imported_dataset.conversations[0].messages[1].role == "assistant"
        assert imported_dataset.conversations[0].messages[1].content == "Hi there!"
    
    finally:
        # Clean up
        os.unlink(temp_path)


def test_conversation_add_message():
    """Test Conversation.add_message() method."""
    # Create an empty conversation
    conv = Conversation(messages=[])
    
    # Add messages
    conv.add_message("user", "Hello", sentiment="positive")
    conv.add_message("assistant", "Hi there!", intent="greeting")
    
    # Check results
    assert len(conv.messages) == 2
    assert conv.messages[0].role == "user"
    assert conv.messages[0].content == "Hello"
    assert conv.messages[0].metadata == {"sentiment": "positive"}
    assert conv.messages[1].role == "assistant"
    assert conv.messages[1].content == "Hi there!"
    assert conv.messages[1].metadata == {"intent": "greeting"}