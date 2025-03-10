"""Tests for the data models module."""

import json
import os
import tempfile
from datetime import datetime

from faker.models import Conversation, Dataset, Message, Speaker


def test_message_to_dict():
    """Test Message.to_dict() method."""
    # Create a message
    timestamp = datetime.now()
    msg = Message(
        speaker_id="user1",
        content="Hello, world!",
        timestamp=timestamp,
        metadata={"emotion": "neutral"},
    )

    # Convert to dict
    result = msg.to_dict()

    # Check result
    assert result["speaker_id"] == "user1"
    assert result["content"] == "Hello, world!"
    assert result["timestamp"] == timestamp.isoformat()
    assert result["metadata"] == {"emotion": "neutral"}


def test_conversation_to_dict():
    """Test Conversation.to_dict() method."""
    # Create speakers
    speakers = {
        "user1": Speaker(id="user1", name="User", role="user"),
        "assistant1": Speaker(id="assistant1", name="Assistant", role="assistant"),
    }

    # Create a conversation
    messages = [
        Message(speaker_id="user1", content="Hello"),
        Message(speaker_id="assistant1", content="Hi there!"),
    ]
    conv = Conversation(
        id="test-123",
        messages=messages,
        speakers=speakers,
        metadata={"domain": "greeting"},
    )

    # Convert to dict
    result = conv.to_dict()

    # Check result
    assert result["id"] == "test-123"
    assert result["metadata"] == {"domain": "greeting"}
    assert len(result["messages"]) == 2
    assert result["messages"][0]["speaker_id"] == "user1"
    assert result["messages"][0]["content"] == "Hello"
    assert result["messages"][1]["speaker_id"] == "assistant1"
    assert result["messages"][1]["content"] == "Hi there!"


def test_dataset_export_import():
    """Test Dataset export and import functionality."""
    # Create speakers
    speakers = {
        "user1": Speaker(id="user1", name="User", role="user"),
        "assistant1": Speaker(id="assistant1", name="Assistant", role="assistant"),
    }

    # Create a dataset
    messages = [
        Message(speaker_id="user1", content="Hello"),
        Message(speaker_id="assistant1", content="Hi there!"),
    ]
    conv = Conversation(
        id="test-123",
        messages=messages,
        speakers=speakers,
        metadata={"domain": "greeting"},
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
        assert imported_dataset.conversations[0].messages[0].speaker_id == "user1"
        assert imported_dataset.conversations[0].messages[0].content == "Hello"
        assert imported_dataset.conversations[0].messages[1].speaker_id == "assistant1"
        assert imported_dataset.conversations[0].messages[1].content == "Hi there!"

    finally:
        # Clean up
        os.unlink(temp_path)


def test_conversation_add_message():
    """Test Conversation.add_message() method."""
    # Create speakers
    speakers = {
        "user1": Speaker(id="user1", name="User", role="user"),
        "assistant1": Speaker(id="assistant1", name="Assistant", role="assistant"),
    }

    # Create an empty conversation
    conv = Conversation(messages=[], speakers=speakers)

    # Add messages
    conv.add_message("user1", "Hello", sentiment="positive")
    conv.add_message("assistant1", "Hi there!", intent="greeting")

    # Check results
    assert len(conv.messages) == 2
    assert conv.messages[0].speaker_id == "user1"
    assert conv.messages[0].content == "Hello"
    assert conv.messages[0].sentiment == "positive"
    assert conv.messages[1].speaker_id == "assistant1"
    assert conv.messages[1].content == "Hi there!"
    assert conv.messages[1].intent == "greeting"
