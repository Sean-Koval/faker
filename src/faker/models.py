"""Data models for conversations and datasets."""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict


@dataclass
class Message:
    """Represents a single message in a conversation."""
    
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class Conversation:
    """Represents a complete conversation with multiple messages."""
    
    messages: List[Message]
    id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'metadata': self.metadata,
            'messages': [msg.to_dict() for msg in self.messages]
        }
    
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Add a new message to the conversation.
        
        Args:
            role: The speaker role (e.g., "user", "assistant")
            content: The message content
            **kwargs: Additional metadata for the message
        """
        self.messages.append(Message(
            role=role,
            content=content,
            metadata=kwargs
        ))


@dataclass
class Dataset:
    """A collection of conversations that can be exported as a dataset."""
    
    conversations: List[Conversation]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'metadata': self.metadata,
            'conversations': [conv.to_dict() for conv in self.conversations]
        }
    
    def export(self, filepath: str, format: str = 'jsonl') -> None:
        """Export the dataset to a file.
        
        Args:
            filepath: Path to save the exported dataset
            format: Export format ('jsonl', 'json', etc.)
        """
        if format == 'jsonl':
            with open(filepath, 'w') as f:
                for conv in self.conversations:
                    f.write(json.dumps(conv.to_dict()) + '\n')
        elif format == 'json':
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    @classmethod
    def from_file(cls, filepath: str) -> 'Dataset':
        """Load a dataset from a file.
        
        Args:
            filepath: Path to the dataset file
            
        Returns:
            A Dataset object
        """
        conversations = []
        
        # Detect format based on extension
        if filepath.endswith('.jsonl'):
            with open(filepath, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    messages = [
                        Message(
                            role=msg['role'],
                            content=msg['content'],
                            timestamp=datetime.fromisoformat(msg['timestamp']),
                            metadata=msg.get('metadata', {})
                        )
                        for msg in data['messages']
                    ]
                    conversation = Conversation(
                        id=data['id'],
                        messages=messages,
                        metadata=data.get('metadata', {})
                    )
                    conversations.append(conversation)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
                for conv_data in data.get('conversations', []):
                    messages = [
                        Message(
                            role=msg['role'],
                            content=msg['content'],
                            timestamp=datetime.fromisoformat(msg['timestamp']),
                            metadata=msg.get('metadata', {})
                        )
                        for msg in conv_data['messages']
                    ]
                    conversation = Conversation(
                        id=conv_data['id'],
                        messages=messages,
                        metadata=conv_data.get('metadata', {})
                    )
                    conversations.append(conversation)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
                    
        return cls(conversations=conversations)