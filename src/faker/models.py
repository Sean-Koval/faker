"""Data models for conversations and datasets."""

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


@dataclass
class Speaker:
    """Represents a participant in a conversation."""

    id: str
    name: str
    role: str  # e.g., "customer", "agent", "user", "assistant"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class Message:
    """Represents a single message in a conversation."""

    content: str
    speaker_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Enhanced metadata for MLOps and analysis
    sentiment: Optional[str] = None
    intent: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    language: Optional[str] = None
    formality: Optional[str] = None  # formal, casual, technical, etc.

    # Extended metadata for any additional properties
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()

        # Clean up empty fields
        for key in list(result.keys()):
            if result[key] is None:
                continue
            # Skip len() check and use simple check for empty collections
            if isinstance(result[key], list) and result[key] == []:
                result[key] = None

        return result


@dataclass
class Conversation:
    """Represents a complete conversation with multiple messages."""

    messages: List[Message]
    speakers: Dict[str, Speaker]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    # Conversation-level metadata
    title: Optional[str] = None
    description: Optional[str] = None
    domain: Optional[str] = None  # e.g., "customer_support", "sales", "healthcare"
    language: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # Generation metadata (for MLOps)
    generation_config: Dict[str, Any] = field(default_factory=dict)
    prompt_template: Optional[str] = None
    prompt_variables: Dict[str, Any] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)
    generation_time: Optional[float] = None

    # Extended metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "speakers": {
                id: speaker.to_dict() for id, speaker in self.speakers.items()
            },
            "messages": [msg.to_dict() for msg in self.messages],
            "title": self.title,
            "description": self.description,
            "domain": self.domain,
            "language": self.language,
            "topics": self.topics,
            "entities": self.entities,
            "tags": self.tags,
            "generation": {
                "config": self.generation_config,
                "prompt_template": self.prompt_template,
                "prompt_variables": self.prompt_variables,
                "model_info": self.model_info,
                "generation_time": self.generation_time,
            },
            "metadata": self.metadata,
        }

        # Clean up empty fields
        for key in list(result.keys()):
            if result[key] is None:
                continue
            # Skip len() check and use simple check for empty collections
            if isinstance(result[key], list) and result[key] == []:
                result[key] = None

        return result

    def add_message(
        self,
        speaker_id: str,
        content: str,
        sentiment: Optional[str] = None,
        intent: Optional[str] = None,
        entities: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Add a new message to the conversation.

        Args:
            speaker_id: The ID of the speaker
            content: The message content
            sentiment: Optional sentiment of the message
            intent: Optional intent of the message
            entities: Optional list of entities mentioned
            topics: Optional list of topics discussed
            **kwargs: Additional metadata for the message
        """
        if speaker_id not in self.speakers:
            raise ValueError(f"Speaker with ID {speaker_id} not found")

        self.messages.append(
            Message(
                content=content,
                speaker_id=speaker_id,
                sentiment=sentiment,
                intent=intent,
                entities=entities or [],
                topics=topics or [],
                metadata=kwargs,
            )
        )

    def add_speaker(self, id: str, name: str, role: str, **metadata) -> None:
        """Add a speaker to the conversation.

        Args:
            id: Unique identifier for the speaker
            name: Name of the speaker
            role: Role of the speaker (e.g., "user", "assistant")
            **metadata: Additional metadata for the speaker
        """
        self.speakers[id] = Speaker(id=id, name=name, role=role, metadata=metadata)

    def extract_topics(self) -> Set[str]:
        """Extract and aggregate topics from all messages.

        Returns:
            A set of unique topics
        """
        topics = set()
        for message in self.messages:
            topics.update(message.topics)
        return topics

    def extract_entities(self) -> Set[str]:
        """Extract and aggregate entities from all messages.

        Returns:
            A set of unique entities
        """
        entities = set()
        for message in self.messages:
            entities.update(message.entities)
        return entities


@dataclass
class RunInfo:
    """Information about a generation run for MLOps tracking."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    config_path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        result["start_time"] = self.start_time.isoformat()
        if self.end_time:
            result["end_time"] = self.end_time.isoformat()
        return result

    def complete(self) -> None:
        """Mark the run as complete."""
        self.end_time = datetime.now()

    def add_stat(self, key: str, value: Any) -> None:
        """Add a statistics entry.

        Args:
            key: Statistic name
            value: Statistic value
        """
        self.stats[key] = value


@dataclass
class Dataset:
    """A collection of conversations that can be exported as a dataset."""

    conversations: List[Conversation]
    name: str = "synthetic_dataset"
    version: str = "1.0.0"
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    run_info: Optional[RunInfo] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "run_info": self.run_info.to_dict() if self.run_info else None,
            "tags": self.tags,
            "metadata": self.metadata,
            "conversations": [conv.to_dict() for conv in self.conversations],
        }

    def export(
        self, filepath: str, format: str = "jsonl", create_dirs: bool = True
    ) -> str:
        """Export the dataset to a file.

        Args:
            filepath: Path to save the exported dataset
            format: Export format ('jsonl', 'json', etc.)
            create_dirs: Whether to create directory structure if it doesn't exist

        Returns:
            The absolute path to the exported file
        """
        # Make path absolute
        filepath = os.path.abspath(filepath)

        # Create directory if it doesn't exist
        if create_dirs:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if format == "jsonl":
            with open(filepath, "w") as f:
                for conv in self.conversations:
                    f.write(json.dumps(conv.to_dict()) + "\n")
        elif format == "json":
            with open(filepath, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        elif format == "split":
            # Export each conversation to a separate file
            base_dir = os.path.dirname(filepath)
            dataset_name = (
                os.path.basename(filepath).replace(".json", "").replace(".jsonl", "")
            )

            # Create dataset directory
            dataset_dir = os.path.join(base_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)

            # Write dataset metadata
            metadata = self.to_dict()
            metadata.pop("conversations")
            with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            # Write individual conversations
            conversations_dir = os.path.join(dataset_dir, "conversations")
            os.makedirs(conversations_dir, exist_ok=True)

            for conv in self.conversations:
                conv_path = os.path.join(conversations_dir, f"{conv.id}.json")
                with open(conv_path, "w") as f:
                    json.dump(conv.to_dict(), f, indent=2)

            # Return the dataset directory
            return dataset_dir
        else:
            raise ValueError(f"Unsupported export format: {format}")

        return filepath

    def export_run_info(self, base_dir: str) -> str:
        """Export run information to a JSON file.

        Args:
            base_dir: Directory to save the run info

        Returns:
            The path to the exported file
        """
        if not self.run_info:
            raise ValueError("No run information available for export")

        os.makedirs(base_dir, exist_ok=True)
        run_path = os.path.join(base_dir, f"run_{self.run_info.id}.json")

        with open(run_path, "w") as f:
            json.dump(self.run_info.to_dict(), f, indent=2)

        return run_path

    def add_run_info(
        self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add run information to the dataset.

        Args:
            config_path: Path to the configuration file used
            config: Configuration dictionary
        """
        self.run_info = RunInfo(config_path=config_path, config=config or {})

    @classmethod
    def from_file(cls, filepath: str) -> "Dataset":
        """Load a dataset from a file.

        Args:
            filepath: Path to the dataset file

        Returns:
            A Dataset object
        """
        filepath = os.path.abspath(filepath)

        # Check if this is a directory (split format)
        if os.path.isdir(filepath):
            metadata_path = os.path.join(filepath, "metadata.json")
            conversations_dir = os.path.join(filepath, "conversations")

            if not os.path.exists(metadata_path) or not os.path.exists(
                conversations_dir
            ):
                raise ValueError(f"Invalid split dataset directory: {filepath}")

            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Create dataset with metadata
            dataset = cls(
                conversations=[],
                name=metadata.get("name", "imported_dataset"),
                version=metadata.get("version", "1.0.0"),
                description=metadata.get("description"),
                tags=metadata.get("tags", []),
                metadata=metadata.get("metadata", {}),
            )

            # Set created_at if available
            if "created_at" in metadata:
                dataset.created_at = datetime.fromisoformat(metadata["created_at"])

            # Set run_info if available
            if metadata.get("run_info"):
                run_info = RunInfo(**metadata["run_info"])
                if "start_time" in metadata["run_info"]:
                    run_info.start_time = datetime.fromisoformat(
                        metadata["run_info"]["start_time"]
                    )
                if (
                    "end_time" in metadata["run_info"]
                    and metadata["run_info"]["end_time"]
                ):
                    run_info.end_time = datetime.fromisoformat(
                        metadata["run_info"]["end_time"]
                    )
                dataset.run_info = run_info

            # Load conversations
            for conv_file in os.listdir(conversations_dir):
                if not conv_file.endswith(".json"):
                    continue

                conv_path = os.path.join(conversations_dir, conv_file)
                with open(conv_path, "r") as f:
                    conv_data = json.load(f)

                # Create speakers
                speakers = {}
                for speaker_id, speaker_data in conv_data.get("speakers", {}).items():
                    speakers[speaker_id] = Speaker(
                        id=speaker_data["id"],
                        name=speaker_data["name"],
                        role=speaker_data["role"],
                        metadata=speaker_data.get("metadata", {}),
                    )

                # Create messages
                messages = []
                for msg_data in conv_data.get("messages", []):
                    msg = Message(
                        content=msg_data["content"],
                        speaker_id=msg_data["speaker_id"],
                        message_id=msg_data.get("message_id", str(uuid.uuid4())),
                        sentiment=msg_data.get("sentiment"),
                        intent=msg_data.get("intent"),
                        entities=msg_data.get("entities", []),
                        topics=msg_data.get("topics", []),
                        language=msg_data.get("language"),
                        formality=msg_data.get("formality"),
                        metadata=msg_data.get("metadata", {}),
                    )

                    if "timestamp" in msg_data:
                        msg.timestamp = datetime.fromisoformat(msg_data["timestamp"])

                    messages.append(msg)

                # Create conversation
                conv = Conversation(
                    id=conv_data.get("id", str(uuid.uuid4())),
                    messages=messages,
                    speakers=speakers,
                    title=conv_data.get("title"),
                    description=conv_data.get("description"),
                    domain=conv_data.get("domain"),
                    language=conv_data.get("language"),
                    topics=conv_data.get("topics", []),
                    entities=conv_data.get("entities", []),
                    tags=conv_data.get("tags", []),
                    metadata=conv_data.get("metadata", {}),
                )

                # Set created_at if available
                if "created_at" in conv_data:
                    conv.created_at = datetime.fromisoformat(conv_data["created_at"])

                # Set generation metadata if available
                if "generation" in conv_data:
                    gen_data = conv_data["generation"]
                    conv.generation_config = gen_data.get("config", {})
                    conv.prompt_template = gen_data.get("prompt_template")
                    conv.prompt_variables = gen_data.get("prompt_variables", {})
                    conv.model_info = gen_data.get("model_info", {})
                    conv.generation_time = gen_data.get("generation_time")

                dataset.conversations.append(conv)

            return dataset

        # Handle file formats
        conversations = []

        # Detect format based on extension
        if filepath.endswith(".jsonl"):
            with open(filepath, "r") as f:
                for line in f:
                    conv_data = json.loads(line)

                    # Create speakers
                    speakers = {}
                    for speaker_id, speaker_data in conv_data.get(
                        "speakers", {}
                    ).items():
                        speakers[speaker_id] = Speaker(
                            id=speaker_data["id"],
                            name=speaker_data["name"],
                            role=speaker_data["role"],
                            metadata=speaker_data.get("metadata", {}),
                        )

                    # Create messages
                    messages = []
                    for msg_data in conv_data.get("messages", []):
                        msg = Message(
                            content=msg_data["content"],
                            speaker_id=msg_data["speaker_id"],
                            message_id=msg_data.get("message_id", str(uuid.uuid4())),
                            sentiment=msg_data.get("sentiment"),
                            intent=msg_data.get("intent"),
                            entities=msg_data.get("entities", []),
                            topics=msg_data.get("topics", []),
                            language=msg_data.get("language"),
                            formality=msg_data.get("formality"),
                            metadata=msg_data.get("metadata", {}),
                        )

                        if "timestamp" in msg_data:
                            msg.timestamp = datetime.fromisoformat(
                                msg_data["timestamp"]
                            )

                        messages.append(msg)

                    # Create conversation
                    conv = Conversation(
                        id=conv_data.get("id", str(uuid.uuid4())),
                        messages=messages,
                        speakers=speakers,
                        title=conv_data.get("title"),
                        description=conv_data.get("description"),
                        domain=conv_data.get("domain"),
                        language=conv_data.get("language"),
                        topics=conv_data.get("topics", []),
                        entities=conv_data.get("entities", []),
                        tags=conv_data.get("tags", []),
                        metadata=conv_data.get("metadata", {}),
                    )

                    # Set created_at if available
                    if "created_at" in conv_data:
                        conv.created_at = datetime.fromisoformat(
                            conv_data["created_at"]
                        )

                    # Set generation metadata if available
                    if "generation" in conv_data:
                        gen_data = conv_data["generation"]
                        conv.generation_config = gen_data.get("config", {})
                        conv.prompt_template = gen_data.get("prompt_template")
                        conv.prompt_variables = gen_data.get("prompt_variables", {})
                        conv.model_info = gen_data.get("model_info", {})
                        conv.generation_time = gen_data.get("generation_time")

                    conversations.append(conv)

        elif filepath.endswith(".json"):
            with open(filepath, "r") as f:
                data = json.load(f)

                # Create dataset with metadata
                dataset = cls(
                    conversations=[],
                    name=data.get("name", "imported_dataset"),
                    version=data.get("version", "1.0.0"),
                    description=data.get("description"),
                    tags=data.get("tags", []),
                    metadata=data.get("metadata", {}),
                )

                # Set created_at if available
                if "created_at" in data:
                    dataset.created_at = datetime.fromisoformat(data["created_at"])

                # Set run_info if available
                if data.get("run_info"):
                    run_info = RunInfo(**data["run_info"])
                    if "start_time" in data["run_info"]:
                        run_info.start_time = datetime.fromisoformat(
                            data["run_info"]["start_time"]
                        )
                    if "end_time" in data["run_info"] and data["run_info"]["end_time"]:
                        run_info.end_time = datetime.fromisoformat(
                            data["run_info"]["end_time"]
                        )
                    dataset.run_info = run_info

                # Load conversations
                for conv_data in data.get("conversations", []):
                    # Create speakers
                    speakers = {}
                    for speaker_id, speaker_data in conv_data.get(
                        "speakers", {}
                    ).items():
                        speakers[speaker_id] = Speaker(
                            id=speaker_data["id"],
                            name=speaker_data["name"],
                            role=speaker_data["role"],
                            metadata=speaker_data.get("metadata", {}),
                        )

                    # Create messages
                    messages = []
                    for msg_data in conv_data.get("messages", []):
                        msg = Message(
                            content=msg_data["content"],
                            speaker_id=msg_data["speaker_id"],
                            message_id=msg_data.get("message_id", str(uuid.uuid4())),
                            sentiment=msg_data.get("sentiment"),
                            intent=msg_data.get("intent"),
                            entities=msg_data.get("entities", []),
                            topics=msg_data.get("topics", []),
                            language=msg_data.get("language"),
                            formality=msg_data.get("formality"),
                            metadata=msg_data.get("metadata", {}),
                        )

                        if "timestamp" in msg_data:
                            msg.timestamp = datetime.fromisoformat(
                                msg_data["timestamp"]
                            )

                        messages.append(msg)

                    # Create conversation
                    conv = Conversation(
                        id=conv_data.get("id", str(uuid.uuid4())),
                        messages=messages,
                        speakers=speakers,
                        title=conv_data.get("title"),
                        description=conv_data.get("description"),
                        domain=conv_data.get("domain"),
                        language=conv_data.get("language"),
                        topics=conv_data.get("topics", []),
                        entities=conv_data.get("entities", []),
                        tags=conv_data.get("tags", []),
                        metadata=conv_data.get("metadata", {}),
                    )

                    # Set created_at if available
                    if "created_at" in conv_data:
                        conv.created_at = datetime.fromisoformat(
                            conv_data["created_at"]
                        )

                    # Set generation metadata if available
                    if "generation" in conv_data:
                        gen_data = conv_data["generation"]
                        conv.generation_config = gen_data.get("config", {})
                        conv.prompt_template = gen_data.get("prompt_template")
                        conv.prompt_variables = gen_data.get("prompt_variables", {})
                        conv.model_info = gen_data.get("model_info", {})
                        conv.generation_time = gen_data.get("generation_time")

                    dataset.conversations.append(conv)

                return dataset
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        # Create dataset from conversations (for JSONL format)
        return cls(conversations=conversations)
