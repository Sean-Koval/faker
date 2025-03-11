"""Logging service for faker.

This module provides a centralized logging service for tracking generation runs,
storing run metadata, and computing metrics.
"""

import asyncio
import functools
import json
import logging
import os
import sqlite3
import time
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from src.faker.models import Conversation, Dataset, RunInfo


class PerformanceTimer:
    """Utility for timing code execution and collecting performance metrics."""
    
    _timers: Dict[str, Dict[str, Any]] = {}
    _counts: Dict[str, int] = {}
    _token_counts: Dict[str, Dict[str, int]] = {}
    
    @classmethod
    def reset(cls):
        """Reset all timers and counters."""
        cls._timers = {}
        cls._counts = {}
        cls._token_counts = {}
    
    @classmethod
    def start_timer(cls, name: str) -> float:
        """Start a timer with the given name.
        
        Args:
            name: Name of the timer
            
        Returns:
            Current time in seconds
        """
        start_time = time.time()
        if name not in cls._timers:
            cls._timers[name] = {
                'starts': [],
                'ends': [],
                'durations': []
            }
        cls._timers[name]['starts'].append(start_time)
        cls._counts[name] = cls._counts.get(name, 0) + 1
        return start_time
    
    @classmethod
    def end_timer(cls, name: str) -> float:
        """End a timer with the given name.
        
        Args:
            name: Name of the timer
            
        Returns:
            Duration in seconds
        """
        end_time = time.time()
        
        if name not in cls._timers or not cls._timers[name]['starts']:
            logging.warning(f"Timer {name} was never started")
            return 0.0
        
        start_time = cls._timers[name]['starts'].pop()
        cls._timers[name]['ends'].append(end_time)
        duration = end_time - start_time
        cls._timers[name]['durations'].append(duration)
        return duration
    
    @classmethod
    def record_tokens(cls, name: str, input_tokens: int, output_tokens: int):
        """Record token counts for an operation.
        
        Args:
            name: Name of the operation
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        if name not in cls._token_counts:
            cls._token_counts[name] = {
                'input': 0,
                'output': 0,
                'total': 0,
                'calls': 0
            }
        
        cls._token_counts[name]['input'] += input_tokens
        cls._token_counts[name]['output'] += output_tokens
        cls._token_counts[name]['total'] += (input_tokens + output_tokens)
        cls._token_counts[name]['calls'] += 1
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Get statistics for all timers.
        
        Returns:
            Dictionary containing timing statistics
        """
        stats = {}
        
        # Process timing data
        for name, timer in cls._timers.items():
            durations = timer['durations']
            if not durations:
                continue
                
            stats[name] = {
                'count': cls._counts.get(name, 0),
                'total_time': sum(durations),
                'avg_time': sum(durations) / len(durations),
                'min_time': min(durations) if durations else 0,
                'max_time': max(durations) if durations else 0
            }
        
        # Add token statistics
        stats['tokens'] = cls._token_counts.copy()
        
        # Add total token counts across all operations
        total_input = sum(tc['input'] for tc in cls._token_counts.values())
        total_output = sum(tc['output'] for tc in cls._token_counts.values())
        total_calls = sum(tc['calls'] for tc in cls._token_counts.values())
        
        stats['tokens']['all'] = {
            'input': total_input,
            'output': total_output,
            'total': total_input + total_output,
            'calls': total_calls
        }
        
        return stats


def timer(name: str):
    """Decorator to time function execution.
    
    Args:
        name: Name for the timer
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            PerformanceTimer.start_timer(name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                PerformanceTimer.end_timer(name)
        return wrapper
    return decorator


def async_timer(name: str):
    """Decorator to time async function execution.
    
    Args:
        name: Name for the timer
        
    Returns:
        Decorated async function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            PerformanceTimer.start_timer(name)
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                PerformanceTimer.end_timer(name)
        return wrapper
    return decorator


class LogStore:
    """Storage interface for run logs and metrics.

    This class provides an abstract base for different storage implementations.
    """

    def save_run(self, run_info: Dict[str, Any]) -> str:
        """Save run information to storage.

        Args:
            run_info: Dictionary containing run information

        Returns:
            The ID of the saved run
        """
        raise NotImplementedError("Subclasses must implement save_run")

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve run information from storage.

        Args:
            run_id: The ID of the run to retrieve

        Returns:
            Dictionary containing run information, or None if not found
        """
        raise NotImplementedError("Subclasses must implement get_run")

    def list_runs(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List runs in storage.

        Args:
            limit: Maximum number of runs to retrieve
            offset: Offset for pagination

        Returns:
            List of dictionaries containing run information
        """
        raise NotImplementedError("Subclasses must implement list_runs")

    def save_metric(self, run_id: str, metric_name: str, metric_value: Any) -> None:
        """Save a metric value for a run.

        Args:
            run_id: The ID of the run
            metric_name: The name of the metric
            metric_value: The value of the metric
        """
        raise NotImplementedError("Subclasses must implement save_metric")

    def get_metrics(self, run_id: str) -> Dict[str, Any]:
        """Retrieve metrics for a run.

        Args:
            run_id: The ID of the run

        Returns:
            Dictionary mapping metric names to values
        """
        raise NotImplementedError("Subclasses must implement get_metrics")

    def search_runs(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for runs matching query criteria.

        Args:
            query: Dictionary mapping fields to search values

        Returns:
            List of dictionaries containing run information
        """
        raise NotImplementedError("Subclasses must implement search_runs")


class JSONLogStore(LogStore):
    """JSON file-based implementation of LogStore.

    This stores run information and metrics in JSON files.
    """

    def __init__(self, base_dir: str = "./logs"):
        """Initialize the JSON log store.

        Args:
            base_dir: Base directory for storing log files
        """
        self.base_dir = os.path.abspath(base_dir)
        self.runs_dir = os.path.join(self.base_dir, "runs")
        self.metrics_dir = os.path.join(self.base_dir, "metrics")

        # Create directories if they don't exist
        os.makedirs(self.runs_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Create index file if it doesn't exist
        self.index_path = os.path.join(self.base_dir, "index.json")
        if not os.path.exists(self.index_path):
            with open(self.index_path, "w") as f:
                json.dump({"runs": []}, f)

    def save_run(self, run_info: Dict[str, Any]) -> str:
        """Save run information to a JSON file.

        Args:
            run_info: Dictionary containing run information

        Returns:
            The ID of the saved run
        """
        # Ensure run has an ID
        if "id" not in run_info:
            run_info["id"] = str(uuid.uuid4())

        run_id = run_info["id"]
        run_path = os.path.join(self.runs_dir, f"{run_id}.json")

        # Add timestamp if not present
        if "timestamp" not in run_info:
            run_info["timestamp"] = datetime.now().isoformat()

        # Save run information
        with open(run_path, "w") as f:
            json.dump(run_info, f, indent=2)

        # Update index
        try:
            with open(self.index_path, "r") as f:
                index = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            index = {"runs": []}

        # Add to index if not already present
        if run_id not in [r.get("id") for r in index["runs"]]:
            index["runs"].append(
                {
                    "id": run_id,
                    "name": run_info.get("name", "Unnamed run"),
                    "timestamp": run_info["timestamp"],
                    "config_path": run_info.get("config_path"),
                }
            )

            # Sort by timestamp (newest first)
            index["runs"].sort(key=lambda r: r["timestamp"], reverse=True)

            with open(self.index_path, "w") as f:
                json.dump(index, f, indent=2)

        return run_id

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve run information from a JSON file.

        Args:
            run_id: The ID of the run to retrieve

        Returns:
            Dictionary containing run information, or None if not found
        """
        run_path = os.path.join(self.runs_dir, f"{run_id}.json")

        if not os.path.exists(run_path):
            return None

        try:
            with open(run_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None

    def list_runs(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List runs from the index.

        Args:
            limit: Maximum number of runs to retrieve
            offset: Offset for pagination

        Returns:
            List of dictionaries containing run information
        """
        try:
            with open(self.index_path, "r") as f:
                index = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

        # Apply pagination
        runs = index.get("runs", [])[offset : offset + limit]

        return runs

    def save_metric(self, run_id: str, metric_name: str, metric_value: Any) -> None:
        """Save a metric value for a run.

        Args:
            run_id: The ID of the run
            metric_name: The name of the metric
            metric_value: The value of the metric
        """
        metrics_path = os.path.join(self.metrics_dir, f"{run_id}.json")

        # Load existing metrics
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            metrics = {}

        # Add or update metric
        metrics[metric_name] = metric_value

        # Save metrics
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    def get_metrics(self, run_id: str) -> Dict[str, Any]:
        """Retrieve metrics for a run.

        Args:
            run_id: The ID of the run

        Returns:
            Dictionary mapping metric names to values
        """
        metrics_path = os.path.join(self.metrics_dir, f"{run_id}.json")

        if not os.path.exists(metrics_path):
            return {}

        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def search_runs(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for runs matching query criteria.

        Args:
            query: Dictionary mapping fields to search values

        Returns:
            List of dictionaries containing run information
        """
        runs = self.list_runs(limit=1000)  # Get all runs (up to 1000)

        # Filter by query criteria
        filtered_runs = []
        for run in runs:
            match = True
            for key, value in query.items():
                if key not in run or run[key] != value:
                    match = False
                    break

            if match:
                filtered_runs.append(run)

        return filtered_runs


class SQLiteLogStore(LogStore):
    """SQLite implementation of LogStore.

    This stores run information and metrics in an SQLite database.
    """

    def __init__(self, db_path: str = "./logs/runs.db"):
        """Initialize the SQLite log store.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = os.path.abspath(db_path)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create runs table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            name TEXT,
            timestamp TEXT,
            config_path TEXT,
            data TEXT
        )
        """
        )

        # Create metrics table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS metrics (
            run_id TEXT,
            name TEXT,
            value TEXT,
            PRIMARY KEY (run_id, name),
            FOREIGN KEY (run_id) REFERENCES runs (id)
        )
        """
        )

        # Create index on timestamp
        cursor.execute(
            """
        CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs (timestamp)
        """
        )

        conn.commit()
        conn.close()

    def save_run(self, run_info: Dict[str, Any]) -> str:
        """Save run information to the database.

        Args:
            run_info: Dictionary containing run information

        Returns:
            The ID of the saved run
        """
        # Ensure run has an ID
        if "id" not in run_info:
            run_info["id"] = str(uuid.uuid4())

        run_id = run_info["id"]

        # Add timestamp if not present
        if "timestamp" not in run_info:
            run_info["timestamp"] = datetime.now().isoformat()

        # Extract fields for the runs table
        name = run_info.get("name", "Unnamed run")
        timestamp = run_info["timestamp"]
        config_path = run_info.get("config_path")

        # Serialize the full run info
        data = json.dumps(run_info)

        # Insert or update the run
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        INSERT OR REPLACE INTO runs (id, name, timestamp, config_path, data)
        VALUES (?, ?, ?, ?, ?)
        """,
            (run_id, name, timestamp, config_path, data),
        )

        conn.commit()
        conn.close()

        return run_id

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve run information from the database.

        Args:
            run_id: The ID of the run to retrieve

        Returns:
            Dictionary containing run information, or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        SELECT data FROM runs WHERE id = ?
        """,
            (run_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    def list_runs(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List runs from the database.

        Args:
            limit: Maximum number of runs to retrieve
            offset: Offset for pagination

        Returns:
            List of dictionaries containing run information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        SELECT id, name, timestamp, config_path FROM runs
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
        """,
            (limit, offset),
        )

        runs = []
        for row in cursor.fetchall():
            runs.append(
                {
                    "id": row[0],
                    "name": row[1],
                    "timestamp": row[2],
                    "config_path": row[3],
                }
            )

        conn.close()
        return runs

    def save_metric(self, run_id: str, metric_name: str, metric_value: Any) -> None:
        """Save a metric value for a run.

        Args:
            run_id: The ID of the run
            metric_name: The name of the metric
            metric_value: The value of the metric
        """
        # Serialize the metric value
        value = json.dumps(metric_value)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        INSERT OR REPLACE INTO metrics (run_id, name, value)
        VALUES (?, ?, ?)
        """,
            (run_id, metric_name, value),
        )

        conn.commit()
        conn.close()

    def get_metrics(self, run_id: str) -> Dict[str, Any]:
        """Retrieve metrics for a run.

        Args:
            run_id: The ID of the run

        Returns:
            Dictionary mapping metric names to values
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
        SELECT name, value FROM metrics WHERE run_id = ?
        """,
            (run_id,),
        )

        metrics = {}
        for row in cursor.fetchall():
            try:
                metrics[row[0]] = json.loads(row[1])
            except json.JSONDecodeError:
                metrics[row[0]] = row[1]

        conn.close()
        return metrics

    def search_runs(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for runs matching query criteria.

        Args:
            query: Dictionary mapping fields to search values

        Returns:
            List of dictionaries containing run information
        """
        conditions = []
        params = []

        # Build query conditions
        for key, value in query.items():
            if key in ["id", "name", "timestamp", "config_path"]:
                conditions.append(f"{key} = ?")
                params.append(value)
            else:
                # Search in JSON data
                conditions.append(f"json_extract(data, '$.{key}') = ?")
                params.append(json.dumps(value))

        if not conditions:
            return self.list_runs()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query_str = f"""
        SELECT id, name, timestamp, config_path FROM runs
        WHERE {' AND '.join(conditions)}
        ORDER BY timestamp DESC
        """

        cursor.execute(query_str, params)

        runs = []
        for row in cursor.fetchall():
            runs.append(
                {
                    "id": row[0],
                    "name": row[1],
                    "timestamp": row[2],
                    "config_path": row[3],
                }
            )

        conn.close()
        return runs


class MetricsService:
    """Service for computing and tracking metrics about datasets and runs."""

    @staticmethod
    def compute_dataset_metrics(dataset: Dataset) -> Dict[str, Any]:
        """Compute metrics for a dataset.

        Args:
            dataset: The dataset to compute metrics for

        Returns:
            Dictionary containing metrics
        """
        # Basic statistics
        num_conversations = len(dataset.conversations)
        total_messages = sum(len(conv.messages) for conv in dataset.conversations)
        avg_messages = total_messages / max(num_conversations, 1)

        # Collect all roles, topics, entities
        all_roles = set()
        all_topics = set()
        all_entities = set()
        all_intents = set()
        all_sentiments = set()
        all_formalities = set()

        # Count by domain
        domains: Dict[str, int] = {}

        # Message length statistics
        message_lengths = []

        # Speaker statistics
        speakers_by_role: Dict[str, int] = {}

        for conv in dataset.conversations:
            # Add domain count
            if conv.domain:
                domains[conv.domain] = domains.get(conv.domain, 0) + 1

            # Add topics and entities
            if conv.topics:
                all_topics.update(conv.topics)
            if conv.entities:
                all_entities.update(conv.entities)

            # Add speaker roles and count by role
            for speaker in conv.speakers.values():
                all_roles.add(speaker.role)
                speakers_by_role[speaker.role] = (
                    speakers_by_role.get(speaker.role, 0) + 1
                )

            # Add message metadata and compute statistics
            for msg in conv.messages:
                if msg.sentiment:
                    all_sentiments.add(msg.sentiment)
                if msg.intent:
                    all_intents.add(msg.intent)
                if msg.formality:
                    all_formalities.add(msg.formality)
                if msg.topics:
                    all_topics.update(msg.topics)
                if msg.entities:
                    # Process entities to handle dictionaries
                    for entity in msg.entities:
                        if isinstance(entity, dict):
                            # For dictionaries, extract a string representation
                            if 'entity' in entity and isinstance(entity['entity'], str):
                                all_entities.add(entity['entity'])
                            elif 'standard_form' in entity and isinstance(entity['standard_form'], str):
                                all_entities.add(entity['standard_form'])
                            else:
                                # Use a frozen representation to make it hashable
                                try:
                                    all_entities.add(str(entity))
                                except (TypeError, ValueError):
                                    logger.warning(f"Could not add entity to set: {entity}")
                        elif isinstance(entity, str):
                            # Strings are already hashable
                            all_entities.add(entity)
                        elif entity is not None:
                            # Convert other types to string
                            try:
                                all_entities.add(str(entity))
                            except (TypeError, ValueError):
                                logger.warning(f"Could not add entity to set: {entity}")

                # Add message length
                message_lengths.append(len(msg.content))

        # Compute message length statistics
        if message_lengths:
            avg_message_length = sum(message_lengths) / len(message_lengths)
            min_message_length = min(message_lengths)
            max_message_length = max(message_lengths)
        else:
            avg_message_length = 0
            min_message_length = 0
            max_message_length = 0

        # Count topic frequency
        topic_frequency: Dict[str, int] = {}
        entity_frequency: Dict[str, int] = {}
        sentiment_frequency: Dict[str, int] = {}
        intent_frequency: Dict[str, int] = {}

        for conv in dataset.conversations:
            for msg in conv.messages:
                if msg.sentiment:
                    sentiment_frequency[msg.sentiment] = (
                        sentiment_frequency.get(msg.sentiment, 0) + 1
                    )
                if msg.intent:
                    intent_frequency[msg.intent] = (
                        intent_frequency.get(msg.intent, 0) + 1
                    )
                if msg.topics:
                    for topic in msg.topics:
                        topic_frequency[topic] = topic_frequency.get(topic, 0) + 1
                if msg.entities:
                    # Process entities to handle dictionaries
                    for entity in msg.entities:
                        entity_key = None
                        if isinstance(entity, dict):
                            # For dictionaries, extract a string representation
                            if 'entity' in entity and isinstance(entity['entity'], str):
                                entity_key = entity['entity']
                            elif 'standard_form' in entity and isinstance(entity['standard_form'], str):
                                entity_key = entity['standard_form']
                            else:
                                # Use a string representation
                                try:
                                    entity_key = str(entity)
                                except Exception:
                                    continue  # Skip if we can't create a string key
                        elif isinstance(entity, str):
                            # Strings can be used directly
                            entity_key = entity
                        elif entity is not None:
                            # Convert other types to string
                            try:
                                entity_key = str(entity)
                            except Exception:
                                continue  # Skip if we can't create a string key
                        
                        # Update frequency for this entity key
                        if entity_key is not None:
                            entity_frequency[entity_key] = entity_frequency.get(entity_key, 0) + 1

        # Compile metrics
        metrics = {
            "num_conversations": num_conversations,
            "total_messages": total_messages,
            "avg_messages_per_conversation": avg_messages,
            "message_length": {
                "avg": avg_message_length,
                "min": min_message_length,
                "max": max_message_length,
            },
            "domains": domains,
            "roles": list(all_roles),
            "speakers_by_role": speakers_by_role,
            "topics": {
                "count": len(all_topics),
                "top_20": dict(
                    sorted(topic_frequency.items(), key=lambda x: x[1], reverse=True)[
                        :20
                    ]
                ),
            },
            "entities": {
                "count": len(all_entities),
                "top_20": dict(
                    sorted(entity_frequency.items(), key=lambda x: x[1], reverse=True)[
                        :20
                    ]
                ),
            },
            "sentiments": {
                "count": len(all_sentiments),
                "frequency": sentiment_frequency,
            },
            "intents": {"count": len(all_intents), "frequency": intent_frequency},
            "formalities": list(all_formalities),
            "timestamp": datetime.now().isoformat(),
        }

        return metrics

    @staticmethod
    def compute_conversation_metrics(conversation: Conversation) -> Dict[str, Any]:
        """Compute metrics for a single conversation.

        Args:
            conversation: The conversation to compute metrics for

        Returns:
            Dictionary containing metrics
        """
        num_messages = len(conversation.messages)
        num_speakers = len(conversation.speakers)

        # Count messages by speaker
        messages_by_speaker: Dict[str, int] = {}
        for msg in conversation.messages:
            messages_by_speaker[msg.speaker_id] = (
                messages_by_speaker.get(msg.speaker_id, 0) + 1
            )

        # Map speaker IDs to roles
        speaker_roles = {
            id: speaker.role for id, speaker in conversation.speakers.items()
        }

        # Count messages by role
        messages_by_role: Dict[str, int] = {}
        for speaker_id, count in messages_by_speaker.items():
            role = speaker_roles.get(speaker_id, "unknown")
            messages_by_role[role] = messages_by_role.get(role, 0) + count

        # Message length statistics
        message_lengths = [len(msg.content) for msg in conversation.messages]
        avg_message_length = sum(message_lengths) / max(len(message_lengths), 1)

        # Sentiment distribution
        sentiments = [msg.sentiment for msg in conversation.messages if msg.sentiment]
        sentiment_counts: Dict[str, int] = {}
        for sentiment in sentiments:
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

        # Intent distribution
        intents = [msg.intent for msg in conversation.messages if msg.intent]
        intent_counts: Dict[str, int] = {}
        for intent in intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        # Topic and entity counts
        topics = set()
        entities = set()
        for msg in conversation.messages:
            if msg.topics:
                topics.update(msg.topics)
            if msg.entities:
                # Process entities to handle dictionaries
                for entity in msg.entities:
                    if isinstance(entity, dict):
                        # For dictionaries, extract a string representation
                        if 'entity' in entity and isinstance(entity['entity'], str):
                            entities.add(entity['entity'])
                        elif 'standard_form' in entity and isinstance(entity['standard_form'], str):
                            entities.add(entity['standard_form'])
                        else:
                            # Use a string representation
                            try:
                                entities.add(str(entity))
                            except (TypeError, ValueError):
                                pass
                    elif isinstance(entity, str):
                        # Strings are already hashable
                        entities.add(entity)
                    elif entity is not None:
                        # Convert other types to string
                        try:
                            entities.add(str(entity))
                        except (TypeError, ValueError):
                            pass

        return {
            "id": conversation.id,
            "num_messages": num_messages,
            "num_speakers": num_speakers,
            "messages_by_speaker": messages_by_speaker,
            "messages_by_role": messages_by_role,
            "avg_message_length": avg_message_length,
            "sentiment_distribution": sentiment_counts,
            "intent_distribution": intent_counts,
            "num_topics": len(topics),
            "topics": list(topics),
            "num_entities": len(entities),
            "entities": list(entities),
            "domain": conversation.domain,
            "timestamp": datetime.now().isoformat(),
        }


class LoggingService:
    """Service for logging and tracking run information."""

    def __init__(self, store: Optional[LogStore] = None, log_dir: str = "./logs"):
        """Initialize the logging service.

        Args:
            store: LogStore implementation to use
            log_dir: Directory for storing logs (if store is not provided)
        """
        self.log_dir = os.path.abspath(log_dir)
        self.store = store or JSONLogStore(base_dir=log_dir)
        self.metrics_service = MetricsService()
        self.logger = logging.getLogger(__name__)

    def init_run(self, config: Dict[str, Any], name: Optional[str] = None) -> str:
        """Initialize a new run and return its ID.

        Args:
            config: Configuration for the run
            name: Optional name for the run

        Returns:
            The ID of the created run
        """
        run_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Reset performance counters at the start of a run
        PerformanceTimer.reset()

        run_info = {
            "id": run_id,
            "name": name or f"Run {timestamp}",
            "timestamp": timestamp,
            "config": config,
            "config_path": config.get("config_path"),
            "status": "started",
            "start_time": timestamp,
        }

        # Save to store
        self.store.save_run(run_info)

        self.logger.info(f"Initialized run {run_id}")
        return run_id

    def complete_run(self, run_id: str, dataset: Dataset) -> None:
        """Mark a run as complete and compute metrics.

        Args:
            run_id: The ID of the run
            dataset: The dataset generated in the run
        """
        # Get the existing run info
        run_info = self.store.get_run(run_id)
        if not run_info:
            self.logger.warning(f"Run {run_id} not found")
            return

        # Update run status
        run_info["status"] = "completed"
        run_info["end_time"] = datetime.now().isoformat()
        run_info["duration"] = (
            datetime.fromisoformat(run_info["end_time"])
            - datetime.fromisoformat(run_info["start_time"])
        ).total_seconds()

        # Save updated run info
        self.store.save_run(run_info)

        # Compute and save metrics
        metrics = self.metrics_service.compute_dataset_metrics(dataset)
        self.store.save_metric(run_id, "dataset_metrics", metrics)

        # Save individual conversation metrics
        for i, conversation in enumerate(dataset.conversations):
            conv_metrics = self.metrics_service.compute_conversation_metrics(
                conversation
            )
            self.store.save_metric(run_id, f"conversation_{i}", conv_metrics)

        # Collect performance metrics
        performance_stats = PerformanceTimer.get_stats()
        self.store.save_metric(run_id, "performance", performance_stats)
        
        # Get token usage
        token_metrics = performance_stats.get('tokens', {})
        
        # Save overall statistics
        self.store.save_metric(run_id, "num_conversations", len(dataset.conversations))
        self.store.save_metric(run_id, "generation_time", run_info["duration"])
        self.store.save_metric(run_id, "token_usage", token_metrics.get('all', {}))
        
        # Log performance summary
        if 'tokens' in performance_stats and 'all' in performance_stats['tokens']:
            all_tokens = performance_stats['tokens']['all']
            self.logger.info(
                f"Token usage: {all_tokens.get('total', 0)} total tokens "
                f"({all_tokens.get('input', 0)} input, {all_tokens.get('output', 0)} output) "
                f"across {all_tokens.get('calls', 0)} API calls"
            )
            
        # Calculate and log throughput metrics
        total_time = run_info["duration"]
        if total_time > 0 and dataset.conversations:
            convs_per_sec = len(dataset.conversations) / total_time
            tokens_per_sec = performance_stats.get('tokens', {}).get('all', {}).get('total', 0) / total_time
            
            self.store.save_metric(run_id, "throughput", {
                "conversations_per_second": convs_per_sec,
                "tokens_per_second": tokens_per_sec
            })
            
            self.logger.info(
                f"Throughput: {convs_per_sec:.2f} conversations/sec, {tokens_per_sec:.2f} tokens/sec"
            )

        self.logger.info(
            f"Completed run {run_id}, duration: {run_info['duration']:.2f}s"
        )

    def log_error(self, run_id: str, error: str) -> None:
        """Log an error for a run.

        Args:
            run_id: The ID of the run
            error: Error message or exception
        """
        # Get the existing run info
        run_info = self.store.get_run(run_id)
        if not run_info:
            self.logger.warning(f"Run {run_id} not found")
            return

        # Update run status
        run_info["status"] = "error"
        run_info["error"] = str(error)
        run_info["end_time"] = datetime.now().isoformat()

        # Save updated run info
        self.store.save_run(run_info)

        self.logger.error(f"Run {run_id} failed: {error}")

    def get_run_info(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a run.

        Args:
            run_id: The ID of the run

        Returns:
            Dictionary containing run information, or None if not found
        """
        return self.store.get_run(run_id)

    def get_run_metrics(self, run_id: str) -> Dict[str, Any]:
        """Get metrics for a run.

        Args:
            run_id: The ID of the run

        Returns:
            Dictionary containing run metrics
        """
        return self.store.get_metrics(run_id)

    def list_runs(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List runs.

        Args:
            limit: Maximum number of runs to retrieve
            offset: Offset for pagination

        Returns:
            List of dictionaries containing run information
        """
        return self.store.list_runs(limit, offset)

    def search_runs(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for runs matching query criteria.

        Args:
            query: Dictionary mapping fields to search values

        Returns:
            List of dictionaries containing run information
        """
        return self.store.search_runs(query)

    def export_run_report(self, run_id: str, format: str = "json") -> str:
        """Export a detailed report for a run.

        Args:
            run_id: The ID of the run
            format: Output format ("json" or "text")

        Returns:
            Report as a string
        """
        run_info = self.store.get_run(run_id)
        if not run_info:
            return f"Run {run_id} not found"

        metrics = self.store.get_metrics(run_id)

        # Combine run info and metrics
        report = {"run": run_info, "metrics": metrics}

        if format == "json":
            return json.dumps(report, indent=2)
        else:
            # Text format
            lines = []
            lines.append(f"Run Report: {run_info.get('name', run_id)}")
            lines.append(f"ID: {run_id}")
            lines.append(f"Status: {run_info.get('status', 'unknown')}")
            lines.append(f"Started: {run_info.get('start_time', 'unknown')}")
            lines.append(f"Completed: {run_info.get('end_time', 'N/A')}")
            lines.append(f"Duration: {run_info.get('duration', 'N/A')} seconds")

            if "error" in run_info:
                lines.append(f"Error: {run_info['error']}")

            lines.append("\nConfiguration:")
            if "config" in run_info:
                for key, value in run_info["config"].items():
                    lines.append(f"  {key}: {value}")

            if "dataset_metrics" in metrics:
                dataset_metrics = metrics["dataset_metrics"]
                lines.append("\nDataset Metrics:")
                lines.append(
                    f"  Conversations: {dataset_metrics.get('num_conversations', 0)}"
                )
                lines.append(
                    f"  Total Messages: {dataset_metrics.get('total_messages', 0)}"
                )
                lines.append(
                    f"  Avg Messages/Conversation: {dataset_metrics.get('avg_messages_per_conversation', 0):.2f}"
                )

                if "message_length" in dataset_metrics:
                    msg_len = dataset_metrics["message_length"]
                    lines.append(
                        f"  Message Length (avg): {msg_len.get('avg', 0):.2f} chars"
                    )
                    lines.append(
                        f"  Message Length (min): {msg_len.get('min', 0)} chars"
                    )
                    lines.append(
                        f"  Message Length (max): {msg_len.get('max', 0)} chars"
                    )

                if "domains" in dataset_metrics:
                    lines.append("\n  Domains:")
                    for domain, count in dataset_metrics["domains"].items():
                        lines.append(f"    {domain}: {count}")

                if (
                    "topics" in dataset_metrics
                    and "top_20" in dataset_metrics["topics"]
                ):
                    lines.append("\n  Top Topics:")
                    for topic, count in dataset_metrics["topics"]["top_20"].items():
                        lines.append(f"    {topic}: {count}")

                if (
                    "entities" in dataset_metrics
                    and "top_20" in dataset_metrics["entities"]
                ):
                    lines.append("\n  Top Entities:")
                    for entity, count in dataset_metrics["entities"]["top_20"].items():
                        lines.append(f"    {entity}: {count}")

                if (
                    "sentiments" in dataset_metrics
                    and "frequency" in dataset_metrics["sentiments"]
                ):
                    lines.append("\n  Sentiments:")
                    for sentiment, count in dataset_metrics["sentiments"][
                        "frequency"
                    ].items():
                        lines.append(f"    {sentiment}: {count}")

                if (
                    "intents" in dataset_metrics
                    and "frequency" in dataset_metrics["intents"]
                ):
                    lines.append("\n  Intents:")
                    for intent, count in dataset_metrics["intents"][
                        "frequency"
                    ].items():
                        lines.append(f"    {intent}: {count}")

            return "\n".join(lines)

    def get_conversation_metrics(
        self, run_id: str, conversation_idx: int
    ) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific conversation in a run.

        Args:
            run_id: The ID of the run
            conversation_idx: Index of the conversation

        Returns:
            Dictionary containing conversation metrics, or None if not found
        """
        metrics = self.store.get_metrics(run_id)
        return metrics.get(f"conversation_{conversation_idx}")

    def save_custom_metric(
        self, run_id: str, metric_name: str, metric_value: Any
    ) -> None:
        """Save a custom metric for a run.

        Args:
            run_id: The ID of the run
            metric_name: Name of the metric
            metric_value: Value of the metric
        """
        self.store.save_metric(run_id, metric_name, metric_value)
        self.logger.info(f"Saved custom metric '{metric_name}' for run {run_id}")
