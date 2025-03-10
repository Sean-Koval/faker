"""Command-line interface for Faker."""

import os
import sys
import json
import argparse
import logging
import time
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from faker import ChatGenerator, Dataset


def setup_logging(level=logging.INFO, log_file=None):
    """Set up logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional path to log file
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Faker: Synthetic Chat Data Generator")
    
    # Create subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate a dataset")
    generate_parser.add_argument(
        "-c", "--config", 
        required=True,
        help="Path to configuration file"
    )
    generate_parser.add_argument(
        "-n", "--num-conversations",
        type=int,
        help="Number of conversations to generate (overrides config)"
    )
    generate_parser.add_argument(
        "-o", "--output-dir",
        help="Output directory (overrides config)"
    )
    generate_parser.add_argument(
        "-f", "--format",
        choices=["json", "jsonl", "split"],
        default="split",
        help="Output format (overrides config)"
    )
    generate_parser.add_argument(
        "--log-file",
        help="Log file path"
    )
    generate_parser.add_argument(
        "--no-export-run-info",
        action="store_true",
        help="Don't export run information"
    )
    
    # View command
    view_parser = subparsers.add_parser("view", help="View an existing dataset")
    view_parser.add_argument(
        "file",
        help="Path to dataset file or directory"
    )
    view_parser.add_argument(
        "-i", "--index",
        type=int,
        help="Index of conversation to view"
    )
    view_parser.add_argument(
        "--id",
        help="ID of conversation to view"
    )
    view_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format"
    )
    view_parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Show only metadata without conversations"
    )
    view_parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistical information about the dataset"
    )
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show information about the system")
    info_parser.add_argument(
        "--check-credentials",
        action="store_true",
        help="Check if credentials are properly configured"
    )
    info_parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configuration files"
    )
    info_parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets"
    )
    
    return parser.parse_args()


def format_conversation(conversation, format="text", wrap_width=100):
    """Format a conversation for display.
    
    Args:
        conversation: Conversation object
        format: Output format (text or json)
        wrap_width: Text wrap width for content
        
    Returns:
        Formatted string
    """
    if format == "json":
        return json.dumps(conversation.to_dict(), indent=2)
    
    # Text format
    lines = []
    lines.append(f"Conversation ID: {conversation.id}")
    lines.append(f"Created: {conversation.created_at}")
    if conversation.title:
        lines.append(f"Title: {conversation.title}")
    if conversation.domain:
        lines.append(f"Domain: {conversation.domain}")
    if conversation.topics:
        lines.append(f"Topics: {', '.join(conversation.topics)}")
    
    lines.append("\nParticipants:")
    for id, speaker in conversation.speakers.items():
        lines.append(f"  {speaker.name} (role: {speaker.role}, id: {id})")
    
    lines.append("\nMessages:")
    for i, msg in enumerate(conversation.messages):
        speaker = conversation.speakers.get(msg.speaker_id, None)
        speaker_name = speaker.name if speaker else "Unknown"
        
        # Add message number and speaker
        lines.append(f"\n[{i+1}] {speaker_name} ({msg.timestamp.strftime('%H:%M:%S')}):")
        
        # Wrap and indent message content
        content_lines = textwrap.wrap(msg.content, width=wrap_width)
        for line in content_lines:
            lines.append(f"    {line}")
        
        # Add metadata if available
        metadata = []
        if msg.sentiment:
            metadata.append(f"sentiment: {msg.sentiment}")
        if msg.intent:
            metadata.append(f"intent: {msg.intent}")
        if msg.entities:
            metadata.append(f"entities: {', '.join(msg.entities)}")
        if msg.topics:
            metadata.append(f"topics: {', '.join(msg.topics)}")
        if msg.formality:
            metadata.append(f"formality: {msg.formality}")
            
        if metadata:
            lines.append(f"    ({'; '.join(metadata)})")
    
    return "\n".join(lines)


def format_dataset_stats(dataset):
    """Generate statistical information about a dataset.
    
    Args:
        dataset: Dataset object
        
    Returns:
        Formatted string with statistics
    """
    # Basic statistics
    num_conversations = len(dataset.conversations)
    total_messages = sum(len(conv.messages) for conv in dataset.conversations)
    avg_messages = total_messages / num_conversations if num_conversations > 0 else 0
    
    # Collect all roles, topics, entities
    all_roles = set()
    all_topics = set()
    all_entities = set()
    all_intents = set()
    all_sentiments = set()
    all_formalities = set()
    
    # Count by domain
    domains = {}
    
    for conv in dataset.conversations:
        # Add domain count
        if conv.domain:
            domains[conv.domain] = domains.get(conv.domain, 0) + 1
            
        # Add topics and entities
        if conv.topics:
            all_topics.update(conv.topics)
        if conv.entities:
            all_entities.update(conv.entities)
            
        # Add speaker roles
        for speaker in conv.speakers.values():
            all_roles.add(speaker.role)
            
        # Add message metadata
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
                all_entities.update(msg.entities)
    
    # Format results
    lines = []
    lines.append(f"Dataset: {dataset.name} (v{dataset.version})")
    if dataset.description:
        lines.append(f"Description: {dataset.description}")
    lines.append(f"Created: {dataset.created_at}")
    lines.append("")
    
    lines.append("Basic Statistics:")
    lines.append(f"  Conversations: {num_conversations}")
    lines.append(f"  Total Messages: {total_messages}")
    lines.append(f"  Average Messages per Conversation: {avg_messages:.2f}")
    
    if domains:
        lines.append("\nDomains:")
        for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {domain}: {count} ({count/num_conversations*100:.1f}%)")
    
    if all_roles:
        lines.append("\nRoles:")
        for role in sorted(all_roles):
            lines.append(f"  {role}")
    
    if all_sentiments:
        lines.append("\nSentiments:")
        for sentiment in sorted(all_sentiments):
            lines.append(f"  {sentiment}")
    
    if all_intents:
        lines.append("\nIntents:")
        for intent in sorted(all_intents):
            lines.append(f"  {intent}")
    
    if all_formalities:
        lines.append("\nFormality Levels:")
        for formality in sorted(all_formalities):
            lines.append(f"  {formality}")
    
    if all_topics:
        lines.append("\nTop Topics:")
        for topic in sorted(list(all_topics))[:20]:  # Show only top 20
            lines.append(f"  {topic}")
            
    if all_entities:
        lines.append("\nTop Entities:")
        for entity in sorted(list(all_entities))[:20]:  # Show only top 20
            lines.append(f"  {entity}")
    
    # Add run information if available
    if dataset.run_info:
        lines.append("\nRun Information:")
        lines.append(f"  Run ID: {dataset.run_info.id}")
        lines.append(f"  Start Time: {dataset.run_info.start_time}")
        lines.append(f"  End Time: {dataset.run_info.end_time if dataset.run_info.end_time else 'N/A'}")
        
        if dataset.run_info.model_info:
            lines.append("  Model:")
            for key, value in dataset.run_info.model_info.items():
                lines.append(f"    {key}: {value}")
        
        if dataset.run_info.stats:
            lines.append("  Statistics:")
            for key, value in dataset.run_info.stats.items():
                lines.append(f"    {key}: {value}")
    
    return "\n".join(lines)


def check_credentials():
    """Check if credentials are properly configured.
    
    Returns:
        A tuple of (success, message)
    """
    # Check for PROJECT_ID
    project_id = os.getenv("PROJECT_ID")
    if not project_id:
        return False, "PROJECT_ID environment variable is not set"
    
    # Check for GOOGLE_APPLICATION_CREDENTIALS
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        return False, "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set"
    
    # Check if the credentials file exists
    if not os.path.exists(credentials_path):
        return False, f"Credentials file not found: {credentials_path}"
    
    # Try to read the credentials file
    try:
        with open(credentials_path, 'r') as f:
            json.load(f)
    except Exception as e:
        return False, f"Failed to read credentials file: {e}"
    
    return True, "Credentials are properly configured"


def list_configs(base_dir="./examples"):
    """List available configuration files.
    
    Args:
        base_dir: Base directory to search for configs
        
    Returns:
        A list of config files
    """
    if not os.path.exists(base_dir):
        return []
    
    configs = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(('.yaml', '.yml')):
                configs.append(os.path.join(root, file))
    
    return configs


def list_datasets(base_dir="./output"):
    """List available datasets.
    
    Args:
        base_dir: Base directory to search for datasets
        
    Returns:
        A list of datasets
    """
    if not os.path.exists(base_dir):
        return []
    
    datasets = []
    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        
        # Check if it's a directory with a metadata.json file (split format)
        if os.path.isdir(entry_path) and os.path.exists(os.path.join(entry_path, 'metadata.json')):
            datasets.append(entry_path)
            continue
            
        # Check if it's a JSON or JSONL file
        if os.path.isfile(entry_path) and entry.endswith(('.json', '.jsonl')):
            datasets.append(entry_path)
    
    return datasets


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Setup logging
    log_file = args.log_file if hasattr(args, 'log_file') and args.log_file else None
    setup_logging(log_file=log_file)
    logger = logging.getLogger(__name__)
    
    if args.command == "generate":
        logger.info(f"Generating dataset using config: {args.config}")
        
        try:
            # Create generator
            generator = ChatGenerator.from_config(args.config)
            
            # Override output directory if provided
            if args.output_dir:
                generator.output_dir = args.output_dir
                
            # Override number of conversations if provided
            num_conversations = args.num_conversations
            if num_conversations is None:
                # Try to get from config
                num_conversations = generator.config.get('dataset', {}).get('num_conversations', 1)
            
            # Start generation
            start_time = time.time()
            logger.info(f"Starting generation of {num_conversations} conversations")
            
            # Generate dataset
            dataset = generator.generate(
                num_conversations=num_conversations,
                output_format=args.format,
                export_run_info=not args.no_export_run_info
            )
            
            # Report total time
            total_time = time.time() - start_time
            logger.info(f"Generation completed in {total_time:.2f} seconds")
            logger.info(f"Generated {len(dataset.conversations)} conversations")
            
        except Exception as e:
            logger.error(f"Error generating dataset: {e}", exc_info=True)
            return 1
    
    elif args.command == "view":
        logger.info(f"Viewing dataset: {args.file}")
        
        try:
            # Load dataset
            dataset = Dataset.from_file(args.file)
            
            if args.stats:
                # Show dataset statistics
                print(format_dataset_stats(dataset))
                return 0
                
            if args.metadata_only:
                # Show dataset metadata
                metadata = {
                    'name': dataset.name,
                    'version': dataset.version,
                    'description': dataset.description,
                    'created_at': dataset.created_at.isoformat(),
                    'tags': dataset.tags,
                    'metadata': dataset.metadata,
                    'run_info': dataset.run_info.to_dict() if dataset.run_info else None,
                    'num_conversations': len(dataset.conversations)
                }
                
                if args.format == "json":
                    print(json.dumps(metadata, indent=2))
                else:
                    print(f"Dataset: {dataset.name} (v{dataset.version})")
                    if dataset.description:
                        print(f"Description: {dataset.description}")
                    print(f"Created: {dataset.created_at}")
                    print(f"Conversations: {len(dataset.conversations)}")
                    if dataset.tags:
                        print(f"Tags: {', '.join(dataset.tags)}")
                    if dataset.run_info:
                        print(f"Run ID: {dataset.run_info.id}")
                        print(f"Run Start: {dataset.run_info.start_time}")
                        print(f"Run End: {dataset.run_info.end_time}")
                
                return 0
            
            # View specific conversation by ID
            if args.id:
                conversation = next((c for c in dataset.conversations if c.id == args.id), None)
                if conversation:
                    print(format_conversation(conversation, format=args.format))
                else:
                    logger.error(f"Conversation with ID {args.id} not found")
                    return 1
            
            # View specific conversation by index
            elif args.index is not None:
                if 0 <= args.index < len(dataset.conversations):
                    conversation = dataset.conversations[args.index]
                    print(format_conversation(conversation, format=args.format))
                else:
                    logger.error(f"Invalid conversation index: {args.index}")
                    return 1
            
            # Show summary of all conversations
            else:
                print(f"Dataset: {dataset.name} (v{dataset.version})")
                if dataset.description:
                    print(f"Description: {dataset.description}")
                print(f"Contains {len(dataset.conversations)} conversations")
                
                print("\nConversations:")
                for i, conv in enumerate(dataset.conversations):
                    num_messages = len(conv.messages)
                    speakers = len(conv.speakers)
                    domain = f", domain: {conv.domain}" if conv.domain else ""
                    
                    print(f"{i}: ID {conv.id} ({num_messages} messages, {speakers} speakers{domain})")
        
        except Exception as e:
            logger.error(f"Error viewing dataset: {e}", exc_info=True)
            return 1
    
    elif args.command == "info":
        # Show system information
        
        if args.check_credentials:
            # Check if credentials are properly configured
            success, message = check_credentials()
            print(f"Credentials check: {'✅ Success' if success else '❌ Failed'}")
            print(message)
        
        if args.list_configs:
            # List available configuration files
            configs = list_configs()
            if configs:
                print(f"Found {len(configs)} configuration files:")
                for config in configs:
                    print(f"  {config}")
            else:
                print("No configuration files found in ./examples")
                print("You can create a configuration file or copy an example from the documentation.")
        
        if args.list_datasets:
            # List available datasets
            datasets = list_datasets()
            if datasets:
                print(f"Found {len(datasets)} datasets:")
                for dataset in datasets:
                    print(f"  {dataset}")
            else:
                print("No datasets found in ./output")
                print("Generate a dataset with 'python -m faker generate -c <config>'")
        
        # If no specific info requested, show general info
        if not any([args.check_credentials, args.list_configs, args.list_datasets]):
            print("Faker: Synthetic Chat Data Generator")
            print(f"Python version: {sys.version}")
            print(f"Current working directory: {os.getcwd()}")
            print("\nAvailable commands:")
            print("  generate  Generate synthetic chat data")
            print("  view      View existing chat data")
            print("  info      Show system information")
            print("\nFor more information, run 'python -m faker <command> --help'")
    
    else:
        logger.error("No command specified. Run 'python -m faker --help' for usage information.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())