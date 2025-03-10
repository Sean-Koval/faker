"""Command-line interface for Faker."""

import os
import sys
import argparse
import logging
from pathlib import Path

from faker import ChatGenerator, Dataset


def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
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
        "-o", "--output",
        help="Output file path (overrides config)"
    )
    generate_parser.add_argument(
        "-f", "--format",
        choices=["json", "jsonl"],
        default="jsonl",
        help="Output format (overrides config)"
    )
    
    # View command
    view_parser = subparsers.add_parser("view", help="View an existing dataset")
    view_parser.add_argument(
        "file",
        help="Path to dataset file"
    )
    view_parser.add_argument(
        "-i", "--index",
        type=int,
        help="Index of conversation to view"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    args = parse_args()
    
    if args.command == "generate":
        logger.info(f"Generating dataset using config: {args.config}")
        
        try:
            generator = ChatGenerator.from_config(args.config)
            
            # Override config with command-line args if provided
            num_conversations = args.num_conversations or 1
            
            # Generate the dataset
            dataset = generator.generate(num_conversations=num_conversations)
            
            # Determine output path
            if args.output:
                output_path = args.output
            else:
                output_dir = os.getenv("OUTPUT_DIR", "./output")
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"dataset_{num_conversations}.{args.format}")
            
            # Export the dataset
            dataset.export(output_path, format=args.format)
            logger.info(f"Dataset exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating dataset: {e}")
            return 1
    
    elif args.command == "view":
        logger.info(f"Viewing dataset: {args.file}")
        
        try:
            dataset = Dataset.from_file(args.file)
            
            if args.index is not None:
                # View a specific conversation
                if 0 <= args.index < len(dataset.conversations):
                    conversation = dataset.conversations[args.index]
                    print(f"Conversation {args.index} (ID: {conversation.id}):")
                    for i, msg in enumerate(conversation.messages):
                        print(f"[{msg.role}] {msg.content}")
                else:
                    logger.error(f"Invalid conversation index: {args.index}")
                    return 1
            else:
                # Show summary
                print(f"Dataset contains {len(dataset.conversations)} conversations")
                print(f"Metadata: {dataset.metadata}")
                print("\nConversation IDs:")
                for i, conv in enumerate(dataset.conversations):
                    print(f"{i}: {conv.id} ({len(conv.messages)} messages)")
        
        except Exception as e:
            logger.error(f"Error viewing dataset: {e}")
            return 1
    
    else:
        logger.error("No command specified")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())