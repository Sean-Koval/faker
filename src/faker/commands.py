"""CLI commands for working with runs and metrics."""

import argparse
import datetime
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from src.faker.logging_service import JSONLogStore, LoggingService, SQLiteLogStore


def setup_logging_commands(subparsers):
    """Set up commands for working with logs.

    Args:
        subparsers: Subparsers object from argparse
    """
    # Runs command
    runs_parser = subparsers.add_parser("runs", help="List and manage runs")
    runs_subparsers = runs_parser.add_subparsers(
        dest="runs_command", help="Runs command"
    )

    # List runs
    list_parser = runs_subparsers.add_parser("list", help="List runs")
    list_parser.add_argument(
        "--limit", type=int, default=10, help="Maximum number of runs to show"
    )
    list_parser.add_argument(
        "--offset", type=int, default=0, help="Offset for pagination"
    )
    list_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    list_parser.add_argument("--log-dir", default="./logs", help="Logs directory")
    list_parser.add_argument(
        "--db", action="store_true", help="Use SQLite database for storage"
    )

    # Show run info
    show_parser = runs_subparsers.add_parser("show", help="Show run information")
    show_parser.add_argument("run_id", help="ID of the run to show")
    show_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    show_parser.add_argument("--log-dir", default="./logs", help="Logs directory")
    show_parser.add_argument(
        "--db", action="store_true", help="Use SQLite database for storage"
    )

    # Delete run
    delete_parser = runs_subparsers.add_parser("delete", help="Delete run")
    delete_parser.add_argument("run_id", help="ID of the run to delete")
    delete_parser.add_argument("--log-dir", default="./logs", help="Logs directory")
    delete_parser.add_argument(
        "--db", action="store_true", help="Use SQLite database for storage"
    )

    # Export run report
    export_parser = runs_subparsers.add_parser("export", help="Export run report")
    export_parser.add_argument("run_id", help="ID of the run to export")
    export_parser.add_argument(
        "-o", "--output", help="Output file path (default: stdout)"
    )
    export_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    export_parser.add_argument("--log-dir", default="./logs", help="Logs directory")
    export_parser.add_argument(
        "--db", action="store_true", help="Use SQLite database for storage"
    )

    # Search runs
    search_parser = runs_subparsers.add_parser("search", help="Search runs")
    search_parser.add_argument("--name", help="Run name to search for")
    search_parser.add_argument(
        "--status",
        choices=["started", "completed", "error"],
        help="Run status to search for",
    )
    search_parser.add_argument("--config-path", help="Config path to search for")
    search_parser.add_argument("--after", help="Show runs after this date (YYYY-MM-DD)")
    search_parser.add_argument(
        "--before", help="Show runs before this date (YYYY-MM-DD)"
    )
    search_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    search_parser.add_argument("--log-dir", default="./logs", help="Logs directory")
    search_parser.add_argument(
        "--db", action="store_true", help="Use SQLite database for storage"
    )

    # Metrics command
    metrics_parser = subparsers.add_parser("metrics", help="View and analyze metrics")
    metrics_subparsers = metrics_parser.add_subparsers(
        dest="metrics_command", help="Metrics command"
    )

    # List metrics
    list_metrics_parser = metrics_subparsers.add_parser(
        "list", help="List metrics for a run"
    )
    list_metrics_parser.add_argument("run_id", help="ID of the run")
    list_metrics_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    list_metrics_parser.add_argument(
        "--log-dir", default="./logs", help="Logs directory"
    )
    list_metrics_parser.add_argument(
        "--db", action="store_true", help="Use SQLite database for storage"
    )

    # Show dataset metrics
    dataset_metrics_parser = metrics_subparsers.add_parser(
        "dataset", help="Show dataset metrics for a run"
    )
    dataset_metrics_parser.add_argument("run_id", help="ID of the run")
    dataset_metrics_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    dataset_metrics_parser.add_argument(
        "--log-dir", default="./logs", help="Logs directory"
    )
    dataset_metrics_parser.add_argument(
        "--db", action="store_true", help="Use SQLite database for storage"
    )

    # Show conversation metrics
    conversation_metrics_parser = metrics_subparsers.add_parser(
        "conversation", help="Show metrics for a conversation"
    )
    conversation_metrics_parser.add_argument("run_id", help="ID of the run")
    conversation_metrics_parser.add_argument(
        "conversation_idx", type=int, help="Index of the conversation"
    )
    conversation_metrics_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    conversation_metrics_parser.add_argument(
        "--log-dir", default="./logs", help="Logs directory"
    )
    conversation_metrics_parser.add_argument(
        "--db", action="store_true", help="Use SQLite database for storage"
    )

    # Compare runs
    compare_parser = metrics_subparsers.add_parser(
        "compare", help="Compare metrics between runs"
    )
    compare_parser.add_argument("run_ids", nargs="+", help="IDs of the runs to compare")
    compare_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    compare_parser.add_argument("--log-dir", default="./logs", help="Logs directory")
    compare_parser.add_argument(
        "--db", action="store_true", help="Use SQLite database for storage"
    )


def get_logging_service(args) -> LoggingService:
    """Get a logging service based on command line arguments.

    Args:
        args: Command line arguments

    Returns:
        LoggingService instance
    """
    log_dir = args.log_dir

    # Create the appropriate log store based on arguments
    store: Union[JSONLogStore, SQLiteLogStore]
    if args.db:
        db_path = os.path.join(log_dir, "runs.db")
        store = SQLiteLogStore(db_path=db_path)
    else:
        store = JSONLogStore(base_dir=log_dir)

    return LoggingService(store=store, log_dir=log_dir)


def handle_runs_command(args):
    """Handle the runs command.

    Args:
        args: Command line arguments

    Returns:
        Command result
    """
    logging_service = get_logging_service(args)

    if args.runs_command == "list":
        runs = logging_service.list_runs(limit=args.limit, offset=args.offset)

        if args.format == "json":
            return json.dumps(runs, indent=2)
        else:
            if not runs:
                return "No runs found"

            lines = ["Runs:"]
            for run in runs:
                lines.append(f"  ID: {run['id']}")
                lines.append(f"    Name: {run.get('name', 'Unnamed run')}")
                lines.append(f"    Timestamp: {run.get('timestamp', 'N/A')}")
                lines.append(f"    Config: {run.get('config_path', 'N/A')}")
                lines.append("")

            return "\n".join(lines)

    elif args.runs_command == "show":
        run_info = logging_service.get_run_info(args.run_id)

        if run_info is None:
            return f"Run {args.run_id} not found"

        if args.format == "json":
            return json.dumps(run_info, indent=2)
        else:
            lines = [f"Run: {run_info.get('name', args.run_id)}"]
            lines.append(f"ID: {args.run_id}")
            lines.append(f"Status: {run_info.get('status', 'unknown')}")
            lines.append(f"Started: {run_info.get('start_time', 'unknown')}")
            lines.append(f"Completed: {run_info.get('end_time', 'N/A')}")

            if "duration" in run_info:
                lines.append(f"Duration: {run_info['duration']} seconds")

            if "error" in run_info:
                lines.append(f"Error: {run_info['error']}")

            lines.append("\nConfiguration:")
            if "config" in run_info:
                for key, value in run_info["config"].items():
                    if key != "config_path":
                        lines.append(f"  {key}: {value}")

            return "\n".join(lines)

    elif args.runs_command == "delete":
        # Note: This would need to be implemented in the LogStore classes
        return "Delete functionality not yet implemented"

    elif args.runs_command == "export":
        report = logging_service.export_run_report(args.run_id, format=args.format)

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            return f"Report exported to {args.output}"
        else:
            return report

    elif args.runs_command == "search":
        query = {}

        if args.name:
            query["name"] = args.name

        if args.status:
            query["status"] = args.status

        if args.config_path:
            query["config_path"] = args.config_path

        # Handle date filtering (simple approximation - a real implementation would be more sophisticated)
        if args.after or args.before:
            runs = logging_service.list_runs(limit=1000)
            filtered_runs = []

            after_date = None
            if args.after:
                after_date = datetime.datetime.strptime(args.after, "%Y-%m-%d").date()

            before_date = None
            if args.before:
                before_date = datetime.datetime.strptime(args.before, "%Y-%m-%d").date()

            for run in runs:
                if "timestamp" in run:
                    try:
                        run_date = datetime.datetime.fromisoformat(
                            run["timestamp"]
                        ).date()

                        if after_date and run_date < after_date:
                            continue

                        if before_date and run_date > before_date:
                            continue

                        if all(run.get(k) == v for k, v in query.items()):
                            filtered_runs.append(run)
                    except (ValueError, TypeError):
                        pass

            runs = filtered_runs
        else:
            runs = logging_service.search_runs(query)

        if args.format == "json":
            return json.dumps(runs, indent=2)
        else:
            if not runs:
                return "No runs found matching criteria"

            lines = ["Matching runs:"]
            for run in runs:
                lines.append(f"  ID: {run['id']}")
                lines.append(f"    Name: {run.get('name', 'Unnamed run')}")
                lines.append(f"    Timestamp: {run.get('timestamp', 'N/A')}")
                lines.append(f"    Config: {run.get('config_path', 'N/A')}")
                lines.append("")

            return "\n".join(lines)

    return "Unknown runs command"


def handle_metrics_command(args):
    """Handle the metrics command.

    Args:
        args: Command line arguments

    Returns:
        Command result
    """
    logging_service = get_logging_service(args)

    if args.metrics_command == "list":
        metrics = logging_service.get_run_metrics(args.run_id)

        if not metrics:
            return f"No metrics found for run {args.run_id}"

        if args.format == "json":
            return json.dumps(metrics, indent=2)
        else:
            lines = [f"Metrics for run {args.run_id}:"]

            # Show metric keys
            for key in metrics.keys():
                lines.append(f"  {key}")

            return "\n".join(lines)

    elif args.metrics_command == "dataset":
        metrics = logging_service.get_run_metrics(args.run_id)
        dataset_metrics = metrics.get("dataset_metrics")

        if not dataset_metrics:
            return f"No dataset metrics found for run {args.run_id}"

        if args.format == "json":
            return json.dumps(dataset_metrics, indent=2)
        else:
            lines = [f"Dataset metrics for run {args.run_id}:"]

            # Basic statistics
            lines.append("\nBasic statistics:")
            lines.append(
                f"  Conversations: {dataset_metrics.get('num_conversations', 0)}"
            )
            lines.append(
                f"  Total messages: {dataset_metrics.get('total_messages', 0)}"
            )
            lines.append(
                f"  Average messages per conversation: {dataset_metrics.get('avg_messages_per_conversation', 0):.2f}"
            )

            # Message length
            if "message_length" in dataset_metrics:
                msg_len = dataset_metrics["message_length"]
                lines.append("\nMessage length:")
                lines.append(f"  Average: {msg_len.get('avg', 0):.2f} characters")
                lines.append(f"  Minimum: {msg_len.get('min', 0)} characters")
                lines.append(f"  Maximum: {msg_len.get('max', 0)} characters")

            # Domains
            if "domains" in dataset_metrics and dataset_metrics["domains"]:
                lines.append("\nDomains:")
                for domain, count in dataset_metrics["domains"].items():
                    lines.append(f"  {domain}: {count}")

            # Roles
            if "roles" in dataset_metrics and dataset_metrics["roles"]:
                lines.append("\nRoles:")
                for role in dataset_metrics["roles"]:
                    lines.append(f"  {role}")

            # Speakers by role
            if (
                "speakers_by_role" in dataset_metrics
                and dataset_metrics["speakers_by_role"]
            ):
                lines.append("\nSpeakers by role:")
                for role, count in dataset_metrics["speakers_by_role"].items():
                    lines.append(f"  {role}: {count}")

            # Topics
            if "topics" in dataset_metrics and "top_20" in dataset_metrics["topics"]:
                lines.append(
                    f"\nTopics (total: {dataset_metrics['topics'].get('count', 0)}):"
                )
                lines.append("  Top 20:")
                for topic, count in dataset_metrics["topics"]["top_20"].items():
                    lines.append(f"    {topic}: {count}")

            # Entities
            if (
                "entities" in dataset_metrics
                and "top_20" in dataset_metrics["entities"]
            ):
                lines.append(
                    f"\nEntities (total: {dataset_metrics['entities'].get('count', 0)}):"
                )
                lines.append("  Top 20:")
                for entity, count in dataset_metrics["entities"]["top_20"].items():
                    lines.append(f"    {entity}: {count}")

            # Sentiments
            if (
                "sentiments" in dataset_metrics
                and "frequency" in dataset_metrics["sentiments"]
            ):
                lines.append(
                    f"\nSentiments (total: {dataset_metrics['sentiments'].get('count', 0)}):"
                )
                for sentiment, count in dataset_metrics["sentiments"][
                    "frequency"
                ].items():
                    lines.append(f"  {sentiment}: {count}")

            # Intents
            if (
                "intents" in dataset_metrics
                and "frequency" in dataset_metrics["intents"]
            ):
                lines.append(
                    f"\nIntents (total: {dataset_metrics['intents'].get('count', 0)}):"
                )
                for intent, count in dataset_metrics["intents"]["frequency"].items():
                    lines.append(f"  {intent}: {count}")

            # Formalities
            if "formalities" in dataset_metrics and dataset_metrics["formalities"]:
                lines.append("\nFormality levels:")
                for formality in dataset_metrics["formalities"]:
                    lines.append(f"  {formality}")

            return "\n".join(lines)

    elif args.metrics_command == "conversation":
        metrics = logging_service.get_conversation_metrics(
            args.run_id, args.conversation_idx
        )

        if not metrics:
            return f"No metrics found for conversation {args.conversation_idx} in run {args.run_id}"

        if args.format == "json":
            return json.dumps(metrics, indent=2)
        else:
            lines = [
                f"Metrics for conversation {args.conversation_idx} in run {args.run_id}:"
            ]
            lines.append(f"  ID: {metrics.get('id', 'unknown')}")
            lines.append(f"  Domain: {metrics.get('domain', 'unknown')}")
            lines.append(f"  Messages: {metrics.get('num_messages', 0)}")
            lines.append(f"  Speakers: {metrics.get('num_speakers', 0)}")
            lines.append(
                f"  Average message length: {metrics.get('avg_message_length', 0):.2f} characters"
            )

            # Messages by role
            if "messages_by_role" in metrics and metrics["messages_by_role"]:
                lines.append("\n  Messages by role:")
                for role, count in metrics["messages_by_role"].items():
                    lines.append(f"    {role}: {count}")

            # Sentiment distribution
            if (
                "sentiment_distribution" in metrics
                and metrics["sentiment_distribution"]
            ):
                lines.append("\n  Sentiment distribution:")
                for sentiment, count in metrics["sentiment_distribution"].items():
                    lines.append(f"    {sentiment}: {count}")

            # Intent distribution
            if "intent_distribution" in metrics and metrics["intent_distribution"]:
                lines.append("\n  Intent distribution:")
                for intent, count in metrics["intent_distribution"].items():
                    lines.append(f"    {intent}: {count}")

            # Topics
            if "topics" in metrics and metrics["topics"]:
                lines.append("\n  Topics:")
                for topic in metrics["topics"]:
                    lines.append(f"    {topic}")

            # Entities
            if "entities" in metrics and metrics["entities"]:
                lines.append("\n  Entities:")
                for entity in metrics["entities"]:
                    lines.append(f"    {entity}")

            return "\n".join(lines)

    elif args.metrics_command == "compare":
        if len(args.run_ids) < 2:
            return "At least two run IDs are required for comparison"

        run_metrics = {}
        for run_id in args.run_ids:
            metrics = logging_service.get_run_metrics(run_id)
            dataset_metrics = metrics.get("dataset_metrics")
            if dataset_metrics:
                run_info = logging_service.get_run_info(run_id)
                run_name = run_info.get("name", run_id) if run_info else run_id
                run_metrics[run_name] = dataset_metrics

        if not run_metrics:
            return "No metrics found for the specified runs"

        if args.format == "json":
            return json.dumps(run_metrics, indent=2)
        else:
            lines = ["Run comparison:"]

            # Basic statistics comparison
            lines.append("\nBasic statistics:")

            # Create a comparison table
            headers = ["Metric"] + list(run_metrics.keys())
            rows = []

            # Number of conversations
            row = ["Conversations"]
            for run_name in headers[1:]:
                row.append(str(run_metrics[run_name].get("num_conversations", 0)))
            rows.append(row)

            # Total messages
            row = ["Total messages"]
            for run_name in headers[1:]:
                row.append(str(run_metrics[run_name].get("total_messages", 0)))
            rows.append(row)

            # Average messages per conversation
            row = ["Avg messages/conv"]
            for run_name in headers[1:]:
                row.append(
                    f"{run_metrics[run_name].get('avg_messages_per_conversation', 0):.2f}"
                )
            rows.append(row)

            # Message length
            row = ["Avg message length"]
            for run_name in headers[1:]:
                msg_len = run_metrics[run_name].get("message_length", {})
                row.append(f"{msg_len.get('avg', 0):.2f}")
            rows.append(row)

            # Format the table
            col_widths = [
                max(len(row[i]) for row in [headers] + rows)
                for i in range(len(headers))
            ]

            # Print header
            header_row = " | ".join(
                headers[i].ljust(col_widths[i]) for i in range(len(headers))
            )
            lines.append(header_row)
            lines.append("-" * len(header_row))

            # Print rows
            for row in rows:
                lines.append(
                    " | ".join(row[i].ljust(col_widths[i]) for i in range(len(row)))
                )

            return "\n".join(lines)

    return "Unknown metrics command"


def handle_logging_command(args):
    """Handle logging commands.

    Args:
        args: Command line arguments

    Returns:
        Command result
    """
    if hasattr(args, "runs_command") and args.runs_command:
        return handle_runs_command(args)
    elif hasattr(args, "metrics_command") and args.metrics_command:
        return handle_metrics_command(args)
    else:
        return "No command specified"
