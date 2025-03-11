"""Core module for generating synthetic conversations."""

import datetime
import json
import logging
import os
import random
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml

from src.faker.llm import GeminiProvider, LLMProvider, MockProvider
from src.faker.models import Conversation, Dataset, Message, RunInfo, Speaker
from src.faker.templates import TemplateEngine


class ChatGenerator:
    """Main class for generating synthetic chat data.

    This class handles configuration, LLM interactions, and dataset generation.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        templates: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
    ):
        """Initialize a new chat generator.

        Args:
            llm_provider: The LLM provider to use for generation
            templates: Dictionary of prompt templates for different generation scenarios
            config: Additional configuration parameters
            output_dir: Directory to store outputs
        """
        self.llm = llm_provider or GeminiProvider()
        self.templates = templates or {}
        self.config = config or {}
        self.output_dir = output_dir or os.getenv("OUTPUT_DIR", "./output")
        self.template_engine = TemplateEngine(self.templates)
        self.logger = logging.getLogger(__name__)

    @classmethod
    def from_config(cls, config_path: str) -> "ChatGenerator":
        """Create a new ChatGenerator from a configuration file.

        Args:
            config_path: Path to a YAML configuration file

        Returns:
            A configured ChatGenerator instance
        """
        config_path = os.path.abspath(config_path)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Replace environment variables in config
        config = cls._replace_env_vars(config)

        templates = config.get("templates", {})
        llm_config = config.get("llm", {})
        output_dir = config.get("dataset", {}).get("output_dir", "./output")

        # Initialize the appropriate LLM provider
        provider_name = llm_config.get("provider", "gemini")
        llm_provider: Union[GeminiProvider, MockProvider]
        if provider_name == "gemini":
            llm_provider = GeminiProvider(
                project_id=llm_config.get("project_id"),
                location=llm_config.get("location"),
                model=llm_config.get("model", "gemini-1.5-pro"),
                credentials_path=llm_config.get("credentials_path"),
                default_temperature=llm_config.get("temperature", 0.7),
                default_top_p=llm_config.get("top_p", 0.95),
                default_top_k=llm_config.get("top_k", 40),
                default_max_tokens=llm_config.get("max_tokens", 1024),
            )
        elif provider_name == "mock":
            # For testing without API calls
            llm_provider = MockProvider(
                responses=llm_config.get("responses", {}),
                default_response=llm_config.get(
                    "default_response",
                    '{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}',
                ),
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

        return cls(
            llm_provider=llm_provider,
            templates=templates,
            config=config,
            output_dir=output_dir,
        )

    @staticmethod
    def _replace_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
        """Replace environment variables in configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Configuration with environment variables replaced
        """
        # Convert to JSON and back to handle nested dictionaries
        config_str = json.dumps(config)

        # Replace ${VAR} or $VAR with environment variable
        def replace_env_var(match):
            var_name = match.group(1) or match.group(2)
            return os.getenv(var_name, match.group(0))

        config_str = re.sub(
            r"\${([^}]+)}|\$([A-Za-z0-9_]+)", replace_env_var, config_str
        )

        return json.loads(config_str)

    def generate(
        self,
        num_conversations: int = 1,
        output_format: str = "split",
        export_run_info: bool = True,
        logging_service=None,
        batch_size: int = 0,  # 0 means no batching
        max_workers: int = 5,
    ) -> Dataset:
        """Generate a dataset of synthetic conversations.

        Args:
            num_conversations: Number of conversations to generate
            output_format: Format to export dataset ('jsonl', 'json', 'split')
            export_run_info: Whether to export run information
            logging_service: Optional LoggingService for tracking metrics
            batch_size: Number of conversations to generate in parallel (0 = sequential)
            max_workers: Maximum number of parallel workers when using batching

        Returns:
            A Dataset object containing the generated conversations
        """
        # Create dataset with run information
        dataset = Dataset(
            conversations=[],
            name=self.config.get("name", "synthetic_dataset"),
            version=self.config.get("version", "1.0.0"),
            description=self.config.get("description", "Synthetic chat data"),
            tags=self.config.get("tags", []),
        )

        # Add run information
        dataset.add_run_info(
            config_path=self.config.get("config_path"), config=self.config
        )

        # Add model information
        if dataset.run_info:
            dataset.run_info.model_info = self.llm.get_model_info()

        # Initialize run in logging service if provided
        run_id = None
        if logging_service:
            run_id = logging_service.init_run(
                config=self.config,
                name=f"Generate {num_conversations} conversations - {dataset.name}",
            )
            self.logger.info(f"Initialized run with ID: {run_id}")

        # Start timing
        start_time = time.time()

        try:
            # Determine if we should use parallel processing
            use_parallel = (batch_size > 1) and (num_conversations > 1)
            actual_batch_size = min(batch_size, max_workers, num_conversations) if use_parallel else 1
            
            # Adjust based on available libraries
            if use_parallel:
                try:
                    import concurrent.futures
                    self.logger.info(f"Using parallel processing with batch size {actual_batch_size}")
                except ImportError:
                    use_parallel = False
                    self.logger.warning("concurrent.futures not available, using sequential processing")
            
            # Generate conversations (parallel or sequential)
            conversations = []
            
            if use_parallel:
                # Process in batches to control memory usage and provide progress updates
                for batch_start in range(0, num_conversations, actual_batch_size):
                    batch_end = min(batch_start + actual_batch_size, num_conversations)
                    batch_indices = list(range(batch_start, batch_end))
                    
                    self.logger.info(f"Generating conversations {batch_start+1}-{batch_end}/{num_conversations}")
                    
                    # Generate batch in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_indices)) as executor:
                        # Create tasks for conversation generation
                        future_to_idx = {
                            executor.submit(self._generate_conversation): idx 
                            for idx in batch_indices
                        }
                        
                        # Collect results as they complete
                        batch_conversations = [None] * len(batch_indices)
                        for future in concurrent.futures.as_completed(future_to_idx):
                            idx = future_to_idx[future]
                            rel_idx = idx - batch_start  # Relative index in the batch
                            
                            try:
                                conversation = future.result()
                                batch_conversations[rel_idx] = conversation
                                
                                # Log progress if using logging service
                                if logging_service and run_id:
                                    progress = (idx + 1) / num_conversations * 100
                                    logging_service.save_custom_metric(run_id, "progress", progress)
                                    
                            except Exception as e:
                                self.logger.error(f"Error generating conversation {idx+1}: {e}")
                                if logging_service and run_id:
                                    logging_service.save_custom_metric(
                                        run_id,
                                        f"error_conversation_{idx}",
                                        {"error": str(e), "index": idx},
                                    )
                    
                    # Add completed batch to conversations
                    conversations.extend([c for c in batch_conversations if c is not None])
                    
                    # Log batch completion
                    self.logger.info(f"Completed batch {batch_start+1}-{batch_end}, total: {len(conversations)}")
            
            else:
                # Sequential generation
                for i in range(num_conversations):
                    self.logger.info(f"Generating conversation {i+1}/{num_conversations}")
                    try:
                        conversation = self._generate_conversation()
                        conversations.append(conversation)

                        # Log progress if using logging service
                        if logging_service and run_id:
                            progress = (i + 1) / num_conversations * 100
                            logging_service.save_custom_metric(run_id, "progress", progress)

                    except Exception as e:
                        self.logger.error(f"Error generating conversation {i+1}: {e}")
                        if logging_service and run_id:
                            logging_service.save_custom_metric(
                                run_id,
                                f"error_conversation_{i}",
                                {"error": str(e), "index": i},
                            )

            # Update dataset
            dataset.conversations = conversations

            # Complete run information
            if dataset.run_info:
                dataset.run_info.complete()
                dataset.run_info.add_stat("num_conversations", len(conversations))
                dataset.run_info.add_stat("total_time", time.time() - start_time)
                dataset.run_info.add_stat(
                    "avg_time_per_conversation",
                    (
                        (time.time() - start_time) / num_conversations
                        if num_conversations > 0
                        else 0
                    ),
                )
                # Add parallel processing stats
                dataset.run_info.add_stat("used_parallel_processing", use_parallel)
                if use_parallel:
                    dataset.run_info.add_stat("batch_size", actual_batch_size)
                    dataset.run_info.add_stat("num_batches", (num_conversations + actual_batch_size - 1) // actual_batch_size)

            # Export dataset
            if self.output_dir:
                # Create timestamped directory
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                dataset_name = f"{dataset.name}_{timestamp}"
                output_path = os.path.join(self.output_dir, dataset_name)

                # Export dataset
                exported_path = dataset.export(
                    os.path.join(output_path, f"{dataset_name}.{output_format}"),
                    format=output_format,
                )
                self.logger.info(f"Dataset exported to: {exported_path}")

                # Export run information
                if export_run_info and dataset.run_info:
                    run_info_path = dataset.export_run_info(output_path)
                    self.logger.info(f"Run information exported to: {run_info_path}")

                # Save output path to logging service
                if logging_service and run_id:
                    logging_service.save_custom_metric(
                        run_id, "output_path", exported_path
                    )

            # Complete run in logging service
            if logging_service and run_id:
                logging_service.complete_run(run_id, dataset)
                self.logger.info(f"Completed run with ID: {run_id}")

            return dataset

        except Exception as e:
            # Log error in logging service
            if logging_service and run_id:
                logging_service.log_error(run_id, str(e))
                self.logger.error(f"Run {run_id} failed: {e}")
            raise

    def _generate_conversation(self) -> Conversation:
        """Generate a single conversation using the hybrid approach.

        Returns:
            A Conversation object containing messages
        """
        # Import response parser here to avoid circular imports
        from src.faker.response_parser import (
            parse_llm_response, 
            validate_conversation_messages,
            extract_conversation_from_text
        )
        
        # Get conversation parameters from config
        conv_config = self.config.get("conversation", {})
        roles = conv_config.get("roles", ["user", "assistant"])
        min_messages = conv_config.get("min_messages", 4)
        max_messages = conv_config.get("max_messages", 12)
        domains = conv_config.get("domains", ["general"])
        
        # Determine if we should use the hybrid approach (default: enabled)
        use_hybrid = conv_config.get("use_hybrid_approach", True)
        
        # Prepare context for template rendering
        context = {
            "domain": random.choice(domains),
            "min_messages": min_messages,
            "max_messages": max_messages,
            "roles": roles,
            **conv_config.get("variables", {}),
        }

        # Get system prompt if available
        system_prompt = None
        if "system_prompt" in self.templates:
            system_prompt = self.template_engine.render("system_prompt", context)

        # Render the conversation template
        prompt = self.template_engine.render("conversation", context)
        
        # Add JSON formatting instructions
        formatted_prompt = self.template_engine.add_formatting_instructions(prompt)

        # Generate the conversation using the LLM
        start_time = time.time()
        result, metadata = self.llm.generate(
            formatted_prompt,
            system_prompt=system_prompt,
            temperature=conv_config.get("temperature", 0.7),
            top_p=conv_config.get("top_p", 0.95),
            top_k=conv_config.get("top_k", 40),
            max_tokens=conv_config.get("max_tokens", 1024),
        )
        generation_time = time.time() - start_time
        
        # Parse and validate the result
        try:
            # Try to parse as structured data
            parsed_data = parse_llm_response(result)
            
            # Process based on the structure
            if isinstance(parsed_data, list):
                # Direct list of messages - validate and clean
                valid_messages, is_valid = validate_conversation_messages(
                    parsed_data, required_roles=roles
                )
                
                if not valid_messages:
                    raise ValueError("No valid messages found after parsing")
                
                if not is_valid and use_hybrid:
                    # If validation failed and hybrid is enabled, enhance each message
                    valid_messages = self._enhance_messages_with_metadata(valid_messages)
            
            elif isinstance(parsed_data, dict) and "messages" in parsed_data:
                # Messages wrapped in object - validate and clean
                valid_messages, is_valid = validate_conversation_messages(
                    parsed_data["messages"], required_roles=roles
                )
                
                if not valid_messages:
                    raise ValueError("No valid messages found after parsing")
                
                if not is_valid and use_hybrid:
                    # If validation failed and hybrid is enabled, enhance each message
                    valid_messages = self._enhance_messages_with_metadata(valid_messages)
            
            else:
                # If structure doesn't match expectations, try text extraction
                self.logger.warning(f"Unexpected result format: {result[:100]}...")
                valid_messages = extract_conversation_from_text(result, roles, max_messages)
                
                if use_hybrid:
                    # Enhance these basic messages with metadata
                    valid_messages = self._enhance_messages_with_metadata(valid_messages)
            
        except (ValueError, json.JSONDecodeError) as e:
            # Fallback to text extraction if structured parsing fails
            self.logger.warning(f"Failed to parse result as structured data: {e}")
            valid_messages = extract_conversation_from_text(result, roles, max_messages)
            
            if use_hybrid:
                # Enhance these messages with metadata
                valid_messages = self._enhance_messages_with_metadata(valid_messages)
        
        # Check if we have predefined speakers in the config
        predefined_speakers = conv_config.get("speakers", {})
        speakers = {}
        role_to_speaker = {}
        
        # Investment advisor-client conversation
        if predefined_speakers and "advisors" in predefined_speakers and "clients" in predefined_speakers:
            # We have predefined speakers - select one advisor and one client
            advisors = predefined_speakers.get("advisors", [])
            clients = predefined_speakers.get("clients", [])
            
            # Select a random advisor and client
            selected_advisor = random.choice(advisors) if advisors else None
            selected_client = random.choice(clients) if clients else None
            
            # Use the predefined speakers if available
            if selected_advisor and "advisor" in roles:
                advisor_id = selected_advisor.get("id")
                advisor_name = selected_advisor.get("name")
                advisor_metadata = selected_advisor.get("metadata", {})
                
                # Add to speaker dictionary
                speakers[advisor_id] = Speaker(
                    id=advisor_id,
                    name=advisor_name,
                    role="advisor",
                    metadata=advisor_metadata
                )
                
                # Map role to speaker ID
                role_to_speaker["advisor"] = advisor_id
                
                # Add advisor info to context for template rendering
                context["advisor_id"] = advisor_id
                context["advisor_name"] = advisor_name
            
            if selected_client and "client" in roles:
                client_id = selected_client.get("id")
                client_name = selected_client.get("name")
                client_metadata = selected_client.get("metadata", {})
                
                # Add to speaker dictionary
                speakers[client_id] = Speaker(
                    id=client_id,
                    name=client_name,
                    role="client",
                    metadata=client_metadata
                )
                
                # Map role to speaker ID
                role_to_speaker["client"] = client_id
                
                # Add client info to context for template rendering
                context["client_id"] = client_id
                context["client_name"] = client_name
        
        # Customer support conversation
        elif predefined_speakers and "support_agents" in predefined_speakers and "users" in predefined_speakers:
            # We have predefined speakers - select one support agent and one user
            support_agents = predefined_speakers.get("support_agents", [])
            users = predefined_speakers.get("users", [])
            
            # Select a random support agent and user
            selected_agent = random.choice(support_agents) if support_agents else None
            selected_user = random.choice(users) if users else None
            
            # Use the predefined speakers if available
            if selected_agent and "support_agent" in roles:
                agent_id = selected_agent.get("id")
                agent_name = selected_agent.get("name")
                agent_metadata = selected_agent.get("metadata", {})
                
                # Add to speaker dictionary
                speakers[agent_id] = Speaker(
                    id=agent_id,
                    name=agent_name,
                    role="support_agent",
                    metadata=agent_metadata
                )
                
                # Map role to speaker ID
                role_to_speaker["support_agent"] = agent_id
                
                # Add agent info to context for template rendering
                context["agent_id"] = agent_id
                context["agent_name"] = agent_name
            
            if selected_user and "user" in roles:
                user_id = selected_user.get("id")
                user_name = selected_user.get("name")
                user_metadata = selected_user.get("metadata", {})
                
                # Add to speaker dictionary
                speakers[user_id] = Speaker(
                    id=user_id,
                    name=user_name,
                    role="user",
                    metadata=user_metadata
                )
                
                # Map role to speaker ID
                role_to_speaker["user"] = user_id
                
                # Add user info to context for template rendering
                context["user_id"] = user_id
                context["user_name"] = user_name
        
        # For any roles that don't have predefined speakers, create them dynamically
        for role in roles:
            if role not in role_to_speaker:
                speaker_id = str(uuid.uuid4())
                speakers[speaker_id] = Speaker(
                    id=speaker_id, name=role.capitalize(), role=role
                )
                role_to_speaker[role] = speaker_id
                
                # Add dynamic speaker info to context
                if role == "advisor":
                    context["advisor_id"] = speaker_id
                    context["advisor_name"] = "Advisor"
                elif role == "client":
                    context["client_id"] = speaker_id
                    context["client_name"] = "Client"
        
        # Create Message objects
        messages = []
        for msg_data in valid_messages:
            role = msg_data.get("role")
            content = msg_data.get("content")
            
            # Default to first role if role is invalid
            if role not in role_to_speaker:
                role = roles[0]
                
            message = Message(
                content=content,
                speaker_id=role_to_speaker[role],
                # Add additional metadata if available
                sentiment=msg_data.get("sentiment"),
                intent=msg_data.get("intent"),
                entities=msg_data.get("entities", []),
                topics=msg_data.get("topics", []),
                language=msg_data.get("language"),
                formality=msg_data.get("formality"),
                metadata=msg_data.get("metadata", {}),
            )
            messages.append(message)
        
        # Create the conversation object
        conversation = Conversation(
            messages=messages,
            speakers=speakers,
            domain=context.get("domain"),
            generation_config=metadata.get("generation_config", {}),
            prompt_template=formatted_prompt,
            prompt_variables=context,
            model_info=metadata,
            generation_time=generation_time,
        )

        # Extract topics and entities if they're missing
        if not conversation.topics:
            conversation.topics = list(conversation.extract_topics())
        if not conversation.entities:
            conversation.entities = list(conversation.extract_entities())

        return conversation
        
    def _enhance_messages_with_metadata(self, messages: List[Dict]) -> List[Dict]:
        """Enhance messages with metadata using additional LLM calls.
        
        For each message that's missing metadata, make an LLM call to 
        generate appropriate metadata. Uses parallel processing for performance.
        
        Args:
            messages: List of basic message dictionaries
            
        Returns:
            Enhanced messages with metadata
        """
        from src.faker.response_parser import parse_llm_response
        
        # If no messages to enhance, return immediately
        if not messages:
            return []
            
        metadata_fields = ['sentiment', 'intent', 'entities', 'topics', 'formality']
        
        # Get conversation config
        conv_config = self.config.get("conversation", {})
        
        # Advanced performance configuration
        use_batch_requests = conv_config.get("use_batch_requests", True)
        max_parallel = conv_config.get("max_parallel_enhancement", 5)
        chunk_size = conv_config.get("batch_chunk_size", 10)  # Process messages in chunks
        
        # Filter messages that need enhancement
        messages_to_enhance = []
        message_indices = []
        
        for i, message in enumerate(messages):
            missing_fields = [field for field in metadata_fields 
                             if message.get(field) is None or 
                             (field in ['entities', 'topics'] and not message.get(field))]
            
            if missing_fields:
                message["_missing_fields"] = missing_fields
                messages_to_enhance.append(message)
                message_indices.append(i)
        
        # If no messages need enhancement, return original messages
        if not messages_to_enhance:
            return messages
        
        # Create a copy of messages to modify
        result_messages = messages.copy()
        
        # Check if the LLM provider supports batch generation
        if use_batch_requests and hasattr(self.llm, "generate_batch"):
            # Process messages in chunks to avoid overwhelming the API
            for i in range(0, len(messages_to_enhance), chunk_size):
                chunk = messages_to_enhance[i:i+chunk_size]
                chunk_indices = message_indices[i:i+chunk_size]
                
                # Create prompts for each message
                prompts = [
                    self._create_metadata_prompt(msg, msg["_missing_fields"])
                    for msg in chunk
                ]
                
                # Use the batch API with fixed parameters for all prompts
                batch_results = self.llm.generate_batch(
                    prompts=prompts,
                    temperature=0.5,  # Lower temperature for consistent metadata
                    max_tokens=256,   # Metadata needs fewer tokens
                    max_parallel=max_parallel,
                    use_cache=True,   # Enable caching for faster repeated runs
                )
                
                # Process results
                for j, (result_text, _) in enumerate(batch_results):
                    msg_idx = chunk_indices[j]
                    original_msg = messages[msg_idx]
                    missing_fields = chunk[j]["_missing_fields"]
                    
                    try:
                        # Parse the result
                        if result_text:
                            metadata = parse_llm_response(result_text)
                            
                            # Create enhanced message, combining original and new metadata
                            enhanced_msg = original_msg.copy()
                            
                            # Update each missing field if provided in response
                            for field in missing_fields:
                                if field in metadata:
                                    enhanced_msg[field] = metadata[field]
                                elif field == 'entities' or field == 'topics':
                                    enhanced_msg[field] = []
                            
                            # Remove temporary field
                            if "_missing_fields" in enhanced_msg:
                                del enhanced_msg["_missing_fields"]
                                
                            result_messages[msg_idx] = enhanced_msg
                    except Exception as e:
                        self.logger.warning(f"Failed to process batch result {j} for message {msg_idx}: {e}")
                        # Keep original message if enhancement fails
                        result_messages[msg_idx] = original_msg
        
        else:
            # Fall back to concurrent.futures for parallel processing
            try:
                import concurrent.futures
                from concurrent.futures import ThreadPoolExecutor
                
                # Define the enhancement function
                def enhance_message(message, idx):
                    try:
                        missing_fields = message["_missing_fields"]
                        
                        # Create and send prompt
                        enhancement_prompt = self._create_metadata_prompt(message, missing_fields)
                        result, _ = self.llm.generate(
                            enhancement_prompt,
                            temperature=0.5,
                            max_tokens=256,
                            use_cache=True,  # Enable caching
                        )
                        
                        # Parse response
                        metadata = parse_llm_response(result)
                        enhanced_msg = message.copy()
                        
                        # Update fields
                        for field in missing_fields:
                            if field in metadata:
                                enhanced_msg[field] = metadata[field]
                            elif field == 'entities' or field == 'topics':
                                enhanced_msg[field] = []
                        
                        # Remove temporary field
                        if "_missing_fields" in enhanced_msg:
                            del enhanced_msg["_missing_fields"]
                            
                        return enhanced_msg
                    except Exception as e:
                        self.logger.warning(f"Failed to enhance message {idx}: {e}")
                        if "_missing_fields" in message:
                            del message["_missing_fields"]
                        return message
                
                # Process in chunks to control memory usage
                for i in range(0, len(messages_to_enhance), chunk_size):
                    chunk = messages_to_enhance[i:i+chunk_size]
                    chunk_indices = message_indices[i:i+chunk_size]
                    
                    # Execute in parallel with controlled concurrency
                    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                        # Submit jobs
                        future_to_idx = {
                            executor.submit(enhance_message, msg, idx): idx 
                            for msg, idx in zip(chunk, chunk_indices)
                        }
                        
                        # Collect results
                        for future in concurrent.futures.as_completed(future_to_idx):
                            msg_idx = future_to_idx[future]
                            try:
                                enhanced_msg = future.result()
                                result_messages[msg_idx] = enhanced_msg
                            except Exception as e:
                                self.logger.error(f"Error in parallel message enhancement for message {msg_idx}: {e}")
                                # Use original message on failure
                                orig_msg = messages[msg_idx]
                                if "_missing_fields" in orig_msg:
                                    del orig_msg["_missing_fields"]
                                result_messages[msg_idx] = orig_msg
            
            except (ImportError, Exception) as e:
                self.logger.warning(f"Parallel processing unavailable: {e}, using sequential")
                
                # Sequential fallback processing
                for i, message in enumerate(messages_to_enhance):
                    msg_idx = message_indices[i]
                    missing_fields = message["_missing_fields"]
                    
                    try:
                        # Create prompt and generate response
                        enhancement_prompt = self._create_metadata_prompt(message, missing_fields)
                        result, _ = self.llm.generate(
                            enhancement_prompt,
                            temperature=0.5,
                            max_tokens=256,
                        )
                        
                        # Parse result
                        metadata = parse_llm_response(result)
                        enhanced_msg = message.copy()
                        
                        # Update fields
                        for field in missing_fields:
                            if field in metadata:
                                enhanced_msg[field] = metadata[field]
                            elif field == 'entities' or field == 'topics':
                                enhanced_msg[field] = []
                        
                        # Remove temporary field
                        if "_missing_fields" in enhanced_msg:
                            del enhanced_msg["_missing_fields"]
                            
                        result_messages[msg_idx] = enhanced_msg
                    except Exception as e:
                        self.logger.warning(f"Failed to enhance message {msg_idx}: {e}")
                        # Use original on failure
                        orig_msg = messages[msg_idx]
                        if "_missing_fields" in orig_msg:
                            del orig_msg["_missing_fields"]
                        result_messages[msg_idx] = orig_msg
        
        # Clean up any remaining temporary fields
        for msg in result_messages:
            if "_missing_fields" in msg:
                del msg["_missing_fields"]
                
        return result_messages
    
    def _create_metadata_prompt(self, message: Dict, missing_fields: List[str]) -> str:
        """Create a prompt for enhancing message metadata.
        
        Args:
            message: The message that needs metadata enhancement
            missing_fields: List of metadata fields to generate
            
        Returns:
            A prompt string for the LLM
        """
        fields_text = ", ".join(missing_fields)
        
        prompt = f"""Analyze the following message and generate {fields_text} metadata for it.

Message: "{message['content']}"
Role: {message['role']}

Generate ONLY the following fields: {fields_text}

Return a JSON object with just these fields. For sentiment, use "positive", "neutral", or "negative".
For intent, use categories like "greeting", "question", "clarification", "solution", or "farewell".
For entities and topics, return arrays of relevant terms.
For formality, use "formal", "casual", or "technical".

Response format example:
{{
  "sentiment": "neutral",
  "intent": "question",
  "entities": ["product name", "error code"],
  "topics": ["technical support", "login issue"],
  "formality": "formal"
}}

Only include the fields requested: {fields_text}
"""
        
        # Add formatting instructions
        formatted_prompt = f"""
{prompt}

IMPORTANT: Your response must be a valid JSON object containing ONLY the requested fields: {fields_text}.
No other text, explanations, or markdown formatting should be included.
"""
        
        return formatted_prompt
