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
        roles_config = conv_config.get("roles", ["user", "assistant"])
        
        # Ensure roles are strings to prevent unhashable type errors
        # Check if we should enforce two-person conversations (new config option)
        enforce_two_person = conv_config.get("enforce_two_person", True)  # Default to true for backwards compatibility
        max_roles = 2 if enforce_two_person else len(roles_config)
        
        roles = []
        for role in roles_config[:max_roles]:  # Limit to max_roles (usually 2 for two-person conversations)
            if isinstance(role, str):
                roles.append(role)
            else:
                # If it's not a string, convert it to a string
                self.logger.warning(f"Non-string role detected: {role}, converting to string")
                roles.append(str(role))
                
        # Log the roles being used
        self.logger.info(f"Using roles for conversation: {roles}")
                
        min_messages = conv_config.get("min_messages", 4)
        max_messages = conv_config.get("max_messages", 12)
        domains = conv_config.get("domains", ["general"])
        
        # Determine if we should use the hybrid approach (default: enabled)
        use_hybrid = conv_config.get("use_hybrid_approach", True)
        
        # Initialize context for template rendering
        context = {
            "domain": random.choice(domains),
            "min_messages": min_messages,
            "max_messages": max_messages,
            "roles": roles,
            **conv_config.get("variables", {}),
        }
        
        # Support dynamic field addition from config
        dynamic_fields = conv_config.get("dynamic_fields", {})
        if dynamic_fields:
            self.logger.info(f"Adding dynamic fields to context: {list(dynamic_fields.keys())}")
            for field_name, field_values in dynamic_fields.items():
                if isinstance(field_values, list) and field_values:
                    # If it's a list, select a random value
                    context[field_name] = random.choice(field_values)
                elif isinstance(field_values, dict):
                    # If it's a dictionary, add it directly
                    context[field_name] = field_values
                else:
                    # Otherwise add the value directly
                    context[field_name] = field_values
        
        # Add speaker information to context BEFORE rendering templates
        # This ensures name variables are available when templates are rendered
        predefined_speakers = conv_config.get("speakers", {})
        
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
                
                # Add advisor info to context for template rendering
                context["advisor_id"] = advisor_id
                context["advisor_name"] = advisor_name
            
            if selected_client and "client" in roles:
                client_id = selected_client.get("id")
                client_name = selected_client.get("name")
                
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
                
                # Add agent info to context for template rendering
                context["agent_id"] = agent_id
                context["agent_name"] = agent_name
            
            if selected_user and "user" in roles:
                user_id = selected_user.get("id")
                user_name = selected_user.get("name")
                
                # Add user info to context for template rendering
                context["user_id"] = user_id
                context["user_name"] = user_name
        
        # For any roles that don't have predefined speakers, create them dynamically
        for role in roles:
            role_key = f"{role}_name"
            if role_key not in context:
                if role == "advisor":
                    context["advisor_id"] = str(uuid.uuid4())
                    context["advisor_name"] = "John Advisor"
                elif role == "client":
                    context["client_id"] = str(uuid.uuid4())
                    context["client_name"] = "Jane Client"
                elif role == "support_agent":
                    context["agent_id"] = str(uuid.uuid4())
                    context["agent_name"] = "Support Agent"
                elif role == "user":
                    context["user_id"] = str(uuid.uuid4())
                    context["user_name"] = "User"
                elif role == "portfolio_manager":
                    context[f"{role}_id"] = str(uuid.uuid4())
                    context[f"{role}_name"] = random.choice(["Morgan Stanley", "William Chen", "Jennifer Adams", "Robert Kim"])
                elif role == "analyst":
                    context[f"{role}_id"] = str(uuid.uuid4())
                    context[f"{role}_name"] = random.choice(["Alex Morgan", "Sarah Lee", "James Wilson", "Emma Parker"])
                elif role == "researcher":
                    context[f"{role}_id"] = str(uuid.uuid4())
                    context[f"{role}_name"] = random.choice(["David Kim", "Lisa Zhang", "Mark Johnson", "Rachel Cohen"])
                elif role == "trader":
                    context[f"{role}_id"] = str(uuid.uuid4()) 
                    context[f"{role}_name"] = random.choice(["Michael Chang", "Jessica Wu", "Brian Smith", "Olivia Davis"])
                else:
                    context[f"{role}_id"] = str(uuid.uuid4())
                    context[f"{role}_name"] = role.capitalize()

        # Get system prompt if available
        system_prompt = None
        if "system_prompt" in self.templates:
            system_prompt = self.template_engine.render("system_prompt", context)

        # First, generate a conversation script/outline if the template exists
        conversation_script = None
        
        # Debug - list all available templates
        self.logger.info(f"Available templates: {list(self.templates.keys())}")
        
        if "conversation_script" in self.templates:
            self.logger.info("Generating conversation script as a guide...")
            
            # Let's fix errors in the template variables before rendering
            try:
                # Fix critical template variables that might be causing rendering issues
                context_copy = context.copy()
                
                # Ensure roles array is accessible and properly formatted
                if "roles" in context_copy:
                    # Log the roles and context for debugging
                    self.logger.info(f"Roles in context: {context_copy['roles']}")
                    self.logger.info(f"Context names: {[k for k in context_copy.keys() if '_name' in k]}")
                    
                    # Fix the variable access syntax in the template
                    context_copy["roles_0"] = context_copy["roles"][0] if len(context_copy["roles"]) > 0 else "analyst"
                    context_copy["roles_1"] = context_copy["roles"][1] if len(context_copy["roles"]) > 1 else "portfolio_manager"
                    
                    # Add explicit speaker names for the template
                    context_copy["speaker1_name"] = context_copy.get(f"{context_copy['roles_0']}_name", context_copy["roles_0"].capitalize())
                    context_copy["speaker2_name"] = context_copy.get(f"{context_copy['roles_1']}_name", context_copy["roles_1"].capitalize())
                    
                    self.logger.info(f"Prepared script variables: roles_0={context_copy['roles_0']}, roles_1={context_copy['roles_1']}")
                    self.logger.info(f"Speaker names: speaker1_name={context_copy['speaker1_name']}, speaker2_name={context_copy['speaker2_name']}")
                
                # Render the script template with the enhanced context
                script_template = self.templates.get("conversation_script", "")
                self.logger.info(f"Conversation script template starts with: {script_template[:100]}...")
                
                script_prompt = self.template_engine.render("conversation_script", context_copy)
                self.logger.info(f"Rendered script prompt successfully, length: {len(script_prompt)}")
            except Exception as e:
                self.logger.warning(f"Error preparing conversation script template: {e}")
                # Create a simpler script template directly
                # Get speaker names for the roles
                speaker1_name = context.get(f"{roles[0]}_name", roles[0].capitalize())
                speaker2_name = context.get(f"{roles[1]}_name", roles[1].capitalize())
                
                script_prompt = f"""Create a conversation script/outline between {roles[0]} ('{speaker1_name}') and {roles[1]} ('{speaker2_name}').
                
                IMPORTANT: ONLY use these two exact roles:
                - {roles[0]}
                - {roles[1]}
                Do NOT introduce any other roles like "trader" or "researcher" that aren't specified above.
                
                The outline should determine the flow of who speaks when, including when the same person might speak twice in a row.
                
                Domain: {context.get('domain', 'general')}
                Meeting Type: {context.get('meeting_type', 'discussion')}
                
                The conversation should have between {context.get('min_messages', 4)} and {context.get('max_messages', 12)} message turns.
                
                Output the script as a JSON array where each object has:
                - "speaker": MUST be EXACTLY "{roles[0]}" or "{roles[1]}" (no variations)
                - "speaker_name": "{speaker1_name}" or "{speaker2_name}" depending on speaker
                - "subject": brief description of what they'll talk about
                - "consecutive": boolean, true if this is a follow-up message from the same speaker
                
                Example of a valid script turn:
                {{"speaker": "{roles[0]}", "speaker_name": "{speaker1_name}", "subject": "Asks about recent market performance", "consecutive": false}}
                """
                self.logger.info("Using fallback script prompt due to template error")
            
            # Add formatting instructions specific to the script
            script_prompt += "\n\nYour response MUST be a valid JSON array. No explanations or additional text."
            
            # Generate the conversation script using the LLM
            script_result, _ = self.llm.generate(
                script_prompt,
                system_prompt=system_prompt,
                temperature=0.7,  # Use consistent temperature for script generation
                max_tokens=512,  # Script needs fewer tokens
            )
            
            try:
                # Show the raw script result for debugging
                self.logger.info(f"Raw script result (first 200 chars): {script_result[:200]}...")
                
                # Try to parse the script as a JSON array
                from src.faker.response_parser import parse_llm_response
                conversation_script = parse_llm_response(script_result)
                
                # Log details about the parsed result
                if isinstance(conversation_script, list):
                    self.logger.info(f"Successfully parsed conversation script as list of {len(conversation_script)} elements")
                    if len(conversation_script) > 0:
                        self.logger.info(f"First script turn: {conversation_script[0]}")
                elif isinstance(conversation_script, dict):
                    self.logger.warning(f"Script parsed as dictionary instead of list: {conversation_script.keys()}")
                else:
                    self.logger.warning(f"Script parsed as unexpected type: {type(conversation_script)}")
                
                if isinstance(conversation_script, list) and len(conversation_script) > 0:
                    self.logger.info(f"Generated conversation script with {len(conversation_script)} turns")
                    
                    # Validate the script turns have the required fields
                    valid_turns = []
                    for i, turn in enumerate(conversation_script):
                        if not isinstance(turn, dict):
                            self.logger.warning(f"Script turn {i} is not a dictionary: {turn}")
                            continue
                            
                        speaker = turn.get("speaker")
                        if not speaker or speaker not in roles:
                            self.logger.warning(f"Script turn {i} has invalid speaker: {speaker}")
                            # Try to assign a valid role
                            turn["speaker"] = roles[i % len(roles)]
                            
                        if "subject" not in turn:
                            turn["subject"] = f"Turn {i+1} in conversation"
                            
                        if "consecutive" not in turn:
                            # Set consecutive to true if same speaker as previous turn
                            if i > 0 and turn.get("speaker") == valid_turns[-1].get("speaker"):
                                turn["consecutive"] = True
                            else:
                                turn["consecutive"] = False
                                
                        valid_turns.append(turn)
                    
                    # Replace with validated turns
                    conversation_script = valid_turns
                    
                    # Add the script to the context for use in the full conversation template
                    context["conversation_script"] = conversation_script
                    
                    # Create a plain text version to include in the prompt
                    script_text = "\n\nFollow this conversation flow (including when the same person speaks consecutively):\n"
                    for i, turn in enumerate(conversation_script):
                        speaker = turn.get("speaker", "unknown")
                        speaker_name = turn.get("speaker_name", context.get(f"{speaker}_name", speaker))
                        subject = turn.get("subject", "")
                        consecutive = turn.get("consecutive", False)
                        consecutive_marker = " (consecutive)" if consecutive else ""
                        
                        script_text += f"{i+1}. {speaker_name} ({speaker}){consecutive_marker}: {subject}\n"
                    
                    # This will be appended to the main prompt
                    context["script_guidance"] = script_text
                    self.logger.info("Added script guidance to conversation prompt")
                    
                else:
                    self.logger.warning("Generated script was not a valid list, proceeding without script guidance")
            except Exception as e:
                import traceback
                self.logger.warning(f"Failed to parse conversation script: {e}")
                self.logger.warning(f"Traceback: {traceback.format_exc()}")
                self.logger.warning("Proceeding without script guidance")
        
        # Now render the main conversation template
        prompt = self.template_engine.render("conversation", context)
        
        # Add the script guidance if available
        if "script_guidance" in context:
            prompt += context["script_guidance"]
        
        # Log the rendered prompt to see if variables were replaced correctly
        self.logger.debug(f"Generated prompt with context: {context}")
        
        # Check if any template variables weren't replaced
        import re
        template_vars = re.findall(r'{{(.+?)}}', prompt)
        
        # Filter out variables from examples like {{name}} which are intentionally there
        if template_vars:
            # Remove single occurrences of 'name' as they're likely from examples
            filtered_vars = [var.strip() for var in template_vars if var.strip() != 'name']
            
            # Also filter out variables in examples section which are part of instructions
            filtered_vars = [var for var in filtered_vars if 
                             not (var == 'advisor_name' and 'EXAMPLES OF CORRECT NAME USAGE' in prompt) and
                             not (var == 'client_name' and 'EXAMPLES OF CORRECT NAME USAGE' in prompt) and
                             not (var == 'agent_name' and 'EXAMPLES OF CORRECT NAME USAGE' in prompt) and
                             not (var == 'user_name' and 'EXAMPLES OF CORRECT NAME USAGE' in prompt) and
                             not (var == 'company_name' and 'EXAMPLES OF CORRECT NAME USAGE' in prompt) and
                             not (var == 'advisor_firm' and 'EXAMPLES OF CORRECT NAME USAGE' in prompt)]
            
            if filtered_vars:
                self.logger.warning(f"Template variables not replaced in prompt: {filtered_vars}")
            
        # Add JSON formatting instructions with the context for few-shot examples
        formatted_prompt = self.template_engine.add_formatting_instructions(prompt, context=context)

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
                try:
                    # Add debug logging to inspect the message structure
                    self.logger.debug(f"Message structure before validation: {json.dumps(parsed_data[:2], default=str)}")
                    self.logger.debug(f"Roles before validation: {roles}")
                    
                    valid_messages, is_valid = validate_conversation_messages(
                        parsed_data, required_roles=roles, context_vars=context
                    )
                except TypeError as e:
                    # Handle unhashable type error if it occurs
                    self.logger.warning(f"Validation error: {e}, attempting with string roles")
                    # Add detailed logging
                    import traceback
                    self.logger.warning(f"Traceback: {traceback.format_exc()}")
                    
                    string_roles = [str(r) for r in roles]
                    valid_messages, is_valid = validate_conversation_messages(
                        parsed_data, required_roles=string_roles, context_vars=context
                    )
                
                if not valid_messages:
                    raise ValueError("No valid messages found after parsing")
                
                if not is_valid and use_hybrid:
                    # If validation failed and hybrid is enabled, enhance each message
                    valid_messages = self._enhance_messages_with_metadata(valid_messages)
            
            elif isinstance(parsed_data, dict) and "messages" in parsed_data:
                # Messages wrapped in object - validate and clean
                try:
                    valid_messages, is_valid = validate_conversation_messages(
                        parsed_data["messages"], required_roles=roles, context_vars=context
                    )
                except TypeError as e:
                    # Handle unhashable type error if it occurs
                    self.logger.warning(f"Validation error: {e}, attempting with string roles")
                    string_roles = [str(r) for r in roles]
                    valid_messages, is_valid = validate_conversation_messages(
                        parsed_data["messages"], required_roles=string_roles, context_vars=context
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
                
                # Apply post-processing to fix any template placeholders
                try:
                    valid_messages, _ = validate_conversation_messages(
                        valid_messages, required_roles=roles, context_vars=context
                    )
                except TypeError as e:
                    # Handle unhashable type error if it occurs
                    self.logger.warning(f"Validation error: {e}, attempting with string roles")
                    string_roles = [str(r) for r in roles]
                    valid_messages, _ = validate_conversation_messages(
                        valid_messages, required_roles=string_roles, context_vars=context
                    )
                
                if use_hybrid:
                    # Enhance these basic messages with metadata
                    valid_messages = self._enhance_messages_with_metadata(valid_messages)
            
        except (ValueError, json.JSONDecodeError) as e:
            # Fallback to text extraction if structured parsing fails
            self.logger.warning(f"Failed to parse result as structured data: {e}")
            valid_messages = extract_conversation_from_text(result, roles, max_messages)
            
            # Apply post-processing to fix any template placeholders
            try:
                valid_messages, _ = validate_conversation_messages(
                    valid_messages, required_roles=roles, context_vars=context
                )
            except TypeError as e:
                # Handle unhashable type error if it occurs
                self.logger.warning(f"Validation error: {e}, attempting with string roles")
                string_roles = [str(r) for r in roles]
                valid_messages, _ = validate_conversation_messages(
                    valid_messages, required_roles=string_roles, context_vars=context
                )
            
            if use_hybrid:
                # Enhance these messages with metadata
                valid_messages = self._enhance_messages_with_metadata(valid_messages)
        
        # Create speaker objects from the names we've already added to context
        speakers = {}
        role_to_speaker = {}
        predefined_speakers = conv_config.get("speakers", {})
        
        # Investment advisor-client conversation
        if "advisor_name" in context and "client_name" in context:
            # Get IDs (either existing or new)
            advisor_id = context.get("advisor_id", str(uuid.uuid4()))
            client_id = context.get("client_id", str(uuid.uuid4()))
            
            # Get metadata if available
            advisor_metadata = {}
            client_metadata = {}
            
            # Try to get metadata from predefined speakers
            if predefined_speakers and "advisors" in predefined_speakers:
                for advisor in predefined_speakers.get("advisors", []):
                    if advisor.get("name") == context["advisor_name"]:
                        advisor_metadata = advisor.get("metadata", {})
                        break
                        
            if predefined_speakers and "clients" in predefined_speakers:
                for client in predefined_speakers.get("clients", []):
                    if client.get("name") == context["client_name"]:
                        client_metadata = client.get("metadata", {})
                        break
            
            # Create speaker objects
            speakers[advisor_id] = Speaker(
                id=advisor_id,
                name=context["advisor_name"],
                role="advisor",
                metadata=advisor_metadata
            )
            
            speakers[client_id] = Speaker(
                id=client_id,
                name=context["client_name"],
                role="client",
                metadata=client_metadata
            )
            
            # Map roles to speaker IDs
            role_to_speaker["advisor"] = advisor_id
            role_to_speaker["client"] = client_id
        
        # Customer support conversation
        elif "agent_name" in context and "user_name" in context:
            # Get IDs (either existing or new)
            agent_id = context.get("agent_id", str(uuid.uuid4()))
            user_id = context.get("user_id", str(uuid.uuid4()))
            
            # Get metadata if available
            agent_metadata = {}
            user_metadata = {}
            
            # Try to get metadata from predefined speakers
            if predefined_speakers and "support_agents" in predefined_speakers:
                for agent in predefined_speakers.get("support_agents", []):
                    if agent.get("name") == context["agent_name"]:
                        agent_metadata = agent.get("metadata", {})
                        break
                        
            if predefined_speakers and "users" in predefined_speakers:
                for user in predefined_speakers.get("users", []):
                    if user.get("name") == context["user_name"]:
                        user_metadata = user.get("metadata", {})
                        break
            
            # Create speaker objects
            speakers[agent_id] = Speaker(
                id=agent_id,
                name=context["agent_name"],
                role="support_agent",
                metadata=agent_metadata
            )
            
            speakers[user_id] = Speaker(
                id=user_id,
                name=context["user_name"],
                role="user",
                metadata=user_metadata
            )
            
            # Map roles to speaker IDs
            role_to_speaker["support_agent"] = agent_id
            role_to_speaker["user"] = user_id
        
        # For any other roles, create speakers for them
        for role in roles:
            if role not in role_to_speaker:
                speaker_id = context.get(f"{role}_id", str(uuid.uuid4()))
                speaker_name = context.get(f"{role}_name", role.capitalize())
                
                speakers[speaker_id] = Speaker(
                    id=speaker_id, 
                    name=speaker_name, 
                    role=role
                )
                role_to_speaker[role] = speaker_id
        
        # Create Message objects
        messages = []
        
        # If we have a conversation script, use it to ensure role consistency
        # This helps with cases where the same person might speak twice in a row
        conversation_script = context.get("conversation_script")
        
        # Add an extra verification step to ensure script exists and has proper structure
        if conversation_script:
            if not isinstance(conversation_script, list):
                self.logger.warning(f"Script is not a list, type: {type(conversation_script)}")
                conversation_script = None
            elif len(conversation_script) == 0:
                self.logger.warning("Script is an empty list")
                conversation_script = None
            else:
                # Check first element to ensure it has expected structure
                first_turn = conversation_script[0]
                if not isinstance(first_turn, dict) or "speaker" not in first_turn:
                    self.logger.warning(f"First script turn has invalid structure: {first_turn}")
                    conversation_script = None
                else:
                    self.logger.info(f"Script validation passed, {len(conversation_script)} turns")
        
        if conversation_script and isinstance(conversation_script, list) and len(conversation_script) > 0:
            self.logger.info("Using conversation script to guide message creation")
            
            # Map each message to a script turn based on its position
            # Allowing for some flexibility if the counts don't exactly match
            script_length = len(conversation_script)
            message_length = len(valid_messages)
            
            # Prepare a mapping of message index to script turn
            script_mapping = []
            current_script_idx = 0
            
            # Build mapping to account for consecutive messages
            while len(script_mapping) < message_length and current_script_idx < script_length:
                script_turn = conversation_script[current_script_idx]
                script_mapping.append(current_script_idx)
                
                # If this turn allows consecutive messages and we haven't mapped all messages yet
                if script_turn.get("consecutive", False) and len(script_mapping) < message_length:
                    # Allow for an extra message with the same speaker
                    script_mapping.append(current_script_idx)
                
                current_script_idx += 1
            
            # Fill any remaining messages by cycling through script turns
            while len(script_mapping) < message_length:
                script_mapping.append((len(script_mapping) % script_length))
            
            # Now create messages using the script mapping
            for i, msg_data in enumerate(valid_messages):
                content = msg_data.get("content", "")
                
                # Get the script turn for this message position
                script_turn_idx = script_mapping[i] if i < len(script_mapping) else i % script_length
                script_turn = conversation_script[script_turn_idx]
                
                # Use the role from the script
                role = script_turn.get("speaker")
                if role not in role_to_speaker:
                    self.logger.warning(f"Script specified unknown role: {role}, defaulting to allowed roles")
                    # Check if the role in the message data matches our allowed roles
                    msg_role = msg_data.get("role")
                    if msg_role in role_to_speaker:
                        role = msg_role
                    else:
                        role = roles[i % len(roles)]  # Fallback to alternating roles
                
                # Log when script is being enforced to override the message role
                orig_role = msg_data.get("role")
                if orig_role != role and orig_role not in role_to_speaker:
                    self.logger.info(f"Script enforcing role {role} instead of invalid {orig_role} at position {i}")
                
                # Get speaker name from the script or context
                speaker_name = script_turn.get("speaker_name") or context.get(f"{role}_name", role.capitalize())
                
                # Process content to replace any template placeholders
                if "{{" in content or "}}" in content:
                    content = content.replace("{{advisor_name}}", context.get("advisor_name", ""))
                    content = content.replace("{{client_name}}", context.get("client_name", ""))
                    content = content.replace("{{agent_name}}", context.get("agent_name", ""))
                    content = content.replace("{{user_name}}", context.get("user_name", ""))
                    
                    # Handle roles specifically to ensure correct names are used
                    for r in roles:
                        if f"{{{{{r}_name}}}}" in content:
                            content = content.replace(f"{{{{{r}_name}}}}", context.get(f"{r}_name", r.capitalize()))
                
                # Check for generic role names that should be replaced with specific names
                if role == "portfolio_manager" and "Portfolio Manager" in content:
                    content = content.replace("Portfolio Manager", speaker_name)
                
                # Process entities for this message
                entities = msg_data.get("entities", [])
                if not isinstance(entities, list):
                    self.logger.warning(f"Entities is not a list, converting: {entities}")
                    try:
                        entities = list(entities) if hasattr(entities, '__iter__') else []
                    except (TypeError, ValueError):
                        entities = []
                
                # Process each entity to ensure it's a simple dict with string keys
                processed_entities = []
                for entity in entities:
                    if isinstance(entity, dict):
                        # Clean the dict to ensure all keys and values are strings or simple types
                        clean_entity = {}
                        for k, v in entity.items():
                            # Convert any complex keys/values to strings
                            try:
                                clean_key = str(k)
                                clean_value = v
                                if isinstance(v, dict):
                                    clean_value = str(v)  # Convert nested dicts to string
                                clean_entity[clean_key] = clean_value
                            except Exception as e:
                                self.logger.warning(f"Error cleaning entity: {e}")
                        processed_entities.append(clean_entity)
                    elif entity is not None:
                        # If it's not a dict but not None, convert to string
                        processed_entities.append({"entity": str(entity), "entity_type": "unknown"})
                
                # Create and add the message
                message = Message(
                    content=content,
                    speaker_id=role_to_speaker[role],
                    # Add additional metadata if available
                    sentiment=msg_data.get("sentiment"),
                    intent=msg_data.get("intent"),
                    entities=processed_entities,
                    topics=msg_data.get("topics", []),
                    language=msg_data.get("language"),
                    formality=msg_data.get("formality"),
                    metadata=msg_data.get("metadata", {}),
                )
                messages.append(message)
        
        # If no conversation script or script application failed, use the original method
        else:
            self.logger.info("No conversation script available, using standard role assignment")
            
            for msg_data in valid_messages:
                role = msg_data.get("role")
                content = msg_data.get("content", "")
                
                # Default to first role if role is invalid
                if role not in role_to_speaker:
                    role = roles[0]
                
                # Get speaker name for this role from context
                speaker_name = context.get(f"{role}_name", role.capitalize())
                
                # Ensure content uses the correct name by replacing template-like patterns
                # This ensures names in message content match the speaker definitions
                if "{{" in content or "}}" in content:
                    content = content.replace("{{advisor_name}}", context.get("advisor_name", ""))
                    content = content.replace("{{client_name}}", context.get("client_name", ""))
                    content = content.replace("{{agent_name}}", context.get("agent_name", ""))
                    content = content.replace("{{user_name}}", context.get("user_name", ""))
                    
                    # Handle roles specifically to ensure correct names are used
                    for r in roles:
                        if f"{{{{{r}_name}}}}" in content:
                            content = content.replace(f"{{{{{r}_name}}}}", context.get(f"{r}_name", r.capitalize()))
                
                # Check for generic role names that should be replaced with specific names
                if role == "portfolio_manager" and "Portfolio Manager" in content:
                    content = content.replace("Portfolio Manager", speaker_name)
                
                # Process entities to ensure they're serializable
                entities = msg_data.get("entities", [])
                if not isinstance(entities, list):
                    self.logger.warning(f"Entities is not a list, converting: {entities}")
                    try:
                        entities = list(entities) if hasattr(entities, '__iter__') else []
                    except (TypeError, ValueError):
                        entities = []
                
                # Process each entity to ensure it's a simple dict with string keys
                processed_entities = []
                for entity in entities:
                    if isinstance(entity, dict):
                        # Clean the dict to ensure all keys and values are strings or simple types
                        clean_entity = {}
                        for k, v in entity.items():
                            # Convert any complex keys/values to strings
                            try:
                                clean_key = str(k)
                                clean_value = v
                                if isinstance(v, dict):
                                    clean_value = str(v)  # Convert nested dicts to string
                                clean_entity[clean_key] = clean_value
                            except Exception as e:
                                self.logger.warning(f"Error cleaning entity: {e}")
                        processed_entities.append(clean_entity)
                    elif entity is not None:
                        # If it's not a dict but not None, convert to string
                        processed_entities.append({"entity": str(entity), "entity_type": "unknown"})
                
                # Create and add the message
                message = Message(
                    content=content,
                    speaker_id=role_to_speaker[role],
                    # Add additional metadata if available
                    sentiment=msg_data.get("sentiment"),
                    intent=msg_data.get("intent"),
                    entities=processed_entities,
                    topics=msg_data.get("topics", []),
                    language=msg_data.get("language"),
                    formality=msg_data.get("formality"),
                    metadata=msg_data.get("metadata", {}),
                )
                messages.append(message)
        
        # Create filtered speakers dictionary with only active speakers
        active_speaker_ids = set(msg.speaker_id for msg in messages)
        active_speakers = {id: speaker for id, speaker in speakers.items() if id in active_speaker_ids}
        
        # Create the conversation object with only the active speakers
        conversation = Conversation(
            messages=messages,
            speakers=active_speakers,
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
