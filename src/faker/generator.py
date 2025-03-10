"""Core module for generating synthetic conversations."""

import os
import json
import yaml
import time
import logging
import datetime
import random
import uuid
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Set

from faker.models import (
    Conversation, Message, Dataset, RunInfo, Speaker
)
from faker.llm import GeminiProvider, MockProvider, LLMProvider
from faker.templates import TemplateEngine


class ChatGenerator:
    """Main class for generating synthetic chat data.
    
    This class handles configuration, LLM interactions, and dataset generation.
    """
    
    def __init__(
        self, 
        llm_provider: Optional[LLMProvider] = None,
        templates: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None
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
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Replace environment variables in config
        config = cls._replace_env_vars(config)
            
        templates = config.get('templates', {})
        llm_config = config.get('llm', {})
        output_dir = config.get('dataset', {}).get('output_dir', './output')
        
        # Initialize the appropriate LLM provider
        provider_name = llm_config.get('provider', 'gemini')
        if provider_name == 'gemini':
            llm_provider = GeminiProvider(
                project_id=llm_config.get('project_id'),
                location=llm_config.get('location'),
                model=llm_config.get('model', 'gemini-1.5-pro'),
                credentials_path=llm_config.get('credentials_path'),
                default_temperature=llm_config.get('temperature', 0.7),
                default_top_p=llm_config.get('top_p', 0.95),
                default_top_k=llm_config.get('top_k', 40),
                default_max_tokens=llm_config.get('max_tokens', 1024)
            )
        elif provider_name == 'mock':
            # For testing without API calls
            llm_provider = MockProvider(
                responses=llm_config.get('responses', {}),
                default_response=llm_config.get('default_response', '{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}')
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
            
        return cls(
            llm_provider=llm_provider,
            templates=templates,
            config=config,
            output_dir=output_dir
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
            
        config_str = re.sub(r'\${([^}]+)}|\$([A-Za-z0-9_]+)', replace_env_var, config_str)
        
        return json.loads(config_str)
        
    def generate(
        self, 
        num_conversations: int = 1,
        output_format: str = 'split',
        export_run_info: bool = True,
        logging_service = None
    ) -> Dataset:
        """Generate a dataset of synthetic conversations.
        
        Args:
            num_conversations: Number of conversations to generate
            output_format: Format to export dataset ('jsonl', 'json', 'split')
            export_run_info: Whether to export run information
            logging_service: Optional LoggingService for tracking metrics
            
        Returns:
            A Dataset object containing the generated conversations
        """
        # Create dataset with run information
        dataset = Dataset(
            conversations=[],
            name=self.config.get('name', 'synthetic_dataset'),
            version=self.config.get('version', '1.0.0'),
            description=self.config.get('description', 'Synthetic chat data'),
            tags=self.config.get('tags', [])
        )
        
        # Add run information
        dataset.add_run_info(
            config_path=self.config.get('config_path'),
            config=self.config
        )
        
        # Add model information
        if dataset.run_info:
            dataset.run_info.model_info = self.llm.get_model_info()
        
        # Initialize run in logging service if provided
        run_id = None
        if logging_service:
            run_id = logging_service.init_run(
                config=self.config,
                name=f"Generate {num_conversations} conversations - {dataset.name}"
            )
            self.logger.info(f"Initialized run with ID: {run_id}")
        
        # Start timing
        start_time = time.time()
        
        try:
            # Generate conversations
            conversations = []
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
                            {"error": str(e), "index": i}
                        )
                    
            # Update dataset
            dataset.conversations = conversations
            
            # Complete run information
            if dataset.run_info:
                dataset.run_info.complete()
                dataset.run_info.add_stat("num_conversations", len(conversations))
                dataset.run_info.add_stat("total_time", time.time() - start_time)
                dataset.run_info.add_stat("avg_time_per_conversation", 
                                        (time.time() - start_time) / num_conversations if num_conversations > 0 else 0)
            
            # Export dataset
            if self.output_dir:
                # Create timestamped directory
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                dataset_name = f"{dataset.name}_{timestamp}"
                output_path = os.path.join(self.output_dir, dataset_name)
                
                # Export dataset
                exported_path = dataset.export(
                    os.path.join(output_path, f"{dataset_name}.{output_format}"),
                    format=output_format
                )
                self.logger.info(f"Dataset exported to: {exported_path}")
                
                # Export run information
                if export_run_info and dataset.run_info:
                    run_info_path = dataset.export_run_info(output_path)
                    self.logger.info(f"Run information exported to: {run_info_path}")
                
                # Save output path to logging service
                if logging_service and run_id:
                    logging_service.save_custom_metric(run_id, "output_path", exported_path)
            
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
        """Generate a single conversation.
        
        Returns:
            A Conversation object containing messages
        """
        # Get conversation parameters from config
        conv_config = self.config.get('conversation', {})
        roles = conv_config.get('roles', ['user', 'assistant'])
        min_messages = conv_config.get('min_messages', 4)
        max_messages = conv_config.get('max_messages', 12)
        domains = conv_config.get('domains', ['general'])
        
        # Prepare context for template rendering
        context = {
            'domain': random.choice(domains),
            'min_messages': min_messages,
            'max_messages': max_messages,
            'roles': roles,
            **conv_config.get('variables', {})
        }
        
        # Get system prompt if available
        system_prompt = None
        if 'system_prompt' in self.templates:
            system_prompt = self.template_engine.render('system_prompt', context)
        
        # Render the conversation template
        prompt = self.template_engine.render('conversation', context)
        
        # Generate the conversation using the LLM
        start_time = time.time()
        result, metadata = self.llm.generate(
            prompt, 
            system_prompt=system_prompt,
            temperature=conv_config.get('temperature', 0.7),
            top_p=conv_config.get('top_p', 0.95),
            top_k=conv_config.get('top_k', 40),
            max_tokens=conv_config.get('max_tokens', 1024)
        )
        generation_time = time.time() - start_time
        
        # Parse the result into a structured conversation
        try:
            # Try to parse as JSON
            parsed_data = json.loads(result)
            
            # If the result is an array of messages
            if isinstance(parsed_data, list):
                # Create speakers
                speakers = {}
                for role in roles:
                    speaker_id = str(uuid.uuid4())
                    speakers[speaker_id] = Speaker(
                        id=speaker_id,
                        name=role.capitalize(),
                        role=role
                    )
                
                # Map role names to speaker IDs
                role_to_speaker = {role: next(id for id, speaker in speakers.items() 
                                             if speaker.role == role) 
                                   for role in roles}
                
                # Create messages
                messages = []
                for msg_data in parsed_data:
                    role = msg_data.get('role')
                    content = msg_data.get('content')
                    
                    if role and content and role in role_to_speaker:
                        message = Message(
                            content=content,
                            speaker_id=role_to_speaker[role],
                            # Add additional metadata if available
                            sentiment=msg_data.get('sentiment'),
                            intent=msg_data.get('intent'),
                            entities=msg_data.get('entities', []),
                            topics=msg_data.get('topics', []),
                            language=msg_data.get('language'),
                            formality=msg_data.get('formality'),
                            metadata=msg_data.get('metadata', {})
                        )
                        messages.append(message)
            
            # If result has a 'messages' field    
            elif isinstance(parsed_data, dict) and 'messages' in parsed_data:
                # Create speakers
                speakers = {}
                for role in roles:
                    speaker_id = str(uuid.uuid4())
                    speakers[speaker_id] = Speaker(
                        id=speaker_id,
                        name=role.capitalize(),
                        role=role
                    )
                
                # Map role names to speaker IDs
                role_to_speaker = {role: next(id for id, speaker in speakers.items() 
                                             if speaker.role == role) 
                                   for role in roles}
                
                # Create messages
                messages = []
                for msg_data in parsed_data['messages']:
                    role = msg_data.get('role')
                    content = msg_data.get('content')
                    
                    if role and content and role in role_to_speaker:
                        message = Message(
                            content=content,
                            speaker_id=role_to_speaker[role],
                            # Add additional metadata if available
                            sentiment=msg_data.get('sentiment'),
                            intent=msg_data.get('intent'),
                            entities=msg_data.get('entities', []),
                            topics=msg_data.get('topics', []),
                            language=msg_data.get('language'),
                            formality=msg_data.get('formality'),
                            metadata=msg_data.get('metadata', {})
                        )
                        messages.append(message)
            
            else:
                # Fallback if the result is not in the expected format
                raise ValueError(f"Unexpected result format: {result[:100]}...")
                
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback to simple parsing if JSON parsing fails
            self.logger.warning(f"Failed to parse result as JSON: {e}")
            self.logger.warning(f"Result: {result[:500]}...")
            self.logger.warning("Using simple parsing fallback")
            
            # Try to extract a conversation using regex
            speakers = {}
            for role in roles:
                speaker_id = str(uuid.uuid4())
                speakers[speaker_id] = Speaker(
                    id=speaker_id,
                    name=role.capitalize(),
                    role=role
                )
            
            # Create a basic conversation with alternating speakers
            messages = []
            speaker_cycle = [id for role in roles for id, speaker in speakers.items() 
                             if speaker.role == role]
            
            # If no speakers found, create a default one
            if not speaker_cycle:
                speaker_id = str(uuid.uuid4())
                speakers[speaker_id] = Speaker(
                    id=speaker_id,
                    name="Unknown",
                    role="unknown"
                )
                speaker_cycle = [speaker_id]
            
            # Split by common patterns that might indicate message boundaries
            lines = re.split(r'\n+|(?:[A-Za-z]+:)', result)
            lines = [line.strip() for line in lines if line.strip()]
            
            for i, line in enumerate(lines[:max_messages]):
                speaker_id = speaker_cycle[i % len(speaker_cycle)]
                messages.append(Message(
                    content=line,
                    speaker_id=speaker_id
                ))
        
        # Create the conversation with the generated messages
        conversation = Conversation(
            messages=messages,
            speakers=speakers,
            domain=context.get('domain'),
            generation_config=metadata.get('generation_config', {}),
            prompt_template=prompt,
            prompt_variables=context,
            model_info=metadata,
            generation_time=generation_time
        )
        
        # Extract topics and entities if available
        conversation.topics = list(conversation.extract_topics())
        conversation.entities = list(conversation.extract_entities())
        
        return conversation