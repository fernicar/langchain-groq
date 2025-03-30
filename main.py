# --- START OF FILE main.py ---
import sys
import os
from dotenv import load_dotenv, set_key
from langchain_groq import ChatGroq
import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from typing import List
import json
from datetime import datetime, timezone
from pathlib import Path
import typing as t
from groq import Groq
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseLanguageModel
from tiktoken import encoding_for_model

from gui import GUI, APIKeyDialog  # Import the GUI class


class APIMonitorCallback(BaseCallbackHandler):
  def __init__(self, api_monitor):
    super().__init__()
    self.api_monitor = api_monitor

  def _serialize_for_json(self, obj):
    if hasattr(obj, "model_dump"):
      return obj.model_dump()
    if hasattr(obj, "__dict__"):
      return obj.__dict__
    if hasattr(obj, "hex"):
      return str(obj)
    return str(obj)

  def on_llm_start(self, serialized, prompts, **kwargs):
    try:
      self.api_monitor.append("RAW GROQ API REQUEST:")
      # Use ensure_ascii=False to show actual characters instead of escape sequences
      self.api_monitor.append(f"Raw message_dicts:\n{json.dumps(prompts, indent=2, ensure_ascii=False)}\n")
      self.api_monitor.append(f"Raw params:\n{json.dumps(serialized, indent=2, ensure_ascii=False)}\n")
      self.api_monitor.append("-" * 50 + "\n")
    except Exception as e:
      self.api_monitor.append(f"Error logging request: {str(e)}\n")

  def on_llm_end(self, response, **kwargs):
    try:
      self.api_monitor.append("RAW GROQ API RESPONSE:")
      response_dict = response.model_dump() if hasattr(response, "model_dump") else str(response)
      # Use ensure_ascii=False here too
      self.api_monitor.append(f"{json.dumps(response_dict, indent=2, ensure_ascii=False)}\n")
      self.api_monitor.append("-" * 50 + "\n")
    except Exception as e:
      self.api_monitor.append(f"Error logging response: {str(e)}\n")

  def on_llm_error(self, error, **kwargs):
    try:
      self.api_monitor.append("RAW API ERROR:")
      error_dict = {"error_type": error.__class__.__name__, "error_message": str(error)}
      error_json = json.dumps(error_dict, indent=2, ensure_ascii=False, default=self._serialize_for_json)
      kwargs_json = json.dumps(kwargs, indent=2, ensure_ascii=False, default=self._serialize_for_json)

      self.api_monitor.append(f"error:\n{error_json}\n")
      self.api_monitor.append(f"kwargs:\n{kwargs_json}\n")
      print(f"Error occurred:\n{error_json}")
      self.api_monitor.append("-" * 50 + "\n")
    except Exception as e:
      self.api_monitor.append(f"Error serializing error: {str(e)}\n")
      self.api_monitor.append(f"Raw error: {str(error)}\n")
      self.api_monitor.append(f"Raw kwargs: {str(kwargs)}\n")
      self.api_monitor.append("-" * 50 + "\n")

  def on_llm_new_token(self, token: str, **kwargs):
    """Called when streaming is enabled and a new token is received"""
    self.api_monitor.append(f"Token: {token}\n")


class SystemPromptManager:
  DEFAULT_PROMPT = """Eres un colaborador narrativo.
  Tu papel es ayudar a crear historias.
  Mantén las respuestas de narrativa solamente, sin explicaciones, listas para publicar."""

  def __init__(self, config_path: str = "system_prompts.json"):
    """Initialize the SystemPromptManager with a config file path"""
    self.config_path = Path(config_path)
    self.prompts = self._load_or_create_config()

    # Get the initial active prompt content and name
    active_name = self.prompts["active_prompt"]
    self.initial_prompt_content = self.prompts["prompts"][active_name]["content"]
    self.initial_prompt_name = active_name

  def _load_or_create_config(self) -> dict:
    """Load existing config or create default one"""
    if self.config_path.exists():
      try:
        with open(self.config_path, "r", encoding="utf-8") as f:
          return json.load(f)
      except json.JSONDecodeError:
        # If JSON is invalid, create new default config
        return self._create_default_config()
    return self._create_default_config()

  def _create_default_config(self) -> dict:
    """Create default configuration"""
    default_config = {"active_prompt": "default", "prompts": {"default": {"content": self.DEFAULT_PROMPT, "created_at": datetime.now(timezone.utc).isoformat(), "last_used": datetime.now(timezone.utc).isoformat()}}}
    self._save_config(default_config)
    return default_config

  def _save_config(self, config: dict) -> None:
    """Save config to file"""
    with open(self.config_path, "w", encoding="utf-8") as f:
      json.dump(config, f, indent=4, ensure_ascii=False)

  def get_active_prompt(self) -> str:
    """Get the content of the active prompt"""
    active_name = self.prompts["active_prompt"]
    prompt_data = self.prompts["prompts"][active_name]
    prompt_data["last_used"] = datetime.now(timezone.utc).isoformat()
    self._save_config(self.prompts)
    return prompt_data["content"]

  def set_active_prompt(self, name: str) -> None:
    """Set the active prompt by name"""
    if name in self.prompts["prompts"]:
      self.prompts["active_prompt"] = name
      self._save_config(self.prompts)

  def save_prompt(self, name: str, content: str) -> None:
    """Save a new prompt or update existing one"""
    self.prompts["prompts"][name] = {"content": content, "created_at": self.prompts["prompts"].get(name, {}).get("created_at", datetime.now(timezone.utc).isoformat()), "last_used": datetime.now(timezone.utc).isoformat()}
    self._save_config(self.prompts)

  def get_all_prompts(self) -> t.Dict[str, dict]:
    """Get all available prompts"""
    return self.prompts["prompts"]

  def delete_prompt(self, name: str) -> bool:
    """Delete a prompt by name"""
    if name == "default":
      return False
    if name in self.prompts["prompts"]:
      del self.prompts["prompts"][name]
      if self.prompts["active_prompt"] == name:
        self.prompts["active_prompt"] = "default"
      self._save_config(self.prompts)
      return True
    return False


class TokenWindowDualStateMemory(BaseChatMessageHistory):
  """Memory implementation that maintains a token window and dual state management"""
    
  def __init__(self, llm: BaseLanguageModel, max_tokens: int = 12000):
    self.max_tokens = max_tokens
    self.llm = llm
    # Use GPT-3.5 tokenizer as a reasonable default
    self.encoding = encoding_for_model("gpt-3.5-turbo")
    
    self._messages_committed = []  # Committed/backup state
    self._messages_proposal = []   # Current proposal state
    self._has_pending_proposal = False

  @property
  def messages(self) -> List[BaseMessage]:
    """Return current active messages (proposal if exists, otherwise committed)"""
    return self._messages_proposal if self._has_pending_proposal else self._messages_committed
  
  @messages.setter
  def messages(self, messages: List[BaseMessage]):
    """Set messages and mark as proposal"""
    self._messages_proposal = messages
    self._has_pending_proposal = True
    self._truncate_messages(self._messages_proposal)
  
  def add_message(self, message: BaseMessage) -> None:
    """Add message to proposal state"""
    if not self._has_pending_proposal:
      # If no proposal exists, create one from current committed state
      self._messages_proposal = self._messages_committed.copy()
      self._has_pending_proposal = True
    self._messages_proposal.append(message)
    self._truncate_messages(self._messages_proposal)
  
  def prepare_for_response(self) -> None:
    """Backup current state before LLM response"""
    self._messages_committed = self.messages.copy()
    self._has_pending_proposal = False
  
  def commit_proposal(self) -> None:
    """Commit current proposal to backup state"""
    if self._has_pending_proposal:
      self._messages_committed = self._messages_proposal.copy()
      self._has_pending_proposal = False
  
  def discard_proposal(self) -> None:
    """Discard current proposal and restore from backup"""
    self._messages_proposal = self._messages_committed.copy()
    self._has_pending_proposal = False
  
  def clear(self) -> None:
    """Clear all messages"""
    self._messages_committed = []
    self._messages_proposal = []
    self._has_pending_proposal = False
  
  def _count_tokens(self, text: str) -> int:
    """Count tokens in a text string"""
    return len(self.encoding.encode(text))
  
  def _truncate_messages(self, messages: List[BaseMessage]) -> None:
    """Remove oldest messages until total tokens is under max_tokens"""
    current_tokens = 0
    truncated_messages = []
      
    # Process messages in reverse (newest first)
    for message in reversed(messages):
      tokens = self._count_tokens(message.content)
      
      # If adding this message would exceed max tokens, stop
      if current_tokens + tokens > self.max_tokens:
          break
          
      # Add message to start of list (maintaining original order)
      truncated_messages.insert(0, message)
      current_tokens += tokens
    
    # Update the appropriate message list
    if messages is self._messages_proposal:
      self._messages_proposal = truncated_messages
    else:
      self._messages_committed = truncated_messages


class Narrative(GUI):  # Inherit from GUI
  def __init__(self):
    super().__init__()

    # Check for API key before initializing
    if not self.ensure_api_key():
      sys.exit(1)

    # Initialize system prompt from prompt manager
    self.prompt_manager = SystemPromptManager()
    self.system_prompt = self.prompt_manager.get_active_prompt()

    # Populate models and prompts in GUI after initialization
    self.populate_models_and_prompts()

    # Now initialize LLM after toolbar is setup
    self.initialize_llm()

  def ensure_api_key(self):
    """Ensure GROQ_API_KEY is available, prompt user if not"""
    # Try to load from .env first
    env_path = Path(".env")
    load_dotenv(env_path)

    api_key = os.getenv("GROQ_API_KEY")
    if api_key:  # Accept any non-empty key from .env
      return True

    # If no key found, show dialog
    dialog = APIKeyDialog(self)
    if dialog.exec():
      api_key = dialog.get_api_key()
      if api_key:
        # Save to .env file
        env_path.touch(exist_ok=True)
        set_key(env_path, "GROQ_API_KEY", api_key)
        os.environ["GROQ_API_KEY"] = api_key
        return True

    return False

  def populate_models_and_prompts(self):
    """Populate model selector and prompt selector with data"""
    available_models = self.get_available_models()
    self.model_selector.clear()
    self.model_selector.addItems(available_models)

    prompts = self.prompt_manager.get_all_prompts()
    self.prompt_selector.clear()
    self.prompt_selector.addItems(prompts.keys())
    self.prompt_selector.setCurrentText(self.prompt_manager.prompts["active_prompt"])

    # Set initial system prompt text
    self.system_input.setPlainText(self.prompt_manager.get_active_prompt())
    self.prompt_name_input.setText(self.prompt_manager.prompts["active_prompt"])

    # Connect signals now that methods are defined in Narrative class
    self.model_selector.currentTextChanged.connect(self.on_model_changed)
    self.temperature_spinner.valueChanged.connect(self.on_temperature_changed)
    self.max_tokens_spinner.valueChanged.connect(self.on_max_tokens_changed)
    self.prompt_selector.currentTextChanged.connect(self.on_prompt_selected)

    # Connect menu actions and buttons
    self.send_button.clicked.connect(self.send_message)
    self.load_action.triggered.connect(self.load_story)
    self.save_action.triggered.connect(self.save_story)
    self.save_as_action.triggered.connect(lambda: self.save_story(save_as=True))
    self.clear_action.triggered.connect(lambda: self.load_story(empty=True))
    self.exit_action.triggered.connect(self.close)
    self.save_prompt_button.clicked.connect(self.save_system_prompt)
    self.delete_prompt_button.clicked.connect(self.delete_system_prompt)

  def initialize_llm(self):
    """Initialize the language model and conversation chain"""
    load_dotenv()

    self.system_prompt = self.prompt_manager.get_active_prompt()
    selected_model = self.model_selector.currentText()
    callback = APIMonitorCallback(self.api_monitor)

    llm = ChatGroq(
      api_key=os.environ["GROQ_API_KEY"],
      model_name=selected_model,
      temperature=self.temperature,
      max_tokens=self.max_tokens,
      streaming=False,
      verbose=True,
      callbacks=[callback]
    )

    prompt = ChatPromptTemplate.from_messages([
      SystemMessage(content=self.system_prompt),
      MessagesPlaceholder(variable_name="history"),
      HumanMessagePromptTemplate.from_template("{input}")
    ])

    chain = prompt | llm

    if self.primed_history is not None:
      history = self.primed_history
      self.primed_history = None
    else:
      history = TokenWindowDualStateMemory(llm, max_tokens=12000)

    self.conversation = RunnableWithMessageHistory(
      chain,
      lambda session_id: history,
      input_messages_key="input",
      history_messages_key="history"
    )

    self.session_id = "default_session"

  def get_available_models(self) -> List[str]:
    """Fetch available models from Groq API"""
    try:
      client = Groq(api_key=os.environ["GROQ_API_KEY"])
      models = client.models.list()
      # Filter and sort models based on our needs
      available_models = []
      for model in models.data:  # Access the data attribute of the response
        model_id = model.id if hasattr(model, "id") else str(model)
        if "whisper" not in model_id.lower():
          available_models.append(model_id)
      return sorted(available_models, reverse=True)
    except Exception as e:
      if hasattr(self, "api_monitor"):
        self.api_monitor.append(f"Error fetching models: {str(e)}\n")
      # Fallback to hardcoded models if API fails
      return ["qwen-qwq-32b", "deepseek-r1-distill-qwen-32b", "deepseek-r1-distill-llama-70b", "mixtral-8x7b-32768", "llama-3.3-70b-versatile"]

  def on_model_changed(self):
    """Reinitialize the LLM when model selection changes"""
    self.initialize_llm()

  def update_prompt_selector(self):
    """Update the prompt selector combo box"""
    current = self.prompt_selector.currentText()
    self.prompt_selector.clear()
    prompts = self.prompt_manager.get_all_prompts()
    self.prompt_selector.addItems(prompts.keys())
    if current in prompts:
      self.prompt_selector.setCurrentText(current)

  def on_prompt_selected(self, prompt_name: str):
    """Handle prompt selection"""
    if not prompt_name:
      return
    prompts = self.prompt_manager.get_all_prompts()
    if prompt_name in prompts:
      # Set the content in the system prompt tab
      self.system_input.setPlainText(prompts[prompt_name]["content"])
      self.prompt_name_input.setText(prompt_name)
      self.prompt_manager.set_active_prompt(prompt_name)
      self.initialize_llm()

  def save_system_prompt(self):
    """Save current prompt"""
    name = self.prompt_name_input.text().strip()
    content = self.system_input.toPlainText().strip()
    if name and content:
      self.prompt_manager.save_prompt(name, content)
      self.prompt_manager.set_active_prompt(name)
      self.update_prompt_selector()
      self.prompt_selector.setCurrentText(name)
      self.initialize_llm()
      print(f"System prompt '{name}' saved and activated.")

  def delete_system_prompt(self):
    """Delete current prompt"""
    name = self.prompt_selector.currentText()
    if self.prompt_manager.delete_prompt(name):
      self.update_prompt_selector()
      print(f"System prompt '{name}' deleted.")
    else:
      print("Cannot delete default prompt.")

  def simulate_conversation_turn(self, content: str):
    """Simulate a conversation turn with the given content"""
    if hasattr(self, "conversation"):
      # Create config with session_id
      config = {"configurable": {"session_id": self.session_id}}
      # Get history through the config mechanism
      history = self.conversation._merge_configs(config)["configurable"]["message_history"]

      # Strip out think XML tags before adding to history
      cleaned_content = re.sub(r"\n*<think>.*?</think>\n*", "", content, flags=re.DOTALL).strip()

      # Get user input from continue tab, fallback to emoji if empty
      user_input = self.current_user_input
      self.current_user_input = self.continue_input.toPlainText().strip()
      if not user_input:
        user_input = "✒️✍️📜"

      # Add simulated conversation pair
      history.add_message(HumanMessage(content=user_input))
      history.add_message(AIMessage(content=cleaned_content))
      self.update_context_display()

  def commit_blue_text(self):
    """Commit current blue text to canon"""
    config = {"configurable": {"session_id": self.session_id}}
    history = self.conversation._merge_configs(config)["configurable"]["message_history"]
    
    # Commit the current proposal
    history.commit_proposal()
    
    # Your existing commit code...
    self.canon_validated.append(self.current_narrative)
    self.update_blue_text("")
    self.update_context_display()

  def discard_last_conversation_pair(self):
    """Remove the last user input and AI response pair from history"""
    config = {"configurable": {"session_id": self.session_id}}
    history = self.conversation._merge_configs(config)["configurable"]["message_history"]
    
    # Discard the current proposal
    history.discard_proposal()
    
    # Update UI to reflect the committed state
    if history.messages:
      last_message = history.messages[-1]
      if isinstance(last_message, AIMessage):
          self.update_blue_text(last_message.content)
    else:
      self.update_blue_text("")
    
    self.update_context_display()

  def update_context_display(self):
    """Update the context monitor display"""
    if hasattr(self, "conversation"):
      config = {"configurable": {"session_id": self.session_id}}
      history = self.conversation._merge_configs(config)["configurable"]["message_history"]
      
      # Use committed messages instead of proposal state
      messages = history._messages_committed
      
      # Calculate token count if using TokenWindowMemory
      token_count = None
      if isinstance(history, TokenWindowDualStateMemory):
        token_count = sum(history._count_tokens(msg.content) for msg in messages)
      
      # Update tab name with appropriate count
      if token_count is not None:
        self.right_tab_widget.setTabText(0, f"Context ({token_count}/{history.max_tokens} tokens)")
      else:
        pair_count = len(messages) // 2
        self.right_tab_widget.setTabText(0, f"Context {pair_count}/5")

      # Clear current display
      self.context_display.clear()

      # Format and display messages
      cursor = self.context_display.textCursor()
      format = self.context_display.currentCharFormat()

      for msg in messages:
        if isinstance(msg, HumanMessage):
          format.setForeground(self.colors["dark" if self.is_dark_mode else "light"]["canon"])
          cursor.insertText("User: ", format)
          cursor.insertText(f"{msg.content}\n\n", format)
        else:  # Assistant message
          format.setForeground(self.colors["dark" if self.is_dark_mode else "light"]["fg"])
          cursor.insertText("Assistant: ", format)
          cursor.insertText(f"{msg.content}\n\n", format)

        self.context_display.setTextCursor(cursor)

  def send_message(self):
    """Handle sending messages based on selected tab"""
    current_tab = self.input_tabs.currentIndex()

    if current_tab == 3:  # Custom System Prompt tab
      new_system_prompt = self.system_input.toPlainText().strip()
      if new_system_prompt:
        self.system_prompt = new_system_prompt
        self.save_system_prompt()
        self.system_prompt = self.prompt_manager.get_active_prompt()
        self.initialize_llm()
        print("System prompt updated and saved.")
      return

    try:
      # Get input based on current tab
      if current_tab == 0:  # Edit Blue tab
        user_input = self.continue_input.toPlainText().strip()
      elif current_tab == 1:  # Continue next section
        user_input = self.continue_input.toPlainText().strip()
        self.commit_blue_text()  # Commit before continuing
      elif current_tab == 2:  # Rewrite previous section
        user_input = self.rewrite_input.toPlainText().strip()
        # Discard current proposal before rewriting
        config = {"configurable": {"session_id": self.session_id}}
        history = self.conversation._merge_configs(config)["configurable"]["message_history"]
        history.discard_proposal()  # This will restore from committed state

      # Wrap input with XML tags if specified
      xml_tag = self.xml_tag_input.text().strip()
      if user_input or xml_tag:
        user_input = self.wrap_text_with_xml(user_input, xml_tag)
        self.current_user_input = user_input  # track last user input

        # Get history before the new response
        config = {"configurable": {"session_id": self.session_id}}
        history = self.conversation._merge_configs(config)["configurable"]["message_history"]

        # Backup current state before getting LLM response
        history.prepare_for_response()

        # # Add user message to history
        # history.add_message(HumanMessage(content=user_input))

        # Invoke the chain with message history
        response = self.conversation.invoke(
          {"input": user_input}, 
          config={"configurable": {"session_id": self.session_id}}
        )

        # Extract the response content
        response_text = response.content if hasattr(response, "content") else str(response)

        # Update conversation log with full response (including think tags)
        self.update_conversation_log(user_input, response_text)

        # Extract narrative content
        narrative_parts = [part.strip() for part in re.split(r"\n*<think>.*?</think>\n*", response_text, flags=re.DOTALL) if part.strip()]
        # Join parts and strip any leading/trailing whitespace to match Edit Blue behavior
        narrative_text = " ".join(narrative_parts).strip()

        # Remove last AI message if it exists
        if history.messages and isinstance(history.messages[-1], AIMessage):
            history._messages_proposal.pop()
            
        # Add AI narrative response to history without think tags
        history.add_message(AIMessage(content=narrative_text))

        # Extract thinking content
        think_blocks = re.findall(r"<think>\n*(.*?)\n*</think>", response_text, flags=re.DOTALL)
        thinking_content = "\n\n".join(block.strip() for block in think_blocks)
        self.thinking_display.setText(thinking_content)

        # Update blue text
        self.update_blue_text(narrative_text)

        # Clear input of current tab
        if current_tab == 1:
          self.continue_input.clear()
        elif current_tab == 2:
          self.rewrite_input.clear()

        # Update context display
        self.update_context_display()

    except Exception as e:
      print(f"Error: {str(e)}")

  def update_conversation_log(self, user_input: str, response: str):
    """Update the conversation log with user input and AI response"""
    cursor = self.conversation_log.textCursor()
    format = self.conversation_log.currentCharFormat()

    # Log user input
    format.setForeground(self.colors["dark" if self.is_dark_mode else "light"]["canon"])  # Changed to canon (grey)
    cursor.insertText("User: ", format)
    cursor.insertText(f"{user_input}\n\n")

    # Log assistant response
    format.setForeground(self.colors["dark" if self.is_dark_mode else "light"]["fg"])
    cursor.insertText("Assistant: ", format)

    # Split and format response based on XML think tags
    parts = re.split(r"(\n*<think>\n*.*?\n*</think>\n*)", response, flags=re.DOTALL)
    for part in parts:
      if part and part.strip():  # Only process non-empty parts
        if "<think>" in part:
          # Extract and format think block content
          think_content = re.sub(r"\n*<think>\n*(.*?)\n*</think>\n*", r"\1", part, flags=re.DOTALL)
          format.setForeground(self.colors["dark" if self.is_dark_mode else "light"]["canon"])
          cursor.insertText(think_content.strip() + "\n", format)
        else:
          # Format regular text
          format.setForeground(self.colors["dark" if self.is_dark_mode else "light"]["fg"])
          cursor.insertText(part.strip() + "\n", format)

    cursor.insertText("\n")
    self.conversation_log.setTextCursor(cursor)
    self.conversation_log.ensureCursorVisible()

  def load_story(self, empty: bool = False):
    """Handle story file loading or create empty story
    Args:
      empty (bool): If True, simulates loading an empty file without path
    """
    if not self.check_unsaved_changes():  # Check handled by GUI parent class
      return

    # Simulate loading empty file
    self.current_file_path = None
    textfile_content = ""
    self.update_blue_text("")  # Use central method instead of direct assignment

    if not empty:
      # Regular file loading
      file_path = self.get_open_file_name("Load Story", "Text Files (*.txt);;All Files (*.*)")
      if not file_path:
        return

      try:
        with open(file_path, "r", encoding="utf-8") as f:
          textfile_content = f.read()
        self.current_file_path = file_path
      except Exception as e:
        self.show_error_message("Error", f"Failed to load story: {str(e)}")
        print(f"Error loading story: {str(e)}")
        return

    if textfile_content:
      # Clean any think tags from the loaded content
      textfile_content = re.sub(r"\n*<think>.*?</think>\n*", "", textfile_content, flags=re.DOTALL).strip()
      # Split into meaningful chunks with minimum content
      MIN_CHUNK_SIZE = 2000  # characters
      raw_chunks = [chunk.strip() for chunk in textfile_content.split("\n\n") if chunk.strip()]

      meaningful_chunks = []
      current_chunk = []
      current_length = 0

      for chunk in raw_chunks:
        current_chunk.append(chunk)
        current_length += len(chunk)

        if current_length >= MIN_CHUNK_SIZE:
          meaningful_chunks.append("\n\n".join(current_chunk))
          current_chunk = []
          current_length = 0

      # Add any remaining content as final chunk
      if current_chunk:
        meaningful_chunks.append("\n\n".join(current_chunk))

      # Store as validated canon
      self.canon_validated = meaningful_chunks
    else:
      self.canon_validated = []

    # Initialize new conversation
    self.system_prompt = self.prompt_manager.get_active_prompt()  # Add this line
    self.initialize_llm()

    if textfile_content:
      # Simulate conversation history with last chunks (up to 5)
      chunks_for_history = self.canon_validated[-5:] if len(self.canon_validated) > 5 else self.canon_validated
      for chunk in chunks_for_history:
        self.simulate_conversation_turn(chunk)
    else:
      self.update_blue_text("")  # Use central method instead of direct assignment

    # Update displays
    self.update_story_display()
    self.update_context_display()

    # Update window title and status
    if empty:
      self.setWindowTitle("Narrative Collaboration System")
      print("Created new empty story")
    else:
      self.setWindowTitle(f"Narrative Collaboration System - {os.path.basename(self.current_file_path)}")
      print(f"Loaded story from: {self.current_file_path}")
      if textfile_content:
        print(f"Created simulated history with {len(chunks_for_history)} chunks")

  def save_story(self, save_as=False):
    """Handle story saving"""
    if save_as or not self.current_file_path:
      file_path = self.get_save_file_name("Save Story", "", "Text Files (*.txt);;All Files (*.*)")
      if not file_path:
        return
      self.current_file_path = file_path

    try:
      # Combine all validated content
      story_content = "\n\n".join(self.canon_validated)

      with open(self.current_file_path, "w", encoding="utf-8") as f:
        f.write(story_content)

      # Update window title
      self.setWindowTitle(f"Narrative Collaboration System - {os.path.basename(self.current_file_path)}")

      # Update status
      print(f"Story saved to: {self.current_file_path}")

    except Exception as e:
      self.show_error_message("Error", f"Failed to save story: {str(e)}")

  def update_blue_text(self, text: str):
    """Central method to update blue text state"""
    # Check if edit_input already contains the same text
    if self.edit_input.toPlainText() != text:
      self.current_narrative = text
      # Temporarily block signals to avoid recursive updates
      self.edit_input.blockSignals(True)
      self.edit_input.setPlainText(text)
      self.edit_input.blockSignals(False)
    else:
      # Just update current_narrative if text is different
      if self.current_narrative != text:
        self.current_narrative = text

    self.update_story_display()

  def restore_previous_response(self):
    """Restore the previous AI response if available"""
    if hasattr(self, "conversation"):
      config = {"configurable": {"session_id": self.session_id}}
      history = self.conversation._merge_configs(config)["configurable"]["message_history"]
      if history.last_response:
        self.update_blue_text(history.last_response.content)
      else:
        self.update_blue_text("")

if __name__ == "__main__":
  from PySide6.QtWidgets import QApplication

  app = QApplication(sys.argv)
  window = Narrative()  # Instantiate Narrative class
  window.show()
  sys.exit(app.exec())
