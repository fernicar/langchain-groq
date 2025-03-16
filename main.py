import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
 QHBoxLayout, QTextEdit, QLabel, QPushButton, QCheckBox, QFrame, QTabWidget,
 QSpinBox, QSplitter, QComboBox, QSizePolicy, QToolBar, QLineEdit, QFileDialog,
 QMessageBox, QDoubleSpinBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor, QPalette, QAction
from dotenv import load_dotenv
import os
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

class APIMonitorCallback(BaseCallbackHandler):
  def __init__(self, api_monitor):
    super().__init__()
    self.api_monitor = api_monitor

  def on_llm_start(self, serialized, prompts, **kwargs):
    self.api_monitor.append(f"RAW API REQUEST:")
    self.api_monitor.append(f"serialized: {serialized}")
    self.api_monitor.append(f"prompts: {prompts}")
    self.api_monitor.append(f"kwargs: {kwargs}")

  def on_llm_end(self, response, **kwargs):
    self.api_monitor.append(f"RAW API RESPONSE:")
    self.api_monitor.append(f"response: {response}")
    self.api_monitor.append(f"kwargs: {kwargs}")

  def on_llm_error(self, error, **kwargs):
    self.api_monitor.append(f"RAW API ERROR:")
    self.api_monitor.append(f"error: {error}")
    self.api_monitor.append(f"kwargs: {kwargs}")

  def on_llm_new_token(self, token: str, **kwargs):
    """Called when streaming is enabled and a new token is received"""
    self.api_monitor.append(f"Token: {token}")

class SystemPromptManager:
  DEFAULT_PROMPT = """Eres un colaborador narrativo. Tu papel es ayudar a crear historias mientras muestras tu proceso de pensamiento.
  Cuando pienses en la narrativa.
  Mantén las respuestas enfocadas en el desarrollo de la narrativa."""

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
        with open(self.config_path, 'r', encoding='utf-8') as f:
          return json.load(f)
      except json.JSONDecodeError:
        # If JSON is invalid, create new default config
        return self._create_default_config()
    return self._create_default_config()

  def _create_default_config(self) -> dict:
    """Create default configuration"""
    default_config = {
      "active_prompt": "default",
      "prompts": {
        "default": {
          "content": self.DEFAULT_PROMPT,
          "created_at": datetime.now(timezone.utc).isoformat(),
          "last_used": datetime.now(timezone.utc).isoformat()
        }
      }
    }
    self._save_config(default_config)
    return default_config

  def _save_config(self, config: dict) -> None:
    """Save config to file"""
    with open(self.config_path, 'w', encoding='utf-8') as f:
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
    self.prompts["prompts"][name] = {
      "content": content,
      "created_at": self.prompts["prompts"].get(name, {}).get("created_at", datetime.now(timezone.utc).isoformat()),
      "last_used": datetime.now(timezone.utc).isoformat()
    }
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

class WindowBufferHistory(BaseChatMessageHistory):
  def __init__(self):
    self._messages = [] # Use _messages as internal storage
    self.max_tokens = 5 # Keep last 5 message pairs

  def add_message(self, message):
    self._messages.append(message)
    # Keep only last 5 pairs of messages
    if len(self._messages) > 10: # 5 pairs = 10 messages
      self._messages = self._messages[-10:]

  def clear(self):
    self._messages = []

  @property
  def messages(self) -> List[BaseMessage]:
    return self._messages

class NarrativeGUI(QMainWindow):
  def __init__(self):
    super().__init__()
    
    # Add this line after super().__init__()
    self.send_shortcut = QAction("Send", self)
    self.send_shortcut.setShortcut("Ctrl+Return")  # Ctrl+Enter/Return
    self.send_shortcut.triggered.connect(self.send_message)
    self.addAction(self.send_shortcut)
    
    # Add file path tracking
    self.current_file_path = None

    # Initialize system prompt from prompt manager
    self.prompt_manager = SystemPromptManager()
    self.system_prompt = self.prompt_manager.get_active_prompt()

    # Initialize other attributes
    self.current_narrative = ""
    self.canon_validated = []
    self.font_size = 10
    self.is_dark_mode = True
    self.colors = {
      'dark': {
        'bg': '#2b2b2b',
        'fg': '#ffffff',
        'canon': '#a9a9a9',
        'current': '#6495ed',
        'xml': '#98fb98'
      },
      'light': {
        'bg': '#ffffff',
        'fg': '#000000',
        'canon': '#696969',
        'current': '#4169e1',
        'xml': '#228b22'
      }
    }

    # Initialize default values for LLM parameters
    self.temperature = 0.7
    self.max_tokens = 4096

    # Setup UI
    self.setWindowTitle("Narrative Collaboration System")
    self.resize(1280, 720)

    # Create main layout
    main_layout = QVBoxLayout()

    # Create main vertical splitter
    main_splitter = QSplitter(Qt.Vertical)

    # Create widget for display areas
    display_widget = QWidget()
    display_layout = QVBoxLayout(display_widget)
    display_layout.setContentsMargins(0, 0, 0, 0)
    self.setup_display_areas(display_layout)

    # Setup input tabs
    self.setup_input_tabs()
    self.input_tabs.currentChanged.connect(self.on_tab_changed)

    # Add widgets to main splitter
    main_splitter.addWidget(display_widget)
    main_splitter.addWidget(self.input_tabs)

    # Set initial sizes
    main_splitter.setSizes([2 * self.height() // 3, self.height() // 3])

    # Add main splitter to layout
    main_layout.addWidget(main_splitter)

    # Create central widget and set layout
    central_widget = QWidget()
    central_widget.setLayout(main_layout)
    self.setCentralWidget(central_widget)

    # Setup toolbar and add it to the bottom
    toolbar = self.setup_toolbar()
    self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, toolbar)

    # Initialize primed history to none
    self.primed_history = None

    # Now initialize LLM after toolbar is setup
    self.initialize_llm()

    # Apply initial theme
    self.toggle_theme(self.is_dark_mode)

    # Setup menu bar
    self.setup_menu_bar()

  def setup_display_areas(self, layout):
    """Setup the main display areas"""
    # Create splitter for resizable sections
    display_splitter = QSplitter(Qt.Horizontal)

    # Left side: Story display
    left_widget = QWidget()
    left_layout = QVBoxLayout(left_widget)
    left_layout.setContentsMargins(0, 0, 0, 0)

    # Add instruction label
    instruction_label = QLabel("Story Development (Blue text is a proposal - decide if you want to keep it as part of the story)")
    instruction_label.setWordWrap(True)
    instruction_label.setStyleSheet("font-weight: bold;")
    left_layout.addWidget(instruction_label)

    # Story display
    story_frame = QFrame()
    story_layout = QVBoxLayout(story_frame)
    story_layout.setContentsMargins(0, 0, 0, 0)
    self.story_display = QTextEdit()
    self.story_display.setReadOnly(True)
    self.story_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    story_layout.addWidget(self.story_display)
    left_layout.addWidget(story_frame)

    # Right side: Tabbed interface for Thinking Process, Conversation Logs, and API Monitor
    right_widget = QWidget()
    right_layout = QVBoxLayout(right_widget)
    right_layout.setContentsMargins(0, 0, 0, 0)

    # Create tab widget
    tab_widget = QTabWidget()
    tab_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # Thinking Process tab
    thinking_tab = QWidget()
    thinking_layout = QVBoxLayout(thinking_tab)
    thinking_layout.setContentsMargins(0, 0, 0, 0)
    self.thinking_display = QTextEdit()
    self.thinking_display.setReadOnly(True)
    # Update Thinking Process placeholder
    self.thinking_display.setPlaceholderText(
      "Thinking Process Display:\n\n"
      "• AI's reasoning process will appear here in XML tags\n"
      "• Example: <think>Analyzing story context and planning next scene...</think>\n"
      "• System notifications and status updates also show here\n"
      "• Model responses are parsed to highlight reasoning in a different color"
    )
    self.thinking_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    thinking_layout.addWidget(self.thinking_display)
    tab_widget.addTab(thinking_tab, "Thinking Process")

    # Conversation Logs tab
    conversation_tab = QWidget()
    conversation_layout = QVBoxLayout(conversation_tab)
    conversation_layout.setContentsMargins(0, 0, 0, 0)
    self.conversation_log = QTextEdit()
    self.conversation_log.setReadOnly(True)
    # Update Conversation Log placeholder
    self.conversation_log.setPlaceholderText(
      "Conversation History:\n\n"
      "• Full dialogue between you and the AI will be recorded here\n"
      "• User inputs are prefixed with 'User:'\n"
      "• AI responses are prefixed with 'Assistant:'\n"
      "• AI's thinking process is highlighted in a distinct color\n"
      "• Helps track the evolution of your story development"
    )
    self.conversation_log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    conversation_layout.addWidget(self.conversation_log)
    tab_widget.addTab(conversation_tab, "Conversation Logs")

    # API Monitor tab
    api_tab = QWidget()
    api_layout = QVBoxLayout(api_tab)
    api_layout.setContentsMargins(0, 0, 0, 0)

    # API Monitor display
    self.api_monitor = QTextEdit()
    self.api_monitor.setReadOnly(True)
    # Update API Monitor placeholder
    self.api_monitor.setPlaceholderText(
      "API Monitor Display:\n\n"
      "• Shows all API interactions with the AI model\n"
      "• Displays model name, timestamp, and token usage\n"
      "• Helps track API costs and performance\n"
      "• Records any API errors or warnings\n"
      "• Use 'Clear Monitor' button to reset the display\n"
      "• Useful for debugging and optimization"
    )
    self.api_monitor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    api_layout.addWidget(self.api_monitor)

    # Clear button
    clear_button = QPushButton("Clear Monitor")
    clear_button.clicked.connect(self.api_monitor.clear)
    api_layout.addWidget(clear_button)

    tab_widget.addTab(api_tab, "API Monitor")

    right_layout.addWidget(tab_widget)

    # Add widgets to splitter
    display_splitter.addWidget(left_widget)
    display_splitter.addWidget(right_widget)

    # Set initial sizes (2:1 ratio)
    display_splitter.setSizes([2 * display_splitter.width() // 3, display_splitter.width() // 3])

    # Add splitter to layout
    layout.addWidget(display_splitter)

  def handle_toggle(self, current_toggle, other_toggle, state):
    """Handle mutual exclusion for toggles"""
    if state == Qt.Checked:
      other_toggle.setChecked(False)
    elif state == Qt.Unchecked and not other_toggle.isChecked():
      # If unchecking current and other is not checked, force current to stay checked
      current_toggle.setChecked(True)

  def update_font_size(self, size):
    """Update font size for all text widgets"""
    self.font_size = size
    font = QFont("Default", size)

    # Update font for display areas
    self.story_display.setFont(font)
    self.thinking_display.setFont(font)
    self.conversation_log.setFont(font)

    # Update font for input tabs
    for widget in self.findChildren(QTextEdit):
      widget.setFont(font)

    # Update font for combo boxes
    for widget in self.findChildren(QComboBox):
      widget.setFont(font)

    # Update font for labels
    for widget in self.findChildren(QLabel):
      widget.setFont(font)

    # Update font for buttons
    for widget in self.findChildren(QPushButton):
      widget.setFont(font)

    # Update font for tab widgets
    for tab_widget in self.findChildren(QTabWidget):
      tab_widget.setFont(font)
      tab_bar = tab_widget.tabBar()
      tab_bar.setFont(font)

    # Update font for line edit fields
    for widget in self.findChildren(QLineEdit):
      widget.setFont(font)

  def toggle_theme(self, checked):
    self.is_dark_mode = checked
    theme = 'dark' if checked else 'light'

    # Set application palette
    palette = self.palette()
    palette.setColor(self.backgroundRole(), QColor(self.colors[theme]['bg']))
    palette.setColor(self.foregroundRole(), QColor(self.colors[theme]['fg']))
    palette.setColor(QPalette.ColorRole.Window, QColor(self.colors[theme]['bg']))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(self.colors[theme]['fg']))
    palette.setColor(QPalette.ColorRole.Base, QColor(self.colors[theme]['bg']))
    palette.setColor(QPalette.ColorRole.Text, QColor(self.colors[theme]['fg']))
    palette.setColor(QPalette.ColorRole.Button, QColor(self.colors[theme]['bg']))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(self.colors[theme]['fg']))
    self.setPalette(palette)

    # Comprehensive style sheet for all widgets
    style_sheet = f"""
      QMainWindow {{
        background-color: {self.colors[theme]['bg']};
        color: {self.colors[theme]['fg']};
      }}
      QMenuBar {{
        background-color: {self.colors[theme]['bg']};
        color: {self.colors[theme]['fg']};
      }}
      QMenuBar::item {{
        background-color: {self.colors[theme]['bg']};
        color: {self.colors[theme]['fg']};
      }}
      QMenuBar::item:selected {{
        background-color: {self.colors[theme]['fg']};
        color: {self.colors[theme]['bg']};
      }}
      QWidget {{
        background-color: {self.colors[theme]['bg']};
        color: {self.colors[theme]['fg']};
      }}
      QTextEdit {{
        background-color: {self.colors[theme]['bg']};
        color: {self.colors[theme]['fg']};
        border: 1px solid {self.colors[theme]['fg']};
      }}
      QTabWidget::pane {{
        border: 1px solid {self.colors[theme]['fg']};
        background-color: {self.colors[theme]['bg']};
      }}
      QTabBar::tab {{
        background-color: {self.colors[theme]['bg']};
        color: {self.colors[theme]['fg']};
        padding: 8px;
        border: 1px solid {self.colors[theme]['fg']};
        margin-right: 2px;
      }}
      QTabBar::tab:selected {{
        background-color: {self.colors[theme]['fg']};
        color: {self.colors[theme]['bg']};
      }}
      QTitleBar {{
        background-color: {self.colors[theme]['bg']};
        color: {self.colors[theme]['fg']};
      }}
      QPushButton {{
        background-color: {self.colors[theme]['bg']};
        color: {self.colors[theme]['fg']};
        border: 1px solid {self.colors[theme]['fg']};
        padding: 5px;
        min-width: 80px;
      }}
      QPushButton:hover {{
        background-color: {self.colors[theme]['fg']};
        color: {self.colors[theme]['bg']};
      }}
      QLabel {{
        color: {self.colors[theme]['fg']};
        background-color: transparent;
      }}
      QCheckBox {{
        color: {self.colors[theme]['fg']};
        background-color: transparent;
      }}
      QSpinBox {{
        background-color: {self.colors[theme]['bg']};
        color: {self.colors[theme]['fg']};
        border: 1px solid {self.colors[theme]['fg']};
      }}
      QFrame {{
        background-color: {self.colors[theme]['bg']};
        color: {self.colors[theme]['fg']};
      }}
    """

    # Apply style sheet to the main window
    self.setStyleSheet(style_sheet)

    # Force update of story display to refresh colors
    self.update_story_display()

    # Update Edit Blue input color
    self.edit_input.setStyleSheet(f"""
      QTextEdit {{
        color: {self.colors[theme]['current']};
        background-color: {self.colors[theme]['bg']};
        border: 1px solid {self.colors[theme]['fg']};
      }}
    """)

  def initialize_llm(self):
    """Initialize the language model and conversation chain"""
    load_dotenv()

    # Get the currently selected model from the combo box
    selected_model = self.model_selector.currentText()
    
    # Create custom callback
    callback = APIMonitorCallback(self.api_monitor)
    
    llm = ChatGroq(
      api_key=os.environ['GROQ_API_KEY'],
      model_name=selected_model,
      temperature=self.temperature,
      max_tokens=self.max_tokens,
      streaming=False,
      verbose=True,
      callbacks=[callback] # Add the custom callback
    )

    # Create the chat prompt template
    prompt = ChatPromptTemplate.from_messages([
      SystemMessage(content=self.system_prompt),
      MessagesPlaceholder(variable_name="history"),
      HumanMessagePromptTemplate.from_template("{input}")
    ])

    # Create the runnable chain
    chain = prompt | llm

    # Create the runnable with message history
    if self.primed_history is not None:
      history = self.primed_history
      self.primed_history = None  # Consume the primed history
    else:
      history = WindowBufferHistory()

    self.conversation = RunnableWithMessageHistory(
      chain,
      lambda session_id: history,
      input_messages_key="input",
      history_messages_key="history"
    )

    # Store session ID for this instance
    self.session_id = "default_session"

  def update_story_display(self):
    """Update the story display with proper formatting"""
    self.story_display.clear()
    cursor = self.story_display.textCursor()

    # Show load prompt if no content
    if not self.canon_validated and not self.current_narrative:
      format = self.story_display.currentCharFormat()
      theme = 'dark' if self.is_dark_mode else 'light'
      format.setForeground(QColor(self.colors[theme]['fg']))
      cursor.setCharFormat(format)
      cursor.insertText(
        "Load your story file using File → Load Story... (Ctrl+O)\n\n"
        "Story Development Display:\n\n"
        "• Black/White text represents validated/saved story content\n"
        "• Blue text shows current AI proposal or your edits\n"
        "• Edit blue proposals in the 'Edit Blue' tab below\n"
        "• Use 'Save Blue & Continue' to validate blue text\n"
        "• Start writing your story by using the 'Edit Blue' tab\n"
        "• Or let AI start by using 'Save Blue & Continue'"
      )
      return

    # Insert validated canon in Black/White
    for piece in self.canon_validated:
      format = self.story_display.currentCharFormat()
      theme = 'dark' if self.is_dark_mode else 'light'
      format.setForeground(QColor(self.colors[theme]['canon']))
      cursor.setCharFormat(format)
      cursor.insertText(piece + "\n\n")

    # Insert current narrative in blue
    if self.current_narrative:
      if self.canon_validated:
        cursor.insertText("─" * 40 + "\n\n")
      format = self.story_display.currentCharFormat()
      theme = 'dark' if self.is_dark_mode else 'light'
      format.setForeground(QColor(self.colors[theme]['current']))
      cursor.setCharFormat(format)
      cursor.insertText(self.current_narrative + "\n")

    self.story_display.setTextCursor(cursor)

  def wrap_text_with_xml(self, text: str, tag: str) -> str:
    """Wrap text with XML tags if a tag is specified"""
    if not tag.strip():
      return text
    return f"<{tag}>{text}</{tag}>"

  def on_tab_changed(self, index: int):
    """Handle tab changes"""
    # You can add specific logic here if needed when tabs change
    pass

  def send_message(self):
    """Handle sending messages based on selected tab"""
    current_tab = self.input_tabs.currentIndex()

    if current_tab == 3: # Custom System Prompt tab
      new_system_prompt = self.system_input.toPlainText().strip()
      if new_system_prompt:
        self.system_prompt = new_system_prompt
        self.save_system_prompt()
        self.initialize_llm()
        self.thinking_display.append("System prompt updated and saved.")
      return

    try:
      # Get input based on current tab
      if current_tab == 0: # Edit Blue tab
        user_input = self.edit_input.toPlainText().strip()
      elif current_tab == 1: # Continue next section
        user_input = self.continue_input.toPlainText().strip()
        if self.current_narrative:
          self.canon_validated.append(self.current_narrative)
      elif current_tab == 2: # Rewrite previous section
        user_input = self.rewrite_input.toPlainText().strip()

      if user_input:
        # Wrap input with XML tags if specified
        xml_tag = self.xml_tag_input.text().strip()
        if xml_tag:
          user_input = self.wrap_text_with_xml(user_input, xml_tag)

        # Invoke the chain with message history
        response = self.conversation.invoke(
          {"input": user_input},
          config={"configurable": {"session_id": self.session_id}}
        )

        # Extract the response content
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Update conversation log
        self.update_conversation_log(user_input, response_text)

        # Extract narrative content
        narrative_parts = [part.strip()
          for part in re.split(r'<think>.*?</think>', response_text, flags=re.DOTALL)
          if part.strip()]
        self.current_narrative = ' '.join(narrative_parts)

        # Update story display
        self.update_story_display()

        # NEW: Update Edit Blue tab with the current narrative
        self.edit_input.setPlainText(self.current_narrative)

        # Extract thinking content
        think_blocks = re.findall(r'<think>(.*?)</think>', response_text, flags=re.DOTALL)
        thinking_content = '\n\n'.join(block.strip() for block in think_blocks)
        self.thinking_display.setText(thinking_content)

        # Clear input of current tab
        if current_tab == 0:
          self.edit_input.clear()
        elif current_tab == 1:
          self.continue_input.clear()
        elif current_tab == 2:
          self.rewrite_input.clear()

    except Exception as e:
      error_msg = f"Error: {str(e)}"
      self.thinking_display.append(error_msg)
      self.api_monitor.append(f"ERROR: {error_msg}\n\n")

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
      self.thinking_display.append(f"System prompt '{name}' saved and activated.")

  def delete_system_prompt(self):
    """Delete current prompt"""
    name = self.prompt_selector.currentText()
    if self.prompt_manager.delete_prompt(name):
      self.update_prompt_selector()
      self.thinking_display.append(f"System prompt '{name}' deleted.")
    else:
      self.thinking_display.append("Cannot delete default prompt.")

  def update_conversation_log(self, user_input: str, response: str):
    """Update the conversation log with user input and AI response"""
    cursor = self.conversation_log.textCursor()
    format = self.conversation_log.currentCharFormat()

    # Log user input
    format.setForeground(QColor(self.colors['dark' if self.is_dark_mode else 'light']['current']))
    cursor.insertText("User: ", format)
    cursor.insertText(f"{user_input}\n\n")

    # Log assistant response
    format.setForeground(QColor(self.colors['dark' if self.is_dark_mode else 'light']['fg']))
    cursor.insertText("Assistant: ", format)

    # Split and format response based on XML think tags
    parts = re.split(r'(<think>.*?</think>)', response, flags=re.DOTALL)
    for part in parts:
      if part.startswith('<think>'):
        format.setForeground(QColor(self.colors['dark' if self.is_dark_mode else 'light']['xml']))
        thinking = re.sub(r'</?think>', '', part)
        cursor.insertText(thinking + "\n", format)
      else:
        format.setForeground(QColor(self.colors['dark' if self.is_dark_mode else 'light']['fg']))
        cursor.insertText(part + "\n", format)

    cursor.insertText("\n")
    self.conversation_log.setTextCursor(cursor)
    self.conversation_log.ensureCursorVisible()

  def get_available_models(self) -> List[str]:
    """Fetch available models from Groq API"""
    try:
      client = Groq(api_key=os.environ['GROQ_API_KEY'])
      models = client.models.list()
      # Filter and sort models based on our needs
      available_models = []
      for model in models.data: # Access the data attribute of the response
        model_id = model.id if hasattr(model, 'id') else str(model)
        if 'whisper' not in model_id.lower():
          available_models.append(model_id)
      return sorted(available_models, reverse=True)
    except Exception as e:
      if hasattr(self, 'api_monitor'):
        self.api_monitor.append(f"Error fetching models: {str(e)}\n")
      # Fallback to hardcoded models if API fails
      return [
        'qwen-qwq-32b',
        'deepseek-r1-distill-qwen-32b',
        'deepseek-r1-distill-llama-70b',
        'mixtral-8x7b-32768',
        'llama-3.3-70b-versatile'
      ]

  def setup_toolbar(self):
    """Setup the application toolbar"""
    toolbar = QToolBar()

    # Model selector
    model_label = QLabel("Model: ")
    toolbar.addWidget(model_label)

    self.model_selector = QComboBox()
    # Fetch models from API
    available_models = self.get_available_models()
    self.model_selector.addItems(available_models)
    self.model_selector.currentTextChanged.connect(self.on_model_changed)
    toolbar.addWidget(self.model_selector)

    # Add separator
    toolbar.addSeparator()

    # Temperature control
    temp_label = QLabel("Temperature: ")
    toolbar.addWidget(temp_label)

    self.temperature_spinner = QDoubleSpinBox()
    self.temperature_spinner.setRange(0.0, 1.0)
    self.temperature_spinner.setSingleStep(0.1)
    self.temperature_spinner.setValue(self.temperature)
    self.temperature_spinner.valueChanged.connect(self.on_temperature_changed)
    toolbar.addWidget(self.temperature_spinner)

    # Add separator
    toolbar.addSeparator()

    # Max tokens control
    tokens_label = QLabel("Max Tokens: ")
    toolbar.addWidget(tokens_label)

    self.max_tokens_spinner = QSpinBox()
    self.max_tokens_spinner.setRange(1, 32768)
    self.max_tokens_spinner.setValue(self.max_tokens)
    self.max_tokens_spinner.valueChanged.connect(self.on_max_tokens_changed)
    toolbar.addWidget(self.max_tokens_spinner)

    # Add wider separator
    separator = QWidget()
    separator.setFixedWidth(20) # Adjust this value to change the width
    toolbar.addWidget(separator)

    # XML Tag wrapper
    xml_label = QLabel("xml tag:")
    toolbar.addWidget(xml_label)
    self.xml_tag_input = QLineEdit()
    self.xml_tag_input.setPlaceholderText("custom tag")
    self.xml_tag_input.setMaximumWidth(100)
    toolbar.addWidget(self.xml_tag_input)

    # Add wider separator
    separator = QWidget()
    separator.setFixedWidth(20) # Adjust this value to change the width
    toolbar.addWidget(separator)

    # Font size controls
    font_label = QLabel("Font Size:")
    toolbar.addWidget(font_label)
    self.font_size_spinner = QSpinBox()
    self.font_size_spinner.setRange(8, 24)
    self.font_size_spinner.setValue(self.font_size)
    self.font_size_spinner.valueChanged.connect(self.update_font_size)
    toolbar.addWidget(self.font_size_spinner)

    # Add wider separator
    separator = QWidget()
    separator.setFixedWidth(20) # Adjust this value to change the width
    toolbar.addWidget(separator)

    # Dark mode toggle
    self.dark_mode_toggle = QCheckBox("Dark Mode")
    self.dark_mode_toggle.setChecked(True) # Set to True for default dark mode
    self.dark_mode_toggle.stateChanged.connect(self.toggle_theme)
    toolbar.addWidget(self.dark_mode_toggle)

    # Add wider separator
    separator = QWidget()
    separator.setFixedWidth(20) # Adjust this value to change the width
    toolbar.addWidget(separator)

    # Prompt selector
    prompt_label = QLabel("System Prompt: ")
    toolbar.addWidget(prompt_label)

    self.prompt_selector = QComboBox()
    prompts = self.prompt_manager.get_all_prompts()
    self.prompt_selector.addItems(prompts.keys())
    self.prompt_selector.setCurrentText(self.prompt_manager.prompts["active_prompt"])
    self.prompt_selector.currentTextChanged.connect(self.on_prompt_selected)
    toolbar.addWidget(self.prompt_selector)

    # Add wider separator
    separator = QWidget()
    separator.setFixedWidth(20) # Adjust this value to change the width
    toolbar.addWidget(separator)

    # Send button
    send_button = QPushButton("Send")
    send_button.clicked.connect(self.send_message)
    toolbar.addWidget(send_button)
    return toolbar

  def setup_input_tabs(self):
    """Setup the input tabs area"""
    self.input_tabs = QTabWidget()

    # Tab 1: Edit Blue Proposal
    edit_tab = QWidget()
    edit_layout = QVBoxLayout(edit_tab)
    self.edit_input = QTextEdit()
    self.edit_input.setPlaceholderText(
      "Edit Blue Proposal:\n\n"
      "• Type or paste text here to modify the blue proposal\n"
      "• Changes appear instantly in the preview panel above\n"
      "• Use this tab to start your story or edit AI suggestions\n"
      "• Switch to 'Save Blue & Continue' when ready to proceed\n"
      "• Your edits are preserved when switching between tabs"
    )
    # Set text color to match the preview blue color
    theme = 'dark' if self.is_dark_mode else 'light'
    self.edit_input.setStyleSheet(f"""
      QTextEdit {{
        color: {self.colors[theme]['current']};
        background-color: {self.colors[theme]['bg']};
        border: 1px solid {self.colors[theme]['fg']};
      }}
    """)
    self.edit_input.textChanged.connect(self.update_blue_preview)
    edit_layout.addWidget(self.edit_input)
    self.input_tabs.addTab(edit_tab, "Edit Blue")

    # Tab 2: Continue next section
    continue_tab = QWidget()
    continue_layout = QVBoxLayout(continue_tab)
    self.continue_input = QTextEdit()
    self.continue_input.setPlaceholderText(
      "Save Blue & Continue:\n\n"
      "• Current blue text will be saved as permanent Black/White text\n"
      "• Type guidance here for the AI to continue the story\n"
      "• Leave empty to let AI continue based on context alone\n"
      "• AI will generate a new blue proposal as continuation\n"
      "• Use XML tags to structure your guidance (optional)"
    )
    continue_layout.addWidget(self.continue_input)
    self.input_tabs.addTab(continue_tab, "Save Blue && Continue")

    # Tab 3: Rewrite previous section
    rewrite_tab = QWidget()
    rewrite_layout = QVBoxLayout(rewrite_tab)
    self.rewrite_input = QTextEdit()
    self.rewrite_input.setPlaceholderText(
      "Discard Blue & Rewrite:\n\n"
      "• Current blue proposal will be discarded\n"
      "• Type guidance here for the AI to generate new content\n"
      "• AI will provide a completely new blue proposal\n"
      "• Useful when the current proposal needs major changes\n"
      "• Previous Black/White (saved) text remains unchanged"
    )
    rewrite_layout.addWidget(self.rewrite_input)
    self.input_tabs.addTab(rewrite_tab, "Discard Blue && Rewrite")

    # Tab 4: Customize System Prompt (existing)
    system_tab = QWidget()
    system_layout = QVBoxLayout(system_tab)

    # Prompt name input
    name_layout = QHBoxLayout()
    name_label = QLabel("Prompt Name:")
    self.prompt_name_input = QLineEdit()
    name_layout.addWidget(name_label)
    name_layout.addWidget(self.prompt_name_input)
    system_layout.addLayout(name_layout)

    # System prompt input
    self.system_input = QTextEdit()
    self.system_input.setPlainText(self.prompt_manager.get_active_prompt())
    self.prompt_name_input.setText(self.prompt_manager.prompts["active_prompt"])
    system_layout.addWidget(self.system_input)

    # Add buttons for save and delete
    button_layout = QHBoxLayout()
    save_button = QPushButton("Save Prompt")
    save_button.clicked.connect(self.save_system_prompt)
    delete_button = QPushButton("Delete Prompt")
    delete_button.clicked.connect(self.delete_system_prompt)
    button_layout.addWidget(save_button)
    button_layout.addWidget(delete_button)
    system_layout.addLayout(button_layout)

    system_tab.setLayout(system_layout)
    self.input_tabs.addTab(system_tab, "Customize System Prompt")

  def update_blue_preview(self):
    """Update the preview panel's blue text in real-time"""
    self.current_narrative = self.edit_input.toPlainText()
    self.update_story_display()

  def setup_menu_bar(self):
    """Setup the application menu bar"""
    menubar = self.menuBar()

    # File Menu
    file_menu = menubar.addMenu('File')

    # Load Story action
    load_action = QAction('Load Story...', self)
    load_action.setShortcut('Ctrl+O')
    load_action.triggered.connect(self.load_story)
    file_menu.addAction(load_action)

    # Save Story action
    save_action = QAction('Save Story', self)
    save_action.setShortcut('Ctrl+S')
    save_action.triggered.connect(self.save_story)
    file_menu.addAction(save_action)

    # Save As action
    save_as_action = QAction('Save Story As...', self)
    save_as_action.setShortcut('Ctrl+Shift+S')
    save_as_action.triggered.connect(lambda: self.save_story(save_as=True))
    file_menu.addAction(save_as_action)

    # Add separator
    file_menu.addSeparator()

    # Clear Story action
    clear_action = QAction('Clear Story', self)
    clear_action.setShortcut('Ctrl+N')
    clear_action.triggered.connect(self.clear_story)
    file_menu.addAction(clear_action)

  def clear_story(self):
    """Clear the story display and reset story-related states"""
    if self.current_narrative or self.canon_validated:
      reply = QMessageBox.question(
        self,
        'Clear Story',
        'Are you sure you want to clear the current story? All unsaved content will be lost.',
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
      )

      if reply == QMessageBox.Yes:
        # Clear story content
        self.canon_validated = []
        self.current_narrative = ""
        self.current_file_path = None
        
        # Reinitialize LLM to reset conversation history
        self.initialize_llm()
        
        # Update display
        self.update_story_display()
        
        # Reset window title
        self.setWindowTitle("Narrative Collaboration System")
        
        # Clear edit input
        self.edit_input.clear()
        
        # Update status
        self.thinking_display.append("Story cleared.")

  def load_story(self):
    """Handle story file loading"""
    file_path, _ = QFileDialog.getOpenFileName(
      self,
      "Load Story",
      "",
      "Text Files (*.txt);;All Files (*.*)"
    )

    if file_path:
      try:
        with open(file_path, 'r', encoding='utf-8') as f:
          story_content = f.read()

        # Store file path
        self.current_file_path = file_path

        # Clear existing content
        self.canon_validated = []
        self.current_narrative = ""

        # Split into meaningful chunks with minimum content
        MIN_CHUNK_SIZE = 2000 # characters
        raw_chunks = [chunk.strip() for chunk in story_content.split('\n\n') if chunk.strip()]

        meaningful_chunks = []
        current_chunk = []
        current_length = 0

        for chunk in raw_chunks:
          current_chunk.append(chunk)
          current_length += len(chunk)

          if current_length >= MIN_CHUNK_SIZE:
            meaningful_chunks.append('\n\n'.join(current_chunk))
            current_chunk = []
            current_length = 0

        # Add any remaining content as final chunk
        if current_chunk:
          meaningful_chunks.append('\n\n'.join(current_chunk))

        # Store as validated canon
        self.canon_validated = meaningful_chunks

        # Simulate conversation history with last chunks
        history = WindowBufferHistory()
        for chunk in meaningful_chunks[-5:]:
          history.add_message(HumanMessage(content="Continue the story"))
          history.add_message(AIMessage(content=chunk))
        
        # Store the primed history for future use
        self.primed_history = history

        # Reset conversation by reinitializing LLM
        self.initialize_llm()

        # Update display
        self.update_story_display()

        # Update window title and status
        self.setWindowTitle(f"Narrative Collaboration System - {os.path.basename(file_path)}")
        self.thinking_display.append(f"Loaded story from: {file_path}")
        self.thinking_display.append("Previous content loaded as context for the AI.")

      except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to load story: {str(e)}")

  def save_story(self, save_as=False):
    """Handle story saving"""
    if save_as or not self.current_file_path:
      file_path, _ = QFileDialog.getSaveFileName(
        self,
        "Save Story",
        "",
        "Text Files (*.txt);;All Files (*.*)"
      )
      if not file_path:
        return
      self.current_file_path = file_path

    try:
      # Combine all validated content
      story_content = '\n\n'.join(self.canon_validated)

      with open(self.current_file_path, 'w', encoding='utf-8') as f:
        f.write(story_content)

      # Update window title
      self.setWindowTitle(f"Narrative Collaboration System - {os.path.basename(self.current_file_path)}")

      # Update status
      self.thinking_display.append(f"Story saved to: {self.current_file_path}")

    except Exception as e:
      QMessageBox.critical(self, "Error", f"Failed to save story: {str(e)}")

  def closeEvent(self, event):
    """Handle application close event"""
    if self.current_narrative:
      reply = QMessageBox.question(
        self,
        'Unsaved Changes',
        'There is an unsaved blue proposal. Do you want to save it before closing?',
        QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
        QMessageBox.Save
      )

      if reply == QMessageBox.Save:
        # Add current narrative to canon and save
        self.canon_validated.append(self.current_narrative)
        self.save_story()
        event.accept()
      elif reply == QMessageBox.Discard:
        event.accept()
      else: # Cancel
        event.ignore()
    else:
      event.accept()

  def on_temperature_changed(self, value):
    """Handle temperature change"""
    self.temperature = value
    self.initialize_llm()

  def on_max_tokens_changed(self, value):
    """Handle max tokens change"""
    self.max_tokens = value
    self.initialize_llm()

if __name__ == "__main__":
  app = QApplication(sys.argv)
  window = NarrativeGUI()
  window.show()
  sys.exit(app.exec())
