import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QTextEdit, QLabel, QPushButton, 
                              QCheckBox, QFrame, QTabWidget, QSpinBox, QSplitter,
                              QComboBox, QSizePolicy, QToolBar, QLineEdit)  # Added QLineEdit
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QColor, QPalette
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import argparse
from typing import Optional
import json
from datetime import datetime, timezone
from pathlib import Path
import typing as t

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
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Get the default prompt from the currently running instance
        default_prompt = getattr(self, 'DEFAULT_PROMPT', 
            """""")
        
        # Create default config
        default_config = {
            "active_prompt": "default",
            "prompts": {
                "default": {
                    "content": default_prompt,
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

class NarrativeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize system prompt from prompt manager
        self.prompt_manager = SystemPromptManager()
        self.system_prompt = self.prompt_manager.get_active_prompt()
        
        # Initialize other attributes
        self.current_narrative = ""
        self.canon_validated = []
        self.font_size = 12
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
        
        # Setup UI
        self.setWindowTitle("Narrative Collaboration System")
        self.resize(1200, 800)
        
        # Create main layout
        main_layout = QVBoxLayout()
        
        # Setup toolbar
        self.setup_toolbar()
        
        # Create main vertical splitter
        main_splitter = QSplitter(Qt.Vertical)
        
        # Create widget for display areas
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        display_layout.setContentsMargins(0, 0, 0, 0)
        self.setup_display_areas(display_layout)
        
        # Setup input tabs
        self.setup_input_tabs()
        
        # Add widgets to main splitter
        main_splitter.addWidget(display_widget)
        main_splitter.addWidget(self.input_tabs)
        
        # Set initial sizes (2:1 ratio for display:input)
        main_splitter.setSizes([2 * self.height() // 3, self.height() // 3])
        
        # Add main splitter to layout
        main_layout.addWidget(main_splitter)
        
        # Create central widget and set layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Initialize LLM
        self.initialize_llm()
        
        # Apply initial theme
        self.toggle_theme(self.is_dark_mode)

    def setup_main_frame(self):
        self.main_frame = QFrame(self)

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
        self.thinking_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        thinking_layout.addWidget(self.thinking_display)
        tab_widget.addTab(thinking_tab, "Thinking Process")
        
        # Conversation Logs tab
        conversation_tab = QWidget()
        conversation_layout = QVBoxLayout(conversation_tab)
        conversation_layout.setContentsMargins(0, 0, 0, 0)
        self.conversation_log = QTextEdit()
        self.conversation_log.setReadOnly(True)
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
        self.api_monitor.setPlaceholderText("API calls will be displayed here...")
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

    def setup_control_panel(self):
        control_container = QWidget()
        control_layout = QVBoxLayout(control_container)
        
        # Config bar (horizontal layout)
        config_bar = QWidget()
        config_layout = QHBoxLayout(config_bar)
        config_layout.setContentsMargins(0, 0, 0, 0)
        
        # Model selector
        model_label = QLabel("Model:")
        config_layout.addWidget(model_label)
        self.model_selector = QComboBox()
        self.model_selector.addItems([
            'qwen-qwq-32b',
            'deepseek-r1-distill-qwen-32b',
            'deepseek-r1-distill-llama-70b',
            'mixtral-8x7b-32768',
            'llama-3.3-70b-versatile'
        ])
        self.model_selector.currentTextChanged.connect(self.on_model_changed)
        config_layout.addWidget(self.model_selector)
        
        # Dark mode toggle - set checked to True
        self.dark_mode_toggle = QCheckBox("Dark Mode")
        self.dark_mode_toggle.setChecked(True)  # Set to True for default dark mode
        self.dark_mode_toggle.stateChanged.connect(self.toggle_theme)
        config_layout.addWidget(self.dark_mode_toggle)
        
        # Font size controls
        font_label = QLabel("Font Size:")
        config_layout.addWidget(font_label)
        self.font_size_spinner = QSpinBox()
        self.font_size_spinner.setRange(8, 24)
        self.font_size_spinner.setValue(self.font_size)
        self.font_size_spinner.valueChanged.connect(self.update_font_size)
        config_layout.addWidget(self.font_size_spinner)
        
        # Send button
        send_button = QPushButton("Send")
        send_button.clicked.connect(self.send_message)
        config_layout.addWidget(send_button)
        
        # Add config bar to main control layout
        control_layout.addWidget(config_bar)
        
        # Input area label
        input_label = QLabel("Leave it Empty or input Memory Alerts\n(First-person memory responses: 'I remember-', 'I know this was different-', 'I'm certain that-')")
        control_layout.addWidget(input_label)
        
        # Create tab widget for input modes
        input_tabs = QTabWidget()
        
        # Tab 1: Accept and Continue (default)
        accept_tab = QWidget()
        accept_layout = QVBoxLayout(accept_tab)
        self.accept_input = QTextEdit()
        self.accept_input.setMaximumHeight(100)
        self.accept_input.setPlaceholderText("Input text to continue with current narrative...")
        accept_layout.addWidget(self.accept_input)
        input_tabs.addTab(accept_tab, "Accept & Continue")
        
        # Tab 2: Reject and Retry
        reject_tab = QWidget()
        reject_layout = QVBoxLayout(reject_tab)
        self.reject_input = QTextEdit()
        self.reject_input.setMaximumHeight(100)
        self.reject_input.setPlaceholderText("Input text to try a different approach...")
        reject_layout.addWidget(self.reject_input)
        input_tabs.addTab(reject_tab, "Reject & Retry")
        
        control_layout.addWidget(input_tabs)
        
        self.main_layout.addWidget(control_container)

    def setup_settings_panel(self):
        settings_frame = QFrame()
        settings_layout = QHBoxLayout(settings_frame)
        
        # Font size controls
        font_control_group = QFrame()
        font_layout = QHBoxLayout(font_control_group)
        font_layout.setContentsMargins(0, 0, 0, 0)
        
        font_label = QLabel("Font Size:")
        font_layout.addWidget(font_label)
        
        self.font_size_spinner = QSpinBox()
        self.font_size_spinner.setRange(8, 24)
        self.font_size_spinner.setValue(self.font_size)
        self.font_size_spinner.valueChanged.connect(self.update_font_size)
        font_layout.addWidget(self.font_size_spinner)
        
        settings_layout.addWidget(font_control_group)
        
        # Dark mode toggle - set checked to True
        self.dark_mode_toggle = QCheckBox("Dark Mode")
        self.dark_mode_toggle.setChecked(True)  # Set to True for default dark mode
        self.dark_mode_toggle.stateChanged.connect(self.toggle_theme)
        settings_layout.addWidget(self.dark_mode_toggle)
        
        # Add stretch to push everything to the left
        settings_layout.addStretch()
        
        self.main_layout.addWidget(settings_frame)

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

    def initialize_llm(self):
        """Initialize the language model and conversation chain"""
        load_dotenv()
        
        # Get the currently selected model from the combo box
        selected_model = self.model_selector.currentText()
        
        llm = ChatGroq(
            api_key=os.environ['GROQ_API_KEY'],
            model_name=selected_model
        )
        
        system_prompt = """Eres un colaborador narrativo. Tu papel es ayudar a crear historias.
        Mantén las respuestas de narrativa solamente, sin explicaciones, listas para publicar."""
        
        memory = ConversationBufferWindowMemory(
            k=5,
            return_messages=True,
            memory_key="history"
        )
        
        self.conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=True,
            prompt=ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}")
            ])
        )

    def update_story_display(self):
        self.story_display.clear()
        cursor = self.story_display.textCursor()
        
        # Insert validated canon in black/grey
        for piece in self.canon_validated:
            format = self.story_display.currentCharFormat()
            theme = 'dark' if self.is_dark_mode else 'light'
            format.setForeground(QColor(self.colors[theme]['canon']))
            cursor.setCharFormat(format)
            cursor.insertText(piece + "\n\n")
        
        # Insert current narrative in blue/light blue
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

    def send_message(self):
        """Handle sending messages based on selected tab"""
        current_tab = self.input_tabs.currentIndex()
        
        if current_tab == 2:  # Custom System Prompt tab
            new_system_prompt = self.system_input.toPlainText().strip()
            if new_system_prompt:
                self.system_prompt = new_system_prompt
                self.save_system_prompt()
                self.initialize_llm()
                self.thinking_display.append("System prompt updated and saved.")
            return
        
        try:
            # Get input based on current tab
            if current_tab == 0:  # Continue next section
                user_input = self.continue_input.toPlainText().strip()
                if self.current_narrative:
                    self.canon_validated.append(self.current_narrative)
            elif current_tab == 1:  # Rewrite previous section
                user_input = self.rewrite_input.toPlainText().strip()
            
            if user_input:
                # Wrap input with XML tags if specified
                xml_tag = self.xml_tag_input.text().strip()
                if xml_tag:
                    user_input = self.wrap_text_with_xml(user_input, xml_tag)
                
                # Log the prompt being sent
                messages = self.conversation.memory.load_memory_variables({})["history"]
                prompt_log = "=== API Call ===\n"
                prompt_log += f"System: {self.system_prompt}\n\n"
                
                for message in messages:
                    role = "Human" if isinstance(message, HumanMessage) else "Assistant"
                    prompt_log += f"{role}: {message.content}\n\n"
                
                prompt_log += f"Human: {user_input}\n"
                prompt_log += "=" * 50 + "\n\n"
                
                self.api_monitor.append(prompt_log)
                
                # Get response from LLM
                response = self.conversation.predict(input=user_input)
                
                # Log the response
                self.api_monitor.append(f"Response:\n{response}\n\n")
                self.api_monitor.append("=" * 50 + "\n\n")
                
                # Update conversation log
                self.update_conversation_log(user_input, response)
                
                # Extract narrative content
                narrative_parts = [part.strip() 
                                 for part in re.split(r'<think>.*?</think>', response, flags=re.DOTALL) 
                                 if part.strip()]
                self.current_narrative = ' '.join(narrative_parts)
                
                # Update story display
                self.update_story_display()
                
                # Extract thinking content
                think_blocks = re.findall(r'<think>(.*?)</think>', response, flags=re.DOTALL)
                thinking_content = '\n\n'.join(block.strip() for block in think_blocks)
                self.thinking_display.setText(thinking_content)
                
                # Clear input of current tab
                if current_tab == 0:
                    self.continue_input.clear()
                elif current_tab == 1:
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
        format.setForeground(QColor(self.colors['dark' if self.is_dark_mode else 'light']['fg']))
        cursor.insertText("User: ", format)
        cursor.insertText(f"{user_input}\n\n")
        
        # Log assistant response
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

    def setup_toolbar(self):
        """Setup the application toolbar"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Model selector
        model_label = QLabel("Model: ")
        toolbar.addWidget(model_label)
        
        self.model_selector = QComboBox()
        self.model_selector.addItems([
            'qwen-qwq-32b',
            'deepseek-r1-distill-qwen-32b',
            'deepseek-r1-distill-llama-70b',
            'mixtral-8x7b-32768',
            'llama-3.3-70b-versatile'
        ])
        self.model_selector.currentTextChanged.connect(self.on_model_changed)
        toolbar.addWidget(self.model_selector)
        
        # Add wider separator
        separator = QWidget()
        separator.setFixedWidth(20)  # Adjust this value to change the width
        toolbar.addWidget(separator)
        
        # XML Tag wrapper
        xml_label = QLabel("XML Tag:")
        toolbar.addWidget(xml_label)
        self.xml_tag_input = QLineEdit()
        self.xml_tag_input.setPlaceholderText("Enter tag name")
        self.xml_tag_input.setMaximumWidth(100)
        toolbar.addWidget(self.xml_tag_input)
        
        # Add wider separator
        separator = QWidget()
        separator.setFixedWidth(20)  # Adjust this value to change the width
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
        separator.setFixedWidth(20)  # Adjust this value to change the width
        toolbar.addWidget(separator)
        
        # Dark mode toggle
        self.dark_mode_toggle = QCheckBox("Dark Mode")
        self.dark_mode_toggle.setChecked(True)  # Set to True for default dark mode
        self.dark_mode_toggle.stateChanged.connect(self.toggle_theme)
        toolbar.addWidget(self.dark_mode_toggle)
        
        # Add wider separator
        separator = QWidget()
        separator.setFixedWidth(20)  # Adjust this value to change the width
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
        separator.setFixedWidth(20)  # Adjust this value to change the width
        toolbar.addWidget(separator)
        
        # Send button
        send_button = QPushButton("Send")
        send_button.clicked.connect(self.send_message)
        toolbar.addWidget(send_button)

    def setup_input_tabs(self):
        """Setup the input tabs area"""
        self.input_tabs = QTabWidget()
        
        # Tab 1: Continue next section
        continue_tab = QWidget()
        continue_layout = QVBoxLayout(continue_tab)
        self.continue_input = QTextEdit()
        self.continue_input.setPlaceholderText("Input text to continue with current narrative...")
        continue_layout.addWidget(self.continue_input)
        self.input_tabs.addTab(continue_tab, "Continue Next Section")
        
        # Tab 2: Rewrite previous section
        rewrite_tab = QWidget()
        rewrite_layout = QVBoxLayout(rewrite_tab)
        self.rewrite_input = QTextEdit()
        self.rewrite_input.setPlaceholderText("Input text to rewrite the previous section...")
        rewrite_layout.addWidget(self.rewrite_input)
        self.input_tabs.addTab(rewrite_tab, "Rewrite Previous Section")
        
        # Tab 3: Customize System Prompt
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NarrativeGUI()
    window.show()
    sys.exit(app.exec())
