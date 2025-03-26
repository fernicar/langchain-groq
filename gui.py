# --- START OF FILE gui.py ---
import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
 QHBoxLayout, QTextEdit, QLabel, QPushButton, QCheckBox, QFrame, QTabWidget,
 QSpinBox, QSplitter, QComboBox, QSizePolicy, QToolBar, QLineEdit, QFileDialog,
 QMessageBox, QDoubleSpinBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor, QPalette, QAction

class GUI(QMainWindow):
  def __init__(self):
    super().__init__()

    # Add this line after super().__init__()
    self.send_shortcut = QAction("Send", self)
    self.send_shortcut.setShortcut("Ctrl+Return")  # Ctrl+Enter/Return
    self.send_shortcut.triggered.connect(self.send_message)
    self.addAction(self.send_shortcut)

    # Add file path tracking
    self.current_file_path = None

    # Initialize other attributes
    self.current_narrative = ""
    self.canon_validated = []
    self.font_size = 10
    self.is_dark_mode = True
    self.colors = {
      'dark': {
        'bg': QColor('#2b2b2b'),
        'fg': QColor('#ffffff'),
        'canon': QColor('#a9a9a9'),
        'current': QColor('#6495ed'),
        'xml': QColor('#98fb98')
      },
      'light': {
        'bg': QColor('#ffffff'),
        'fg': QColor('#000000'),
        'canon': QColor('#696969'),
        'current': QColor('#4169e1'),
        'xml': QColor('#228b22')
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
    self.right_tab_widget = QTabWidget()  # Store as class attribute
    self.right_tab_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # context Monitor tab
    context_tab = QWidget()
    context_layout = QVBoxLayout(context_tab)
    context_layout.setContentsMargins(0, 0, 0, 0)
    
    self.context_display = QTextEdit()
    self.context_display.setReadOnly(True)
    self.context_display.setPlaceholderText(
        "Context Monitor:\n\n"
        "• Shows last 5 conversation pairs used for context\n"
        "• Helps track what information the AI remembers\n"
        "• Updates automatically as you interact\n"
        "• Pairs: 0/5"
    )
    context_layout.addWidget(self.context_display)

    # Add Commit Blue button
    self.commit_blue_button = QPushButton("Commit Blue now")
    self.commit_blue_button.clicked.connect(self.commit_blue_text)
    context_layout.addWidget(self.commit_blue_button)

    # Add the tab with initial text
    self.right_tab_widget.addTab(context_tab, "Context 0/5")

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
    self.right_tab_widget.addTab(thinking_tab, "Thinking Process")

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
    self.right_tab_widget.addTab(conversation_tab, "Conversation Logs")

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

    self.right_tab_widget.addTab(api_tab, "API Monitor")

    right_layout.addWidget(self.right_tab_widget)

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
    
    # Create a list of widget types to check
    widget_types = {
        QTextEdit: "Text Edit",
        QComboBox: "Combo Box",
        QLabel: "Label",
        QPushButton: "Button",
        QTabWidget: "Tab Widget",
        QLineEdit: "Line Edit"
    }

    # Check each widget type
    for widget_type, name in widget_types.items():
        widgets = self.findChildren(widget_type)
        for widget in widgets:
            widget.setFont(font)
            # Preserve the blue color for edit_input while clearing other styles
            if widget == self.edit_input:
                theme = 'dark' if self.is_dark_mode else 'light'
                widget.setStyleSheet(f"""
                    QTextEdit {{
                        color: {self.colors[theme]['current'].name()};
                        background-color: {self.colors[theme]['bg'].name()};
                        border: 1px solid {self.colors[theme]['fg'].name()};
                    }}
                """)
            elif isinstance(widget, (QTextEdit, QLineEdit)):
                widget.setStyleSheet("")

    # Special handling for tab bars
    for tab_widget in self.findChildren(QTabWidget):
        tab_bar = tab_widget.tabBar()
        tab_bar.setFont(font)

    # Force update of specific displays
    self.story_display.document().setDefaultFont(font)
    self.thinking_display.document().setDefaultFont(font)
    self.conversation_log.document().setDefaultFont(font)
    self.api_monitor.document().setDefaultFont(font)

    # Force a refresh of the story display
    self.update_story_display()

  def toggle_theme(self, checked):
    self.is_dark_mode = checked
    theme = 'dark' if checked else 'light'

    # Set application palette
    palette = self.palette()
    palette.setColor(self.backgroundRole(), self.colors[theme]['bg'].name())
    palette.setColor(self.foregroundRole(), self.colors[theme]['fg'].name())
    palette.setColor(QPalette.ColorRole.Window, self.colors[theme]['bg'].name())
    palette.setColor(QPalette.ColorRole.WindowText, self.colors[theme]['fg'].name())
    palette.setColor(QPalette.ColorRole.Base, self.colors[theme]['bg'].name())
    palette.setColor(QPalette.ColorRole.Text, self.colors[theme]['fg'].name())
    palette.setColor(QPalette.ColorRole.Button, self.colors[theme]['bg'].name())
    palette.setColor(QPalette.ColorRole.ButtonText, self.colors[theme]['fg'].name())
    self.setPalette(palette)

    # Comprehensive style sheet for all widgets
    style_sheet = f"""
      QMainWindow {{
        background-color: {self.colors[theme]['bg'].name()};
        color: {self.colors[theme]['fg'].name()};
      }}
      QMenuBar {{
        background-color: {self.colors[theme]['bg'].name()};
        color: {self.colors[theme]['fg'].name()};
      }}
      QMenuBar::item {{
        background-color: {self.colors[theme]['bg'].name()};
        color: {self.colors[theme]['fg'].name()};
      }}
      QMenuBar::item:selected {{
        background-color: {self.colors[theme]['fg'].name()};
        color: {self.colors[theme]['bg'].name()};
      }}
      QWidget {{
        background-color: {self.colors[theme]['bg'].name()};
        color: {self.colors[theme]['fg'].name()};
      }}
      QTextEdit {{
        background-color: {self.colors[theme]['bg'].name()};
        color: {self.colors[theme]['fg'].name()};
        border: 1px solid {self.colors[theme]['fg'].name()};
      }}
      QTabWidget::pane {{
        border: 1px solid {self.colors[theme]['fg'].name()};
        background-color: {self.colors[theme]['bg'].name()};
      }}
      QTabBar::tab {{
        background-color: {self.colors[theme]['bg'].name()};
        color: {self.colors[theme]['fg'].name()};
        padding: 8px;
        border: 1px solid {self.colors[theme]['fg'].name()};
        margin-right: 2px;
      }}
      QTabBar::tab:selected {{
        background-color: {self.colors[theme]['fg'].name()};
        color: {self.colors[theme]['bg'].name()};
      }}
      QTitleBar {{
        background-color: {self.colors[theme]['bg'].name()};
        color: {self.colors[theme]['fg'].name()};
      }}
      QPushButton {{
        background-color: {self.colors[theme]['bg'].name()};
        color: {self.colors[theme]['fg'].name()};
        border: 1px solid {self.colors[theme]['fg'].name()};
        padding: 5px;
        min-width: 80px;
      }}
      QPushButton:hover {{
        background-color: {self.colors[theme]['fg'].name()};
        color: {self.colors[theme]['bg'].name()};
      }}
      QLabel {{
        color: {self.colors[theme]['fg'].name()};
        background-color: transparent;
      }}
      QCheckBox {{
        color: {self.colors[theme]['fg'].name()};
        background-color: transparent;
      }}
      QSpinBox {{
        background-color: {self.colors[theme]['bg'].name()};
        color: {self.colors[theme]['fg'].name()};
        border: 1px solid {self.colors[theme]['fg'].name()};
      }}
      QFrame {{
        background-color: {self.colors[theme]['bg'].name()};
        color: {self.colors[theme]['fg'].name()};
      }}
    """

    # Apply style sheet to the main window
    self.setStyleSheet(style_sheet)

    # Force update of story display to refresh colors
    self.update_story_display()

    # Update Edit Blue input color
    self.edit_input.setStyleSheet(f"""
      QTextEdit {{
        color: {self.colors[theme]['current'].name()};
        background-color: {self.colors[theme]['bg'].name()};
        border: 1px solid {self.colors[theme]['fg'].name()};
      }}
    """)

  def on_tab_changed(self, index: int):
    """Handle tab changes"""
    # You can add specific logic here if needed when tabs change
    pass

  def update_story_display(self):
    """Update the story display with proper formatting"""
    self.story_display.clear()
    cursor = self.story_display.textCursor()

    # Show load prompt if no content
    if not self.canon_validated and not self.current_narrative:
      format = self.story_display.currentCharFormat()
      theme = 'dark' if self.is_dark_mode else 'light'
      format.setForeground(self.colors[theme]['fg'])
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
      format.setForeground(self.colors[theme]['canon'])
      cursor.setCharFormat(format)
      cursor.insertText(piece + "\n\n")

    # Insert current narrative in blue
    if self.current_narrative:
      format = self.story_display.currentCharFormat()
      theme = 'dark' if self.is_dark_mode else 'light'
      format.setForeground(self.colors[theme]['current'])
      cursor.setCharFormat(format)
      cursor.insertText(self.current_narrative + "\n")

    self.story_display.setTextCursor(cursor)

  def wrap_text_with_xml(self, text: str, tag: str) -> str:
    """Wrap text with XML tags if a tag is specified"""
    if not tag.strip():
      return text
    return f"<{tag}>{text}</{tag}>"

  def setup_toolbar(self):
    """Setup the application toolbar"""
    toolbar = QToolBar()

    # Model selector
    model_label = QLabel("Model: ")
    toolbar.addWidget(model_label)

    self.model_selector = QComboBox()
    # Placeholder for models - will be populated in main.py
    self.model_selector.addItem("Loading Models...") # Initial placeholder
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
    toolbar.addWidget(self.temperature_spinner)

    # Add separator
    toolbar.addSeparator()

    # Max tokens control
    tokens_label = QLabel("Max Tokens: ")
    toolbar.addWidget(tokens_label)

    self.max_tokens_spinner = QSpinBox()
    self.max_tokens_spinner.setRange(1, 32768)
    self.max_tokens_spinner.setValue(self.max_tokens)
    toolbar.addWidget(self.max_tokens_spinner)

    # Add separator
    toolbar.addSeparator()

    # XML Tag wrapper
    xml_label = QLabel("xml tag:")
    toolbar.addWidget(xml_label)
    self.xml_tag_input = QLineEdit()
    self.xml_tag_input.setPlaceholderText("custom tag")
    self.xml_tag_input.setMaximumWidth(100)
    toolbar.addWidget(self.xml_tag_input)

    # Add separator
    toolbar.addSeparator()
    # Font size controls
    font_label = QLabel("Font Size:")
    toolbar.addWidget(font_label)
    self.font_size_spinner = QSpinBox()
    self.font_size_spinner.setRange(8, 24)
    self.font_size_spinner.setValue(self.font_size)
    self.font_size_spinner.valueChanged.connect(self.update_font_size)
    toolbar.addWidget(self.font_size_spinner)

    # Add separator
    toolbar.addSeparator()

    # Dark mode toggle
    self.dark_mode_toggle = QCheckBox("Dark Mode")
    self.dark_mode_toggle.setChecked(True) # Set to True for default dark mode
    self.dark_mode_toggle.stateChanged.connect(self.toggle_theme)
    toolbar.addWidget(self.dark_mode_toggle)

    # Add separator
    toolbar.addSeparator()

    # Prompt selector
    prompt_label = QLabel("System Prompt: ")
    toolbar.addWidget(prompt_label)

    self.prompt_selector = QComboBox()
    # Placeholder for prompts - will be populated in main.py
    self.prompt_selector.addItem("Loading Prompts...") # Initial placeholder
    toolbar.addWidget(self.prompt_selector)

    # Add separator
    toolbar.addSeparator()

    # Send button
    self.send_button = QPushButton("Send")
    toolbar.addWidget(self.send_button)

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
        color: {self.colors[theme]['current'].name()};
        background-color: {self.colors[theme]['bg'].name()};
        border: 1px solid {self.colors[theme]['fg'].name()};
      }}
    """)
    self.edit_input.textChanged.connect(self.update_blue_preview)
    edit_layout.addWidget(self.edit_input)
    
    # Create button layout for Edit Blue tab
    edit_button_layout = QHBoxLayout()
    
    # Add "Commit Blue" button
    self.commit_blue_button = QPushButton("Commit Blue Now")
    self.commit_blue_button.clicked.connect(self.commit_blue_text)
    edit_button_layout.addWidget(self.commit_blue_button)
    
    # Add "Discard Blue Now" button
    self.discard_blue_button = QPushButton("Discard Blue Now")
    self.discard_blue_button.clicked.connect(self.discard_last_conversation_pair)
    edit_button_layout.addWidget(self.discard_blue_button)
    
    # Add button layout to Edit Blue tab
    edit_layout.addLayout(edit_button_layout)
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
    # Placeholder for system prompt - will be populated in main.py
    self.system_input.setPlaceholderText("Load System Prompt...")
    self.prompt_name_input.setPlaceholderText("Prompt Name")
    system_layout.addWidget(self.system_input)

    # Add buttons for save and delete
    button_layout = QHBoxLayout()
    self.save_prompt_button = QPushButton("Save Prompt")
    self.delete_prompt_button = QPushButton("Delete Prompt")
    button_layout.addWidget(self.save_prompt_button)
    button_layout.addWidget(self.delete_prompt_button)
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

    # Initialize actions as instance attributes
    self.clear_action = QAction('New Story', self)
    self.clear_action.setShortcut('Ctrl+N')
    file_menu.addAction(self.clear_action)

    self.load_action = QAction('Load Story...', self)
    self.load_action.setShortcut('Ctrl+O')
    file_menu.addAction(self.load_action)

    self.save_action = QAction('Save Story', self)
    self.save_action.setShortcut('Ctrl+S')
    file_menu.addAction(self.save_action)

    self.save_as_action = QAction('Save Story As...', self)
    self.save_as_action.setShortcut('Ctrl+Shift+S')
    file_menu.addAction(self.save_as_action)

    # Add separator
    file_menu.addSeparator()

    # Exit action
    self.exit_action = QAction('Exit', self)
    self.exit_action.setShortcut('Ctrl+Q')
    file_menu.addAction(self.exit_action)


  def check_unsaved_changes(self) -> bool:
    """Check for unsaved changes and ask user what to do
    Returns True if operation should continue, False if cancelled"""
    if self.current_narrative:
      reply = QMessageBox.question(
        self,
        'Unsaved Changes',
        'There is an unsaved blue proposal. Do you want to save it before continue?',
        QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
        QMessageBox.Save
      )

      if reply == QMessageBox.Save:
        # Add current narrative to canon and save
        self.canon_validated.append(self.current_narrative)
        self.save_story()
        return True
      elif reply == QMessageBox.Discard:
        return True
      else: # Cancel
        return False
    return True

  # Placeholder methods - Implement in main.py with application logic
  def send_message(self):
    pass
  def on_model_changed(self):
    pass
  def on_temperature_changed(self, value):
    pass
  def on_max_tokens_changed(self, value):
    pass
  def on_prompt_selected(self, prompt_name: str):
    pass
  def save_system_prompt(self):
    pass
  def delete_system_prompt(self):
    pass
  def load_story(self):
    """Placeholder - implement in main.py"""
    if self.check_unsaved_changes():
        pass  # Main implementation will go here
  def save_story(self, save_as=False):
    pass

  # --- GUI interaction methods ---
  def get_open_file_name(self, caption, filter_str):
    """Wrapper for QFileDialog.getOpenFileName"""
    file_path, _ = QFileDialog.getOpenFileName(self, caption, "", filter_str)
    return file_path

  def get_save_file_name(self, caption, dir_str, filter_str):
    """Wrapper for QFileDialog.getSaveFileName"""
    file_path, _ = QFileDialog.getSaveFileName(self, caption, dir_str, filter_str)
    return file_path

  def show_error_message(self, title, message):
    """Wrapper for QMessageBox.critical"""
    QMessageBox.critical(self, title, message)

  def show_question_message(self, title, message):
    """Wrapper for QMessageBox.question, returns True if Yes, False if No"""
    reply = QMessageBox.question(
      self,
      title,
      message,
      QMessageBox.Yes | QMessageBox.No,
      QMessageBox.No
    )
    return reply == QMessageBox.Yes
