![preview](https://github.com/fernicar/langchain-groq/blob/main/images/preview.jpg)

# Narrative Collaboration System

A Qt-based desktop application for collaborative story writing with AI assistance using the Groq API and LangChain framework.

## Core Workflow

The application manages story content in two states:
- **Black Text**: Permanent, validated content that has been approved and saved
- **Blue Text**: Current AI proposal or work in progress that hasn't been committed

### Main Interaction Modes

1. **Edit Blue**
   - Direct editing interface for modifying the current blue proposal
   - Make precise adjustments to AI-generated text
   - Fix specific words or phrases before committing
   - Changes reflect instantly in the preview panel

2. **Save Blue & Continue**
   - Two actions in one:
     1. Converts current blue proposal to permanent black text
     2. Generates new blue proposal based on your guidance
   - Can provide specific instructions for the next section
   - Leave guidance empty to let AI continue based on context
   - Supports XML tags for structured instructions

3. **Discard Blue & Rewrite**
   - Completely removes current blue proposal from history
   - Generates new alternative content based on your instructions
   - Previous black (committed) text remains unchanged
   - Useful when the current proposal needs complete replacement
   - As if the discarded proposal never existed

### Story Development Process

1. **Starting Point**
   - Begin with either your own text or let AI generate initial content
   - All new content appears in blue for review

2. **Review and Iteration**
   - Review blue proposals
   - Choose one of three actions:
     - Edit: Fine-tune the current blue text
     - Save & Continue: Commit and progress
     - Discard & Rewrite: Start fresh with new instructions

3. **Content Management**
   - Committed (black) text is saved to file
   - Blue proposals can be edited until committed
   - Clear separation between approved and proposed content

### Story Management
- File operations (Open, Save, Save As)
- Automatic story chunking for context management
- Unsaved changes protection
- Story preview with color-coded sections:
  - Black: Committed/saved content
  - Blue: Current proposal/work in progress

### Technical Features
- LangChain integration with Groq API
- Multiple AI model support:
  - qwen-qwq-32b
  - deepseek-r1-distill-qwen-32b
  - deepseek-r1-distill-llama-70b
  - mixtral-8x7b-32768
  - llama-3.3-70b-versatile
- Conversation buffer window memory (last 5 interactions)
- XML tag support for structured input
- JSON-based system prompt configuration
- UTF-8 text encoding support
- Spanish language interface and AI responses

## Advanced Workflow

### Story Continuation
The application implements a sophisticated context management system:

1. **Loading Existing Stories**
   - Load your existing story file (Ctrl+O)
   - The system automatically processes the text into meaningful chunks
   - The last 5 chunks are used to simulate a conversation history with the AI
   - This simulated history provides context as if the story had been written in one session

2. **Context Management**
   - The application maintains a rolling window of the last 5 interactions
   - When loading a story, the system creates artificial conversation turns:
     - Each chunk is treated as if it was a "Continue the story" request
     - The AI's responses are simulated using the actual story content
   - This creates seamless continuation capability for:
     - Stories written in previous sessions
     - Stories written outside the application
     - Collaborative works between multiple sessions

3. **Session Workflow**
   - Start by loading your story file
   - The system automatically reconstructs the conversation context
   - Continue writing as if you never left the session
   - All features (Edit Blue, Save & Continue, Rewrite) work seamlessly
   - The AI maintains narrative consistency with previous content

### Basic Workflow
For new stories:
   - Start with a blank slate
   - Use Edit Blue tab to write initial content
   - Or let AI generate the starting point
   - Progress using Save Blue & Continue
   - Use Discard Blue & Rewrite for revisions
   - Save regularly (Ctrl+S)

## Requirements

- Python 3.x
- PySide6 (Qt for Python)
- LangChain and LangChain Groq
- Groq API access
- python-dotenv
- Additional dependencies:
  - typing
  - json
  - datetime
  - pathlib
  - re
  - argparse

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install PySide6 langchain langchain-groq python-dotenv
```
3. Create a `.env` file with your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Basic workflow:
   - Start a new story or load existing one (Ctrl+O)
   - Use the Edit Blue tab to write or modify content
   - Use Save Blue & Continue to progress the story
   - Use Discard Blue & Rewrite for major revisions
   - Save your work regularly (Ctrl+S)
   - Select your preferred AI model from the dropdown menu

## Keyboard Shortcuts
- Ctrl+O: Load Story
- Ctrl+S: Save Story
- Ctrl+Shift+S: Save Story As

## File Format
- Saves stories as plain text (.txt) files
- Uses UTF-8 encoding
- Automatically chunks content for optimal AI context

## Configuration
- System prompts are stored in `system_prompts.json`
- Default system prompt in Spanish
- Supports custom prompt creation and management
- Configurable conversation memory window (default: 5 messages)

## Notes
- Unsaved blue proposals will trigger a save prompt when closing
- The application maintains conversation context for consistent story flow
- Large stories are automatically chunked for better AI context management
- Interface and AI responses are in Spanish by default
