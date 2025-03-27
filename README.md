![preview](https://github.com/fernicar/langchain-groq/blob/main/images/preview.jpg)

# Narrative Collaboration System

A desktop application designed for writers who want to maintain creative control while leveraging AI assistance in their writing process. Unlike traditional online chatbots, this system provides a structured, distraction-free environment specifically tailored for story development.

## Why This Project?

### Problems with Traditional AI Chat Interfaces
- **Context Loss**: Chat interfaces often lose track of longer conversations, making it difficult to maintain story coherence
- **Content Management**: No clear distinction between work-in-progress and finalized content, leading to confusion
- **Limited Creative Control**: Writers often find themselves following the AI's lead rather than directing the story
- **Workflow Disruption**: Constant copying and pasting between chat and writing tools breaks creative flow
- **Content Overload**: Easy to accumulate massive amounts of AI-generated text without proper review structure
- **Local Management**: Lack of proper tools for organizing and managing story content offline
- **Review Process**: No structured way to review, edit, and approve AI suggestions, leading to content backlogs

### User-Friendly Solution
This application provides:
- A dedicated writing environment with clear visual feedback
- Local file management for your stories
- Structured workflow for story development
- Full control over AI involvement
- Clear separation between drafts (blue) and approved content (black/white)
- Dark and light themes for comfortable reading

## Key Features

### Unique Workflow
- **Two-State Content Management**
  - Black/White Text: Your approved, permanent content
  - Blue Text: AI proposals or work in progress
  - Clear visual distinction between draft and final content
  - No copying and pasting between tools
  - Efficient interaction model with minimal clicks
  - Keyboard shortcuts for rapid workflow
  - XML tag support for structured instructions

### Three Interaction Modes
1. **Edit Blue** Enforces story coherence manually 
   - Direct editing of AI proposals
   - Immediate visual feedback
   - Full creative control
   - Persistent edits across tab switches

2. **Save Blue & Continue** When story coherence is maintained
   - Save current blue text as permanent content
   - Provide optional guidance for next section
   - Let AI continue based on context alone
   - Automatic conversation history management

3. **Discard Blue & Rewrite** Useful for major revisions
   - Automatic discard of blue text
   - Reroll without discarded blue text influence
   - Request alternative proposals
   - Provide specific rewriting instructions

### Advanced Context Management
- Maintains story coherence across sessions
- Automatic content chunking for long stories
- Rolling context window of recent content
- Seamless continuation of previous work

## Getting Started

### System Requirements
- Operating System: Windows/Mac/Linux
- Python 3.x
- 4GB RAM minimum
- Internet connection (for AI interactions only)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/fernicar/langchain-groq
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API key:
   - Create a `.env` file in the project root
   - Add your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

### Quick Start
1. Launch the application:
```bash
python main.py
```

2. Choose your preferred AI model from the dropdown
3. Start writing or load an existing story (Ctrl+O)
4. Use the three tabs below to interact with your content

## Usage Tips

### Best Practices
- Save regularly (Ctrl+S)
- Use XML tags for structured instructions
- Keep context window in mind (5 interactions)
- Review AI proposals before committing

### Keyboard Shortcuts
- Ctrl+O: Load Story
- Ctrl+S: Save Story
- Ctrl+Shift+S: Save Story As

## Limitations
- Maximum context window of 5 interactions
- English-only interface
- Requires Groq API access
- No cloud synchronization (local files only)
- Single story session at a time

## Technical Details

### AI Configuration
- Adjustable temperature for creativity control
- Configurable maximum tokens
- Multiple model support:
  - qwen-qwq-32b
  - deepseek-r1-distill-qwen-32b
  - deepseek-r1-distill-llama-70b
  - mixtral-8x7b-32768
  - llama-3.3-70b-versatile

### File Management
- UTF-8 text encoding
- Plain text (.txt) file format
- Automatic story chunking
- Local file storage

### Customization
- System prompts via `system_prompts.json`
- Editable system prompt, Default Spanish to test.
- Configurable conversation memory
- XML tag support for structured input

## License
MIT License

## Acknowledgments
- Built with LangChain and Groq API
- UI powered by Qt/PySide6
- Special thanks to our contributors
