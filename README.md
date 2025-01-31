Source: [Chatbot-with-Conversational-Memory-on-LangChain](https://replit.com/@GroqCloud/Chatbot-with-Conversational-Memory-on-LangChain)

#Groq LangChain Conversational Chatbot

A simple application that allows users to interact with a conversational chatbot powered by LangChain. The application uses the Groq API to generate responses and leverages LangChain's [ConversationBufferWindowMemory](https://python.langchain.com/v0.1/docs/modules/memory/types/buffer_window/) to maintain a history of the conversation to provide context for the chatbot's responses.
Features

- Conversational Interface: The application provides a conversational interface where users can ask questions or make statements, and the chatbot responds accordingly.

- Contextual Responses: The application maintains a history of the conversation, which is used to provide context for the chatbot's responses.

- LangChain Integration: The chatbot is powered by the LangChain API, which uses advanced natural language processing techniques to generate human-like responses.

Usage

You will need to store a valid Groq API Key as a secret to proceed with this example. You can generate one for free [here](https://console.groq.com/keys).

You can [fork and run this application on Replit](https://replit.com/@GroqCloud/Chatbot-with-Conversational-Memory-on-LangChain) or run it on the command line with python main.py


### For Windows Users:
1. Right-click on **This PC** (or **Computer**) and select **Properties**.
2. Go to **Advanced system settings**.
3. Click on **Environment Variables**.
4. Under **System variables**, click **New**.
5. In the **New System Variable** window:
   - For **Variable name**, enter `GROQ_API_KEY`.
   - For **Variable value**, enter the user's API key.
6. Click **OK** to save the changes.

### Using a `.env` File (Alternative Method):
If the user prefers, they can use a `.env` file in their project directory to store the API key. Here's how:
1. Create a `.env` file in the root of the project.
2. Add the following line to the `.env` file:
   ```
   GROQ_API_KEY='your_api_key_here'
   ```
3. In the Python script, use a library like `python-dotenv` to load the environment variables from the `.env` file:
   ```python
   from dotenv import load_dotenv
   import os

   load_dotenv()
   groq_api_key = os.environ['GROQ_API_KEY']
   ```

### Important Notes:
- The user should never share their API key publicly or hardcode it into their scripts.
- If this is part of a larger application, consider providing instructions for different operating systems and environments (e.g., development, production).

## main.py Script Documentation

The `main.py` script is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.

### How to Run

1. Ensure you have set up your Groq API key as described in the Usage section.
2. Run the script using Python:
   ```sh
   python main.py
   ```

### Script Details

- **Groq API Key**: The script retrieves the Groq API key from the environment variables.
- **Model**: The script uses the `deepseek-r1-distill-llama-70b` model.
- **Chat Initialization**: The script initializes the Groq LangChain chat object and sets up the conversation.
- **System Prompt**: The chatbot uses a system prompt to define its behavior as a friendly conversational chatbot.
- **Conversational Memory**: The chatbot maintains a history of the last 5 messages to provide context for its responses.
- **User Interaction**: The script enters a loop where it continuously prompts the user for input and generates responses using the Groq API.

### Example Usage

```sh
python main.py
```

When you run the script, you will be greeted by the chatbot and can start asking questions or making statements. The chatbot will respond accordingly, leveraging the context of the conversation to provide relevant answers.

### Important Notes

- Do not share your API key publicly or hardcode it into your scripts.
- Consider providing instructions for different operating systems and environments if this is part of a larger application.

## Setting Up Dependencies

The application requires several Python packages to run. These dependencies are listed in the `requirements.txt` file.

### Installing Dependencies

To install the required packages, use the following command:

```sh
pip install -r requirements.txt
```

This command will install all the packages listed in the `requirements.txt` file, ensuring that your environment is set up correctly to run the application.

### requirements.txt File

The `requirements.txt` file includes the following dependencies:

- `groq`
- `langchain==0.1.16`
- `langchain-core`
- `langchain-groq`

Make sure to install these dependencies before running the `main.py` script to avoid any import errors or missing package issues.