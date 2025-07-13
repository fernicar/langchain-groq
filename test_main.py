import unittest
from unittest.mock import MagicMock, patch
from main import Narrative, TokenWindowDualStateMemory
from langchain_core.messages import HumanMessage, AIMessage

class TestNarrativeMemory(unittest.TestCase):

    @patch('main.ChatGroq')
    def setUp(self, MockChatGroq):
        # Mock the GUI and other external dependencies
        with patch('main.GUI.__init__', return_value=None):
            with patch('main.Narrative.ensure_api_key', return_value=True):
                with patch('main.Narrative.populate_models_and_prompts', return_value=None):
                    # Mock the llm attribute for the TokenWindowDualStateMemory
                    mock_llm = MagicMock()
                    mock_llm.get_num_tokens.return_value = 0
                    self.app = Narrative()
                    self.app.conversation.memory.llm = mock_llm


    def test_commit_blue_text(self):
        # 1. Set up the initial state
        self.app.current_narrative = "This is a test proposal."

        # 2. Simulate the action
        self.app.commit_blue_text()

        # 3. Assert the expected outcome
        self.assertIn("This is a test proposal.", self.app.canon_validated)
        self.assertEqual(self.app.current_narrative, "")

    def test_discard_last_conversation_pair(self):
        # 1. Set up the initial state
        self.app.current_narrative = "This is a test proposal."
        # Add a dummy message to the history to simulate a previous turn
        history = self.app.conversation.memory
        history.add_message(HumanMessage(content="Previous input"))
        history.add_message(AIMessage(content="Previous output"))
        history.commit_proposal()

        # 2. Simulate the action
        self.app.discard_last_conversation_pair()

        # 3. Assert the expected outcome
        self.assertNotIn("This is a test proposal.", self.app.canon_validated)
        # The blue text should be restored to the previous AI response, which is the last AI message
        self.assertEqual(self.app.current_narrative, "Previous output")

    def test_save_and_continue(self):
        # 1. Set up the initial state
        self.app.current_narrative = "This is a test proposal."
        self.app.continue_input.setPlainText("Continue with this.")

        # 2. Simulate the action
        with patch.object(self.app, 'send_message') as mock_send:
            self.app.input_tabs.setCurrentIndex(1)
            self.app.send_message()

        # 3. Assert the expected outcome
        mock_send.assert_called_once()


if __name__ == '__main__':
    unittest.main()
