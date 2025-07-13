import unittest
from unittest.mock import MagicMock, patch
from main import Narrative
from langchain_core.messages import HumanMessage, AIMessage

class TestNarrativeMemory(unittest.TestCase):

    @patch('main.ChatGroq')
    def setUp(self, MockChatGroq):
        # Mock the GUI and other external dependencies
        with patch('main.GUI.__init__', return_value=None):
            with patch('main.Narrative.ensure_api_key', return_value=True):
                with patch('main.Narrative.populate_models_and_prompts', return_value=None):
                    self.app = Narrative()

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
        history.save_context({"input": "Previous input"}, {"output": "Previous output"})

        # 2. Simulate the action
        self.app.discard_last_conversation_pair()

        # 3. Assert the expected outcome
        self.assertNotIn("This is a test proposal.", self.app.canon_validated)
        # The blue text should be restored to the previous AI response, which is the last AI message
        self.assertEqual(self.app.current_narrative, "Previous output")

if __name__ == '__main__':
    unittest.main()
