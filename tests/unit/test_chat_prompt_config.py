import unittest
from unittest.mock import patch

from app.chat import presenter
from app.query_parser import PROMPTS_DIR
from app.settings import AGENT_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT, PRESENT_SYSTEM_PROMPT


class ChatPromptConfigTests(unittest.TestCase):
    def test_configured_prompts_exist_and_are_not_empty(self):
        for name in (AGENT_SYSTEM_PROMPT, PRESENT_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT):
            with self.subTest(name=name):
                self.assertTrue((PROMPTS_DIR / name).read_text(encoding='utf-8').strip())

    def test_default_selection_is_ver2_for_chat_and_unchanged_for_judgment(self):
        self.assertEqual(AGENT_SYSTEM_PROMPT, 'chat/agent_system_ver2.txt')
        self.assertEqual(PRESENT_SYSTEM_PROMPT, 'chat/present_system_ver2.txt')
        self.assertEqual(JUDGE_SYSTEM_PROMPT, 'system.txt')

    def test_ver1_files_are_preserved_for_rollback(self):
        for name in ('chat/agent_system_ver1.txt', 'chat/present_system_ver1.txt'):
            with self.subTest(name=name):
                self.assertTrue((PROMPTS_DIR / name).read_text(encoding='utf-8').strip())

    def test_present_reads_configured_prompt(self):
        with patch.object(presenter, 'PRESENT_SYSTEM_PROMPT', 'chat/present_system_ver1.txt'):
            system, _ = presenter.build_prompt('가위 돼?', {}, [], [])
        self.assertEqual(system, (PROMPTS_DIR / 'chat/present_system_ver1.txt').read_text(encoding='utf-8').strip())
