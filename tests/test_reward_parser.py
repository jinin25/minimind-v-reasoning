import unittest
from trainer.train_grpo_vlm import LocalRewarder


class RewardParserTest(unittest.TestCase):
    def score(self, body, answer, answer_type):
        return LocalRewarder._answer_reward(f"<think>x</think><answer>{body}</answer>", answer, answer_type)

    def test_multiple_choice(self):
        self.assertEqual(self.score("Option (D)", ["D"], "multiple-choice"), 1.0)
        self.assertEqual(self.score("The answer is B.", ["B"], "multiple-choice"), 1.0)
        self.assertEqual(self.score("A then C", ["C"], "multiple-choice"), 1.0)
        self.assertEqual(self.score("Option E", ["E"], "multiple-choice"), 1.0)
        self.assertEqual(self.score("B", ["D"], "multiple-choice"), 0.0)

    def test_number(self):
        self.assertEqual(self.score("Therefore, 1,250", ["1250"], "number"), 1.0)
        self.assertEqual(self.score("first 2, finally 7", ["7"], "number"), 1.0)
        self.assertEqual(self.score("7", ["8"], "number"), 0.0)

    def test_ocr(self):
        self.assertEqual(self.score("General, Assembly!", ["general assembly"], "ocrtext"), 1.0)
        self.assertEqual(self.score("Ａ B-C", ["a b c"], "ocrtext"), 1.0)
        self.assertEqual(self.score("hello", ["world"], "ocrtext"), 0.0)

    def test_invalid_formats(self):
        self.assertEqual(LocalRewarder._answer_reward("D", ["D"], "multiple-choice"), 0.0)
        self.assertEqual(LocalRewarder._answer_reward("<answer>A</answer><answer>D</answer>", ["D"], "multiple-choice"), 0.0)
        self.assertEqual(self.score("[0,0,1,1]", ["0,0,1,1"], "bbox"), 0.0)


if __name__ == "__main__":
    unittest.main()
