import unittest
from src.scoring import clauses,head_from,prediction_sets


class Protocol(unittest.TestCase):
    def test_clauses(self):
        self.assertEqual(clauses("red cup and blue pot"),["red cup","blue pot"])

    def test_preference_swaps(self):
        row=dict(split="train",winner=1,features=[[.1,.1,.1,0],[.3,.3,.3,0]])
        h=head_from([row])
        self.assertEqual(prediction_sets(row,h)["preference_head"],1)
        self.assertEqual(prediction_sets(row|{"features":row["features"][::-1]},h)["preference_head"],0)
