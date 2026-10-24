import unittest
from src.scoring import attribute_queries,clauses,head_from,prediction_sets


class Protocol(unittest.TestCase):
    def test_clauses(self):
        self.assertEqual(clauses("vivid textures and warm light"),["vivid textures","warm light"])
        self.assertEqual(set(k for k,_ in attribute_queries("two red cups left of a blue bowl")),{"object","count","color","position"})

    def test_preference_swaps(self):
        row=dict(split="train",winner=1,features=[[.1,.1,.1,0],[.3,.3,.3,0]])
        h=head_from([row])
        self.assertEqual(prediction_sets(row,h)["preference_head"],1)
        self.assertEqual(prediction_sets(row|{"features":row["features"][::-1]},h)["preference_head"],0)
