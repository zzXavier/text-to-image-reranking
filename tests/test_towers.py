"""Check that loading one frozen tower at a time gives monolithic PickScore arithmetic."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPProcessor,
                          CLIPTextConfig, CLIPTokenizer, CLIPVisionConfig)
from src.scoring import FrozenScorer,clauses


class FrozenTowerTests(unittest.TestCase):
    def test_tower_load_matches_joint_model(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory)
            (d/"vocab.json").write_text('{"<|startoftext|>":0,"<|endoftext|>":1}')
            (d/"merges.txt").write_text("#version: 0.2\n")
            tokenizer=CLIPTokenizer(str(d/"vocab.json"),str(d/"merges.txt"))
            text=CLIPTextConfig(vocab_size=len(tokenizer),hidden_size=16,intermediate_size=32,
                                num_hidden_layers=1,num_attention_heads=2,max_position_embeddings=77)
            vision=CLIPVisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                                    num_attention_heads=2,image_size=32,patch_size=8)
            model=CLIPModel(CLIPConfig.from_text_vision_configs(text,vision,projection_dim=8)).eval()
            proc=CLIPProcessor(tokenizer=tokenizer,image_processor=CLIPImageProcessor(
                size={"shortest_edge":32},crop_size={"height":32,"width":32}))
            model.save_pretrained(d)
            proc.save_pretrained(d)
            prompt="two red cups left of a blue bowl"
            imgs=[Image.new("RGB",(32,32),"red"),Image.new("RGB",(32,32),"white")]
            separate=FrozenScorer(d,"cpu",[prompt]).features(prompt,imgs)
            norm=torch.nn.functional.normalize
            with torch.inference_mode():
                t=norm(model.get_text_features(**proc(text=[prompt]+clauses(prompt),
                            return_tensors="pt",padding=True)),dim=-1)
                i=norm(model.get_image_features(**proc(images=imgs,return_tensors="pt")),dim=-1)
                raw=(i@t.T).numpy()
            joint=np.column_stack([raw[:,0],raw[:,1:].mean(1),raw[:,1:].min(1),raw[:,1:].std(1)])
            np.testing.assert_allclose(separate,joint,rtol=1e-5,atol=1e-6)
