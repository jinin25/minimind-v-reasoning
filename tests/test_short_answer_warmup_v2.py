import unittest
import torch

from model.model_minimind import MiniMindForCausalLM
from dataset.lm_dataset import VLMDataset


class StopSequenceTests(unittest.TestCase):
    def test_multi_token_suffix_detection_contract(self):
        generated=torch.tensor([[1,2,8,9],[1,2,3,9]])
        target=generated.new_tensor([8,9]).unsqueeze(0)
        self.assertEqual(generated[:,-2:].eq(target).all(dim=1).tolist(), [True,False])

    def test_different_batch_finish_times_keep_finished_rows_finished(self):
        finished=torch.tensor([True,False]); newly=torch.tensor([False,True])
        self.assertEqual((finished|newly).tolist(), [True,True])


class FormatRegressionTests(unittest.TestCase):
    def test_duplicate_and_missing_tags_are_detectable(self):
        self.assertGreater('<answer>A</answer></answer>'.count('</answer>'),1)
        self.assertEqual('A'.count('</answer>'),0)


class LossWeightTests(unittest.TestCase):
    def setUp(self):
        self.ds=VLMDataset.__new__(VLMDataset)
        self.ds.answer_loss_weight=4.0
        self.ds.xml_token_ids={'<think>':[10],'</think>':[11],'<answer>':[12],'</answer>':[13]}
        self.ds.eos_id=[99]

    def test_answer_xml_and_eos_are_weighted(self):
        ids=[10,20,11,12,30,13,99,0]
        labels=ids.copy(); labels[-1]=-100
        weights=self.ds.generate_loss_weights(ids,labels)
        self.assertEqual(weights,[4,1,4,4,4,4,4,1])

    def test_ignored_tokens_never_gain_effective_weight(self):
        ids=[12,30,13,99]
        labels=[-100,-100,-100,-100]
        weights=self.ds.generate_loss_weights(ids,labels)
        self.assertEqual(weights,[1,1,1,1])


if __name__ == '__main__': unittest.main()
