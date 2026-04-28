import unittest
import os
import sys
import json
import tempfile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from sian.experiment import SyntheticSweepExperiment

#fuzzy tests giving some eps approximation on some dataset






class TestTrainingNoErrors(unittest.TestCase):
    def setUp(self):
        self.variants = [
            {"is_masked_mlp": False, "is_masked_sian": False, "use_mnist_scaling": "smooth"},
            {"is_masked_mlp": False, "is_masked_sian": False, "use_mnist_scaling": "mnist"},
            {"is_masked_mlp": True, "is_masked_sian": True, "use_mnist_scaling": "smooth"},
            {"is_masked_mlp": True, "is_masked_sian": True, "use_mnist_scaling": "mnist"},
        ]
        self.base_config = {
            "data_base_path": "data/",
            "dataset_str": "SYNTH_simple_discrete_synthwave_v9_D10_Ik5_s3.9.27_seed0",
            "preproc_owner": None,
            "FIS_style": "maximal",
            "MAX_K": 1,
            "BS": 32,
            "LR": 5e-3,
            "STEP": 10,
            "number_of_rounds": None,
            "inters_per_round": None
        }

    def test_training_variants(self):
        for variant in self.variants:
            with self.subTest(variant=variant):
                test_config = self.base_config.copy()
                test_config.update({
                    "is_masked_mlp": variant["is_masked_mlp"],
                    "is_masked_sian": variant["is_masked_sian"],
                    "use_mnist_scaling": variant["use_mnist_scaling"]
                })
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                    json.dump(test_config, tmp_file, indent=4)
                    tmp_file_path = tmp_file.name

                try:
                    exp = SyntheticSweepExperiment()
                    exp.run(test_config)
                    
                    self.assertEqual(test_config["is_masked_mlp"], variant["is_masked_mlp"],
                                   f"Mismatch in is_masked_mlp for variant {variant}")
                    self.assertEqual(test_config["is_masked_sian"], variant["is_masked_sian"],
                                   f"Mismatch in is_masked_sian for variant {variant}")
                    self.assertEqual(test_config["use_mnist_scaling"], variant["use_mnist_scaling"],
                                   f"Mismatch in use_mnist_scaling for variant {variant}")
                    self.assertEqual(test_config["STEP"], 10,
                                   f"Number of steps is not 10 for variant {variant}")
                    
                except Exception as e:
                    self.fail(f"Training failed for variant {variant}: {str(e)}")
                finally:
                    os.unlink(tmp_file_path)

if __name__ == '__main__':
    unittest.main()