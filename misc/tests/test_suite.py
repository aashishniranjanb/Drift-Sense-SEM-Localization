import os
import sys
import unittest
import subprocess
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SUBMISSION = os.path.join(_ROOT, "FINAL_SUBMISSION")
_SAMPLE_PREDS = os.path.join(_SUBMISSION, "verification", "sample_pairs", "predictions.csv")

class TestSubmissionSchema(unittest.TestCase):
    def test_predictions_schema(self):
        self.assertTrue(os.path.exists(_SAMPLE_PREDS), f"Sample predictions missing: {_SAMPLE_PREDS}")
        df = pd.read_csv(_SAMPLE_PREDS)
        expected = ["pair_id", "x", "y", "theta", "scale", "found", "score"]
        self.assertEqual(list(df.columns), expected)
        self.assertGreater(len(df), 0)

    def test_predictions_types_and_invariants(self):
        df = pd.read_csv(_SAMPLE_PREDS)
        self.assertTrue(pd.api.types.is_numeric_dtype(df["x"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(df["y"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(df["theta"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(df["scale"]))
        self.assertTrue(set(df["found"].unique()).issubset({0, 1}))
        self.assertTrue(((df["score"] >= 0.0) & (df["score"] <= 1.0)).all())
        # Check found=0 invariant
        for _, row in df[df["found"] == 0].iterrows():
            self.assertEqual(row["x"], 0.0)
            self.assertEqual(row["y"], 0.0)
            self.assertEqual(row["theta"], 0.0)
            self.assertEqual(row["scale"], 0.0)

class TestCLIInterfaces(unittest.TestCase):
    def test_inference_cli(self):
        inf_script = os.path.join(_SUBMISSION, "inference.py")
        ref_img = os.path.join(_SUBMISSION, "verification", "sample_pairs", "images", "ref_val_dram_000.png")
        search_img = os.path.join(_SUBMISSION, "verification", "sample_pairs", "images", "search_val_dram_000.png")
        if not os.path.exists(ref_img):
            self.skipTest("Sample images not found")
        cmd = [sys.executable, inf_script, "--reference", ref_img, "--search", search_img]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        lines = [l.strip() for l in res.stdout.strip().splitlines() if l.strip()]
        self.assertTrue(any(l.startswith("x=") for l in lines))
        self.assertTrue(any(l.startswith("y=") for l in lines))

    def test_generator_cli(self):
        gen_script = os.path.join(_SUBMISSION, "generate_dataset.py")
        tmp_dir = os.path.join(_ROOT, "tests", "tmp_gen")
        cmd = [sys.executable, gen_script, "--style", "finfet", "--num_pairs", "1", "--output_dir", tmp_dir, "--seed", "101"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Error: {res.stderr}")
        gt_path = os.path.join(tmp_dir, "ground_truth.csv")
        self.assertTrue(os.path.exists(gt_path))
        if os.path.exists(tmp_dir):
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
