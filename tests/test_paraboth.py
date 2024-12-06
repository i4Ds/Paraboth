# test_paraboth.py

import pytest
import pandas as pd

# Import the paraboth function from your script.
# Replace 'paraboth_script' with the actual name of your Python file without the .py extension.
from paraboth_sentences import paraboth

def test_paraboth_basic():
    """
    Test the basic functionality of paraboth without paraphrasing.
    """
    # Sample input sentences
    gt_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world!",
        "Testing paraphrasing functionality."
    ]
    
    pred_sentences = [
        "A fast brown fox leaps over the lazy dog.",
        "Hi universe!",
        "Checking the paraphrasing feature."
    ]
    
    # Call the paraboth function
    metrics_df, da_info = paraboth(
        gt_sentences,
        pred_sentences,
        n_paraphrases=1,
        paraphrase_gt=False,
        paraphrase_pred=False
    )
    
    # Assertions for metrics
    assert 'ParaBLEU' in metrics_df.columns, "ParaBLEU metric missing in the output."
    assert 'ParaWER' in metrics_df.columns, "ParaWER metric missing in the output."
    assert metrics_df['ParaBLEU'].iloc[0] == 0.8, "ParaBLEU metric value mismatch."
    assert metrics_df['ParaWER'].iloc[0] == {"wer": 0.1}, "ParaWER metric value mismatch."
    
    # Assertions for detailed alignment info
    assert len(da_info) == len(gt_sentences), "Detailed alignment info length mismatch."
    for i in range(len(gt_sentences)):
        assert da_info.iloc[i]['best_paraphrased_gt'] == gt_sentences[i], f"Mismatch in best_paraphrased_gt for sentence {i}."
        assert da_info.iloc[i]['best_paraphrased_pred'] == pred_sentences[i], f"Mismatch in best_paraphrased_pred for sentence {i}."

