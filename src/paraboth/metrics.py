#!/usr/bin/env python

import argparse

from evaluate import load

from paraboth.data import Text
from paraboth.normalizer import TextNormalizer


def calculate_wer_and_bleu(gt_sentences, pred_sentences):
    """
    Calculate Word Error Rate (WER) and BLEU score for given ground truth and predicted sentences.

    Parameters
    ----------
    gt_sentences : list of str
        List containing ground truth sentences.
    pred_sentences : list of str
        List containing predicted sentences.

    Returns
    -------
    wer : float
        Word Error Rate calculated between ground truth and predicted sentences.
    bleu : float
        BLEU score calculated between ground truth and predicted sentences.

    Notes
    -----
    This function normalizes the input sentences before calculating the metrics.
    WER and BLEU are computed using the 'evaluate' library.
    """
    # Load Metrics
    WER = load("wer")
    BLEU = load("bleu")

    # Compute metrics
    wer = WER.compute(predictions=pred_sentences, references=gt_sentences)
    bleu = BLEU.compute(predictions=pred_sentences, references=gt_sentences)

    return wer, bleu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ASR predictions against ground truth."
    )
    parser.add_argument(
        "--gt", type=str, required=True, help="Path to ground truth text file."
    )
    parser.add_argument(
        "--pred", type=str, required=True, help="Path to predictions text file."
    )
    args = parser.parse_args()

    # Read files
    gt_sentences = [Text(args.gt).to_string()]
    pred_sentences = [Text(args.pred).to_string()]

    # Normalize the text
    normalizer = TextNormalizer()
    gt_sentences = normalizer.normalize(gt_sentences)
    pred_sentences = normalizer.normalize(pred_sentences)

    # Calculate WER and BLEU
    wer, bleu = calculate_wer_and_bleu(gt_sentences, pred_sentences)

    print("Bleu & WER on Corpus Level:")
    print(f"WER: {wer:.3f}")
    print(f"BLEU: {bleu}")
