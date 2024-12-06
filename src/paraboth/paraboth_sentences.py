#!/usr/bin/env python

import argparse
import os

import pandas as pd
from evaluate import load

from paraboth.data import Text
from paraboth.normalizer import TextNormalizer
from paraboth.paraphraser import Paraphraser

from paraboth.paraboth_utils import (
    best_match_n_to_n_sentences
)

def paraboth(
    gt_sentences,
    pred_sentences,
    n_paraphrases,
    paraphrase_gt=True,
    paraphrase_pred=True,
):
    # Load Metrics
    WER = load("wer")
    BLEU = load("bleu")

    # Initialize paraphraser
    paraphraser = Paraphraser()

    # Generate multiple paraphrases for each ground truth sentence
    if paraphrase_gt:
        paraphrased_gt_list = paraphraser.paraphrase_list(gt_sentences, n_paraphrases)
    else:
        paraphrased_gt_list = [[x] for x in gt_sentences]

    # Paraphrase predictions
    if paraphrase_pred:
        paraphrased_pred_list = paraphraser.paraphrase_list(
            pred_sentences, n_paraphrases
        )
    else:
        paraphrased_pred_list = [[x] for x in pred_sentences]

    # Align predictions with the best paraphrased ground truth sentences
    final_predictions = []
    final_references = []
    detailed_alignment_info = []

    # Go through the alignment, select the paraphrases sentences.
    for gt_paraphrases, pred_paraphrases in zip(
        paraphrased_gt_list, paraphrased_pred_list
    ):
        # Keep the best match for this alignment.
        best_gt, best_pred = best_match_n_to_n_sentences(
            gt_paraphrases, pred_paraphrases, WER
        )
        final_predictions.append(best_pred)
        final_references.append(best_gt)

        # Store the detailed info for debugging.
        detailed_alignment_info.append(
            {
                "best_paraphrased_pred": best_pred,
                "best_paraphrased_gt": best_gt,
            }
        )

    # Add detailed alignment info to DataFrame
    da_info = pd.DataFrame(detailed_alignment_info)

    # Calculate metrics
    metrics = {
        "ParaBLEU": BLEU.compute(
            predictions=final_predictions,
            references=paraphrased_gt_list, # Bleu already accepts 1-to-n references.
        )["bleu"],
        "ParaWER": WER.compute(
            predictions=final_predictions,
            references=final_references,
        ),
    }

    # Convert the metrics dictionary to a pandas DataFrame
    metrics_df = pd.DataFrame([metrics])
    return metrics_df, da_info


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
    parser.add_argument(
        "--n_paraphrases",
        type=int,
        default=6,
        help="Number of paraphrases to generate.",
    )
    parser.add_argument(
        "--paraphrase_gt",
        type=bool,
        default=True,
        help="Whether to paraphrase ground truth as well.",
    )
    parser.add_argument(
        "--paraphrase_pred",
        type=bool,
        default=True,
        help="Whether to paraphrase predictions as well.",
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="results",
        help="Base output directory for results.",
    )
    args = parser.parse_args()

    # Read files
    gt_sentences = Text(args.gt).to_list_of_strings()
    pred_sentences = Text(args.pred).to_list_of_strings()

    # Normalize the sentences

    normalizer = TextNormalizer()
    normalized_gt = normalizer.normalize(gt_sentences)
    normalized_pred = normalizer.normalize(pred_sentences)

    metrics, detailed_alignment_info = paraboth(
        normalized_gt,
        normalized_pred,
        n_paraphrases=args.n_paraphrases,
        paraphrase_gt=args.paraphrase_gt,
        paraphrase_pred=args.paraphrase_pred,
    )

    # Save results
    os.makedirs(args.base_output_dir, exist_ok=True)
    metrics.to_csv(os.path.join(args.base_output_dir, "metrics.csv"), index=False)
    detailed_alignment_info.to_csv(
        os.path.join(args.base_output_dir, "detailed_alignment_info.csv"), index=False
    )
