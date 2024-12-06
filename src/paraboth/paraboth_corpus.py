#!/usr/bin/env python

import argparse
import os

import pandas as pd
from evaluate import load

from paraboth.data import Text
from paraboth.embedder import Embedder
from paraboth.normalizer import TextNormalizer
from paraboth.paraphraser import Paraphraser

from paraboth.paraboth_utils import (
    align_corpus,
    create_sentence_combinations,
    best_match_n_to_n_sentences,
)


def paraboth_corpus(
    gt_sentences,
    pred_sentences,
    window_size,
    n_paraphrases,
    min_matching_value,
    paraphraser,
    embedder,
    paraphrase_gt=True,
    paraphrase_pred=True,
):
    # Load Metrics
    WER = load("wer")
    BLEU = load("bleu")

    # Optionally create sentence combinations
    combined_gt = create_sentence_combinations(gt_sentences, window_size)
    combined_pred = create_sentence_combinations(pred_sentences, window_size)

    # Align sentences
    alignment, score = align_corpus(
        combined_gt, combined_pred, min_matching_value, embedder
    )

    # Fix alignment
    print(
        f"Mean embedding score of the best sentence-to-sentence alignment: {score/len(alignment):.3f}"
    )

    # Generate multiple paraphrases for each ground truth sentence
    if paraphrase_gt:
        paraphrased_gt_list = paraphraser.paraphrase_list(combined_gt, n_paraphrases)
    else:
        paraphrased_gt_list = [[x] for x in combined_gt]

    # Paraphrase predictions
    if paraphrase_pred:
        paraphrased_pred_list = paraphraser.paraphrase_list(
            combined_pred, n_paraphrases
        )
    else:
        paraphrased_pred_list = [[x] for x in combined_pred]

    # Align predictions with the best paraphrased ground truth sentences
    final_predictions = []
    final_references = []
    detailed_alignment_info = []

    # Go through the alignment, select the paraphrases sentences.
    for pred_idx, gt_idx in alignment:
        pred_paraphrases = paraphrased_pred_list[pred_idx]
        gt_paraphrases = paraphrased_gt_list[gt_idx]

        # Keep the best match for this alignment.
        best_gt, best_pred = best_match_n_to_n_sentences(
            gt_paraphrases, pred_paraphrases, WER
        )
        final_predictions.append(best_pred)
        final_references.append(best_gt)

        # Store the detailed info for debugging.
        detailed_alignment_info.append(
            {
                "original_pred": paraphrased_pred_list[pred_idx],
                "original_gt": paraphrased_gt_list[gt_idx],
                "best_paraphrased_pred": best_pred,
                "best_paraphrased_gt": best_gt,
            }
        )

    # Add detailed alignment info to DataFrame
    da_info = pd.DataFrame(detailed_alignment_info)

    # Calculate metrics
    metrics = {
        "ParaBLEU": BLEU.compute(
            predictions=[" ".join(final_predictions)],
            references=[[" ".join(final_references)]],
        )["bleu"],
        "ParaWER": WER.compute(
            predictions=[" ".join(final_predictions)],
            references=[" ".join(final_references)],
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
        "--window_size",
        type=int,
        default=0,
        help="Window size for sentence combinations.",
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="results",
        help="Base output directory for results.",
    )
    parser.add_argument(
        "--min_matching_value",
        type=float,
        default=0.5,
        help="Minimum matching value for sentence alignment. If below, no matching is possible.",
    )
    args = parser.parse_args()

    # Read files
    gt_sentences = Text(args.gt).to_list_of_strings()
    pred_sentences = Text(args.pred).to_list_of_strings()

    # Normalize the sentences

    normalizer = TextNormalizer()
    normalized_gt = normalizer.normalize(gt_sentences)
    normalized_pred = normalizer.normalize(pred_sentences)

    metrics, detailed_alignment_info = paraboth_corpus(
        normalized_gt,
        normalized_pred,
        window_size=args.window_size,
        n_paraphrases=args.n_paraphrases,
        paraphrase_gt=args.paraphrase_gt,
        paraphrase_pred=args.paraphrase_pred,
        min_matching_value=args.min_matching_value,
    )

    # Save results
    os.makedirs(args.base_output_dir, exist_ok=True)
    metrics.to_csv(os.path.join(args.base_output_dir, "metrics.csv"), index=False)
    detailed_alignment_info.to_csv(
        os.path.join(args.base_output_dir, "detailed_alignment_info.csv"), index=False
    )
