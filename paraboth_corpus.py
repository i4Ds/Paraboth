#!/usr/bin/env python

import argparse
import os

from paraboth.data import Text
from paraboth.normalizer import TextNormalizer

from paraboth_corpus import (
    paraboth_corpus
)

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
