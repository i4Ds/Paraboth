#!/usr/bin/env python

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np

from paraboth.data import Text
from metrics import calculate_wer_and_bleu
from paraboth_corpus import paraboth
from paraboth.paraphraser import Paraphraser


def text_file_to_list_of_sentences(file_path):
    return Text(file_path).to_list_of_strings()


def scramble_words(sentence, scramble_fraction=1.0, seed=42):
    words = sentence.split()
    num_to_scramble = int(len(words) * scramble_fraction)
    indices = list(range(len(words)))
    rng = random.Random(seed)  # Create a Random instance with the given seed
    rng.shuffle(indices)
    scramble_indices = indices[:num_to_scramble]
    words_to_scramble = [words[i] for i in scramble_indices]
    rng.shuffle(words_to_scramble)
    for idx, word_idx in enumerate(scramble_indices):
        words[word_idx] = words_to_scramble[idx]
    return " ".join(words)


def apply_scenarios(pred_sentences):
    scenarios = {}
    scenario_names = []

    # Scenario 1: Normal
    scenario_names.append("Normal")
    scenarios["Normal"] = pred_sentences.copy()

    # Scenario 2: 25% words scrambled
    scenario_names.append("25% Words Scrambled")
    scrambled_25 = [scramble_words(s, scramble_fraction=0.25) for s in pred_sentences]
    scenarios["25% Words Scrambled"] = scrambled_25

    # Scenario 3: 50% words scrambled
    scenario_names.append("50% Words Scrambled")
    scrambled_50 = [scramble_words(s, scramble_fraction=0.50) for s in pred_sentences]
    scenarios["50% Words Scrambled"] = scrambled_50

    # Scenario 4: 75% words scrambled
    scenario_names.append("75% Words Scrambled")
    scrambled_75 = [scramble_words(s, scramble_fraction=0.75) for s in pred_sentences]
    scenarios["75% Words Scrambled"] = scrambled_75

    # Scenario 5: All words scrambled
    scenario_names.append("100% Words Scrambled")
    scrambled_all = [scramble_words(s, scramble_fraction=1.0) for s in pred_sentences]
    scenarios["100% Words Scrambled"] = scrambled_all

    # Scenario 6: Every second sentence scrambled
    scenario_names.append("Every Second Sentence Scrambled")
    scrambled_every_second = pred_sentences.copy()
    for i in range(1, len(scrambled_every_second), 2):
        scrambled_every_second[i] = scramble_words(
            scrambled_every_second[i], scramble_fraction=1.0
        )
    scenarios["Every Second Sentence Scrambled"] = scrambled_every_second

    # Scenario 7: Every second sentence paraphrased
    scenario_names.append("Every Second Sentence Paraphrased")
    paraphrased_every_second = pred_sentences.copy()
    paraphraser = Paraphraser()
    for i in range(1, len(paraphrased_every_second), 2):
        paraphrased_every_second[i] = paraphraser.paraphrase(
            paraphrased_every_second[i], 1
        )
    scenarios["Every Second Sentence Paraphrased"] = paraphrased_every_second

    # Scenario 8: Drop Every Second Sentence
    scenario_names.append("Drop Every Second Sentence")
    dropped_every_second_sentence = pred_sentences.copy()
    for i in range(1, len(dropped_every_second_sentence), 2):
        dropped_every_second_sentence[i] = ""
    scenarios["Drop Every Second Sentence"] = dropped_every_second_sentence

    return scenarios, scenario_names


def plot_metrics(scenario_names, bleu_scores, wer_scores, paraboth_bleu_scores):
    x_indices = np.arange(len(scenario_names))
    width = 0.3  # Width of the bars

    # Plot BLEU and ParaBLEU Scores
    plt.figure(figsize=(12, 7))
    plt.bar(
        x_indices - width / 2,
        bleu_scores,
        width=width,
        label="BLEU",
        color="#1f77b4",
        edgecolor="black",
    )
    plt.bar(
        x_indices + width / 2,
        paraboth_bleu_scores,
        width=width,
        label="ParaBLEU",
        color="#2ca02c",
        edgecolor="black",
    )
    plt.xlabel("Scenarios", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.title("BLEU and ParaBLEU Scores Across Different Scenarios", fontsize=16)
    plt.xticks(x_indices, scenario_names, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.legend(fontsize=12)
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.savefig(f"scramble_bleu.png")  # Save the figure
    plt.tight_layout()
    plt.show()


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
    gt_sentences = text_file_to_list_of_sentences(args.gt)
    pred_sentences = text_file_to_list_of_sentences(args.pred)

    # Apply scenarios
    scenarios, scenario_names = apply_scenarios(pred_sentences)

    # Calculate metrics for each scenario
    bleu_scores = []
    wer_scores = []
    paraboth_bleu_scores = []
    for name in scenario_names:
        wer, bleu = calculate_wer_and_bleu(gt_sentences, scenarios[name])
        bleu_scores.append(bleu)
        wer_scores.append(wer)

        # Calculate ParaBLEU
        metrics, _ = paraboth(
            gt_sentences,
            scenarios[name],
            window_size=1,
            n_paraphrases=5,
            min_matching_value=0.5,
            paraphrase_gt=True,
            paraphrase_pred=True,
        )
        paraboth_bleu = metrics["Aligned and Paraphrased Corpus"]["ParaBLEU"]
        paraboth_wer = metrics["Aligned and Paraphrased Corpus"]["ParaWER"]
        paraboth_bleu_scores.append(paraboth_bleu)

        print(f"Scenario: {name}")
        print(f"  WER: {wer:.3f}")
        print(f"  BLEU: {bleu:.3f}")
        print(f"  ParaBLEU: {paraboth_bleu:.3f}\n")
        print(f"  ParaWER: {paraboth_wer:.3f}\n")

    # Plot the results
    plot_metrics(scenario_names, bleu_scores, wer_scores, paraboth_bleu_scores)
