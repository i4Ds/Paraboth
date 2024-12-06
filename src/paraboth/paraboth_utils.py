#!/usr/bin/env python

from dtwsa import SentenceAligner
from sklearn.metrics.pairwise import cosine_similarity

from paraboth.embedder import Embedder

def create_sentence_combinations(texts, window_size=1):
    combined_texts = []
    n = len(texts)

    for i in range(n):
        # Add the original sentence
        combined_texts.append(texts[i])

        # Generate combinations within the window size
        for w in range(1, window_size + 1):
            # Combine with previous sentences
            if i - w >= 0:
                prev_combination = " ".join(texts[i - w : i + 1])
                combined_texts.append(prev_combination)

            # Combine with next sentences
            if i + w < n:
                next_combination = " ".join(texts[i : i + w + 1])
                combined_texts.append(next_combination)

            # Combine with both previous and next sentences
            if i - w >= 0 and i + w < n:
                full_combination = " ".join(texts[i - w : i + w + 1])
                combined_texts.append(full_combination)

    return combined_texts


def compute_similarity_matrix(
    pred_embeddings, gt_embeddings, similariy_metric="cosine"
):
    if similariy_metric == "cosine":
        return cosine_similarity(pred_embeddings, gt_embeddings)
    else:
        raise ValueError(f"Unknown similarity metric: {similariy_metric}")


def align_sentences(
    predictions, ground_truth, similarity_matrix, min_matching_value=0.5
):
    aligner = SentenceAligner(
        similarity_matrix=similarity_matrix, min_matching_value=min_matching_value
    )
    alignment, score = aligner.align_sentences(predictions, ground_truth)
    return alignment, score


def align_corpus(gt_sentences, pred_sentences, min_matching_value, embedder=None):
    """
    Align two corpora of sentences using embedding-based similarity.

    This function aligns sentences from the ground truth corpus with sentences
    from the predicted corpus. It can handle sentences that are split by a
    delimiter or by token number.

    Parameters
    ----------
    gt_sentences : list of str
        Ground truth sentences to align.
    pred_sentences : list of str
        Predicted sentences to align with ground truth.
    min_matching_value : float
        Minimum similarity score required for a match between sentences.
    embedder: Embedder
        Embedder object to use for sentence embeddings.

    Returns
    -------
    alignment : list of tuple
        List of (pred_idx, gt_idx) pairs representing the aligned sentences.
    score : float
        Total similarity score of the alignment.

    Notes
    -----
    The function creates sentence combinations, computes embeddings, calculates
    a similarity matrix, and then aligns the sentences based on this similarity.
    """
    # Initialize the model
    model = Embedder() if not embedder else embedder

    # Compute embeddings
    gt_embeddings = model.embed_chunks(gt_sentences)
    pred_embeddings = model.embed_chunks(pred_sentences)

    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(pred_embeddings, gt_embeddings)

    # Align sentences
    alignment, score = align_sentences(
        pred_sentences,
        gt_sentences,
        similarity_matrix,
        min_matching_value=min_matching_value,
    )

    # Print alignment score
    print(
        f"Mean embedding score of the best sentence-to-sentence alignment: {score/len(alignment):.3f}"
    )
    return alignment, score


def best_match_n_to_n_sentences(
    gt_paraphrases, pred_sentences, metric
):
    """
    Find the best matching pair of sentences between ground truth paraphrases and predicted sentences.

    Parameters
    ----------
    gt_paraphrases : list of str
        List of ground truth paraphrased sentences.
    pred_sentences : list of str
        List of predicted sentences.
    metric : str
        The metric to use for comparing sentences (currently only supports 'wer').

    Returns
    -------
    best_gt : str
        The best matching ground truth sentence.
    best_pred : str
        The best matching predicted sentence.

    Notes
    -----
    This function compares each prediction with each paraphrased ground truth sentence
    using the Word Error Rate (WER) metric. It returns the pair with the lowest WER.

    Examples
    --------
    >>> gt_paraphrases = ["The quick brown fox", "A fast brown fox"]
    >>> pred_sentences = ["The quick brown dog", "A rapid brown fox"]
    >>> best_gt, best_pred = best_match_n_to_n_sentences(gt_paraphrases, pred_sentences, 0.5, 'wer')
    >>> print(best_gt, best_pred)
    A fast brown fox A rapid brown fox
    """
    # Based on 1-to-1 alignment, find the best match of the paraphrased sentences.
    min_wer = float("inf")
    best_pred = ""
    best_gt = ""

    # Compare each prediction with each paraphrased ground truth sentence
    for pred_variant in pred_sentences:
        for gt_variant in gt_paraphrases:
            # Compute WER between the current prediction and ground truth variant
            current_wer = metric.compute(
                predictions=[pred_variant], references=[gt_variant]
            )
            # Update the best match if the current WER is lower
            if current_wer < min_wer:
                min_wer = current_wer
                best_pred = pred_variant
                best_gt = gt_variant

    return best_gt, best_pred
