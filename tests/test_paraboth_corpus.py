from paraboth.paraboth_corpus import paraboth_corpus
from embedder import Embedder
from paraphraser import Paraphraser

def test_paraboth_basic():
    """
    Test the basic functionality of paraboth without paraphrasing.
    """
    # Sample input sentences
    gt_sentences = [
        "der flinke braune fuchs hüpft über den faulen hund.",
        "Hallo Welt!",
        "Wie gehts?",
    ]

    pred_sentences = [
        "der schnelle braune fuchs springt über den trägen hund.",
        "Hoi Welt!",
        "Wie fühlst du dich?",
    ]

    # Init embedder and paraphraser
    embedder = Embedder()
    paraphraser = Paraphraser()

    # Call the paraboth function
    metrics_df, da_info = paraboth_corpus(
        gt_sentences,
        pred_sentences,
        embedder=embedder,
        paraphraser=paraphraser,
        window_size=3,
        n_paraphrases=4,
        min_matching_value=0.5,
        paraphrase_gt=True,
        paraphrase_pred=True,
    )

    # Assertions for metrics
    assert "ParaBLEU" in metrics_df.columns, "ParaBLEU metric missing in the output."
    assert "ParaWER" in metrics_df.columns, "ParaWER metric missing in the output."
    assert metrics_df["ParaBLEU"].iloc[0] > 0.5, "ParaBLEU metric mismatch."
    assert metrics_df["ParaWER"].iloc[0] < 0.4, "ParaWER metric value mismatch."
