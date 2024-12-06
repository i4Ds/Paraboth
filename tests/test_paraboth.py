# test_paraboth.py
from paraboth.paraboth_sentences import paraboth
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
    paraphraser = Paraphraser()

    # Call the paraboth function
    metrics_df, da_info = paraboth(
        gt_sentences,
        pred_sentences,
        paraphraser=paraphraser,
        n_paraphrases=4,
        paraphrase_gt=True,
        paraphrase_pred=True,
    )

    # Assertions for metrics
    assert "ParaBLEU" in metrics_df.columns, "ParaBLEU metric missing in the output."
    assert "ParaWER" in metrics_df.columns, "ParaWER metric missing in the output."
    assert metrics_df["ParaBLEU"].iloc[0] > 0.5, "ParaBLEU metric mismatch."
    assert metrics_df["ParaWER"].iloc[0] < 0.4, "ParaWER metric value mismatch."
