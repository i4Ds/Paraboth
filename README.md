# Diarization Evaluation

This project provides tools for evaluating ASR (Automatic Speech Recognition) predictions against ground truth, using both standard metrics and paraphrase-based metrics. The following steps are performed:
1. Text normalization.
2. Sentence emebdding via an embedding model.
3. Sentence alignment via dynamic time warping aka DTW.
4. Paraphrasing of the sentences via an LLM.
5. Selection of the best paraphrases via the word error rate.
6. Calculation of the paraphrase-based metrics.

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
## Usage
### Normalizer
Please have a look at `normalizer.py` to see how the text normalization is done.

### Standard Metrics (metrics.py)

To calculate standard metrics (WER and BLEU) on corpus:
```bash
python metrics.py --gt <path_to_ground_truth_file> --pred <path_to_predictions_file>
```
Parameters:
- `--gt`: Path to the ground truth text file (required)
- `--pred`: Path to the predictions text file (required)

Example:
```bash
python metrics.py --gt 01_eval_gt.txt --pred 01_eval_gladia.txt
```

### Paraphrase-based Metrics (paraboth.py)

To calculate paraphrase-based metrics:
python paraboth.py --gt <path_to_ground_truth_file> --pred <path_to_predictions_file> [optional_parameters]

Parameters:
- `--gt`: Path to ground truth text file (required)
- `--pred`: Path to predictions text file (required)
- `--n_paraphrases`: Number of paraphrases to generate (default: 3)
- `--paraphrase_gt`: Whether to paraphrase ground truth (default: True)
- `--paraphrase_pred`: Whether to paraphrase predictions (default: True)
- `--window_size`: Window size for sentence combinations (default: 2)
- `--min_matching_value`: Minimum matching value for sentence alignment (default: 0.5)

Example:
```bash
python paraboth.py --gt 01_eval_gt.txt --pred 01_eval_gladia.txt --n_paraphrases 5
```

This command will:
1. Generate 5 paraphrases for each sentence
2. Paraphrase the ground truth (default behavior)
3. Calculate various metrics including ParaBLEU and ParaWER

Results will be saved in a TSV file named with the prediction file and timestamp.

## Output

Both scripts will print metrics to the console. `paraboth.py` will also save detailed results to a TSV file.
