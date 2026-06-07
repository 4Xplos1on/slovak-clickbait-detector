# Slovak Clickbait Detector

Binary classifier that detects clickbait in Slovak news headlines using a fine-tuned [SlovakBERT](https://huggingface.co/gerulata/slovakbert) model. Includes a FastAPI backend and a web demo.

![App demo](screenshots/app_legit.png)

## Why this exists

There is no publicly available clickbait detection dataset for Slovak. This project creates one from scratch (447 manually reviewed headlines from 6 Slovak news sources) and fine-tunes SlovakBERT to classify headlines as legitimate news or clickbait.

This connects to broader work on disinformation detection in low-resource languages, an area actively researched at [KInIT](https://kinit.sk) (vera.ai, DisAI).

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 89% |
| F1 (macro) | 0.86 |
| Precision (clickbait) | 0.84 |
| Recall (clickbait) | 0.89 |

Evaluated on a held-out test set of 90 headlines (stratified split). Manual review of the 10 misclassified headlines showed that 3-4 were labeling errors rather than model mistakes, putting effective accuracy closer to 93-94%.

Variance between training runs (F1 0.86-0.90) is expected given the small dataset.

![Confusion Matrix](screenshots/confusion_matrix.png)

### The model learns language patterns, not topics

The same political topic gets classified differently depending on how it's phrased:

| Headline | Prediction | Confidence |
|----------|------------|------------|
| *Fico oznamil nove opatrenia proti inflacii* | LEGIT | 89% |
| *Fico oznamil nove opatrenia proti inflacii. Tento krok zmeni zivoty tisicov Slovakov* | CLICKBAIT | 89% |

Adding sensationalist language ("this step will change the lives of thousands of Slovaks") flips the prediction. This confirms the model picked up on clickbait writing style, not subject matter.

![App - Clickbait result](screenshots/app_clickbait.png)

## Dataset

**447 headlines** scraped from 6 Slovak news websites using `requests` + `BeautifulSoup`, then manually reviewed.

Sources (clickbait): cas.sk, topky.sk

Sources (legitimate): dennikn.sk, pravda.sk, aktuality.sk, tasr.sk

**Labeling process:**

1. Pre-labeled by source (tabloid sites = clickbait, quality outlets = legit)
2. All 447 headlines manually reviewed
3. 31 labels corrected where the source-based label was wrong
4. 18 truncated headlines removed

Final distribution: 178 clickbait, 269 legitimate. Class imbalance handled with weighted cross-entropy loss during training.

The dataset is included as [`headlines_clean.csv`](headlines_clean.csv).

## Model

**Base model:** [gerulata/slovakbert](https://huggingface.co/gerulata/slovakbert), a BERT-based transformer pre-trained on ~20 GB of Slovak text, developed by Gerulata/KInIT.

**Fine-tuning details:**

- Task: binary sequence classification
- Epochs: 5 (best model selected by validation F1, typically epoch 2-4)
- Batch size: 16, learning rate: 2e-5, weight decay: 0.01
- Max token length: 64 (sufficient for headlines)
- Custom `WeightedTrainer` with class-weighted `CrossEntropyLoss`
- Trained on Google Colab (T4 GPU), takes ~2 minutes

The training notebook is in [`notebooks/`](notebooks/) for full reproducibility.

## Running locally

### Requirements

- Python 3.9+
- The fine-tuned model files (not included in the repo due to size, ~500 MB)

### Setup

```bash
git clone https://github.com/4Xplos1on/slovak-clickbait-detector.git
cd slovak-clickbait-detector

pip install -r requirements.txt

# Place your fine-tuned model in best_model/
# (train using the notebook in notebooks/, or contact me)

uvicorn app:app --reload
```

Open `http://localhost:8000` in your browser.

### API usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"headline": "Fico oznamil nove opatrenia proti inflacii"}'
```

```json
{
  "headline": "Fico oznamil nove opatrenia proti inflacii",
  "label": "LEGIT",
  "confidence": 0.8941
}
```

## Project structure

```
slovak-clickbait-detector/
  app.py                  # FastAPI backend (model loading + /predict endpoint)
  index.html              # Web frontend (Slovak UI)
  requirements.txt        # Python dependencies
  headlines_clean.csv     # Dataset (447 labeled headlines)
  notebooks/              # Training notebook from Google Colab
  screenshots/            # Demo screenshots and confusion matrix
  best_model/             # Fine-tuned model weights (not tracked, ~500 MB)
```

## Limitations

- **Small dataset.** 447 headlines demonstrates the approach but is not production-ready. A real system would need thousands of examples.
- **Source-based pre-labeling.** Initial labels came from which site published the headline, not from the headline itself. Manual review caught the obvious mismatches, but some label noise likely remains.
- **Slovak only.** Not tested on Czech or other related languages, despite mutual intelligibility.
- **No diacritics robustness.** Headlines without proper Slovak diacritics (common in informal text) were not tested.

## Acknowledgments

- [SlovakBERT](https://huggingface.co/gerulata/slovakbert) by [Gerulata](https://gerulata.com) / [KInIT](https://kinit.sk)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers), [FastAPI](https://fastapi.tiangolo.com), [PyTorch](https://pytorch.org)

## Author

Richard Simek | [GitHub](https://github.com/4Xplos1on)
