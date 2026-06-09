# Slovak Clickbait Detector

Classifies Slovak news headlines as clickbait or legitimate using a fine-tuned [SlovakBERT](https://huggingface.co/gerulata/slovakbert) model. The app runs on FastAPI with a simple web interface.

![App demo](screenshots/app_legit.png)

## Why I built this

There is no public clickbait dataset for Slovak. I scraped 447 headlines from 6 Slovak news sites, manually reviewed every single one, and used them to fine-tune SlovakBERT for binary classification.

Slovak is a low-resource language, so there is not much NLP work done for it compared to English. This project ties into work being done at [KInIT](https://kinit.sk) on disinformation detection (vera.ai, DisAI).

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 89% |
| F1 (macro) | 0.86 |
| Precision (clickbait) | 0.84 |
| Recall (clickbait) | 0.89 |

Tested on 90 held-out headlines (stratified split). Results vary slightly between training runs (F1 between 0.86 and 0.90) because the dataset is small. That is expected.

![Confusion Matrix](screenshots/confusion_matrix.png)

### Error analysis

The model got 10 out of 90 test headlines wrong. But when I went through each one manually, I found that 3-4 of those "mistakes" were actually mislabeled in the dataset, not real model errors.

For example, a headline like *"Premier Fico sa v Bratislave stretol so srbskym ministrom obrany Gasicom"* was labeled as clickbait (because it came from a tabloid site), but it is clearly a normal news headline. The model correctly called it legit.

This is a side effect of how I labeled the data: the first pass was based on which site published the headline, not the headline itself. Some legit headlines from tabloid sites and some clickbaity headlines from quality outlets got the wrong label. I caught and fixed 31 of these during manual review, but a few slipped through into the test set.

After accounting for these, the model's real accuracy is probably closer to 93-94%.

### It learns language patterns, not topics

Same political topic, different phrasing, different result:

| Headline | Prediction | Confidence |
|----------|------------|------------|
| *Fico oznamil nove opatrenia proti inflacii* | LEGIT | 89% |
| *Fico oznamil nove opatrenia proti inflacii. Tento krok zmeni zivoty tisicov Slovakov* | CLICKBAIT | 89% |

Adding "this step will change the lives of thousands of Slovaks" flips it from legit to clickbait. The model learned to detect clickbait writing style, not just certain topics.

![App - Clickbait result](screenshots/app_clickbait.png)

## Dataset

447 headlines scraped from 6 Slovak news websites using `requests` + `BeautifulSoup`, then manually reviewed one by one.

Clickbait sources: cas.sk, topky.sk

Legitimate sources: dennikn.sk, pravda.sk, aktuality.sk, tasr.sk

How I labeled them:

1. First pass: labeled by source (tabloid = clickbait, quality outlet = legit)
2. Went through all 447 headlines manually
3. Corrected 31 labels where the source-based label was wrong
4. Removed 18 truncated headlines (ones ending with "...")

Final split: 178 clickbait, 269 legitimate. I used weighted cross-entropy loss during training to handle the imbalance.

The full dataset is in [`headlines_clean.csv`](headlines_clean.csv).

## Model

**Base model:** [gerulata/slovakbert](https://huggingface.co/gerulata/slovakbert). BERT-based transformer pre-trained on ~20 GB of Slovak text, made by Gerulata/KInIT.

**Fine-tuning setup:**

- Binary classification (clickbait vs. legit)
- 5 epochs, best model picked by validation F1 (usually epoch 2-4)
- Batch size: 16, learning rate: 2e-5, weight decay: 0.01
- Max token length: 64 (headlines are short, this is more than enough)
- Custom `WeightedTrainer` with class-weighted `CrossEntropyLoss`
- Trained on Google Colab with a T4 GPU, takes about 2 minutes

The full training notebook is in [`notebooks/`](notebooks/).

## Running locally

**You need:**

- Python 3.9+
- ~4 GB of disk space (the model is about 500 MB, PyTorch is about 2 GB)

**Setup:**

```bash
git clone https://github.com/4Xplos1on/slovak-clickbait-detector.git
cd slovak-clickbait-detector
python -m pip install -r requirements.txt
```

Download the fine-tuned model from [Google Drive](https://drive.google.com/drive/folders/11YiFrgsaFwWIy0VwSFc-ZCpk1IODcJT6?usp=sharing) and put the contents into a folder called `best_model/` in the project root.

Then run:

```bash
uvicorn app:app --reload
```

Open `http://localhost:8000` in your browser.

**API example:**

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
  app.py                  # FastAPI backend
  index.html              # Web frontend (all in Slovak)
  requirements.txt        # Python dependencies
  headlines_clean.csv     # The dataset (447 labeled headlines)
  notebooks/              # Training notebook from Google Colab
  screenshots/            # Demo screenshots and confusion matrix
  best_model/             # Fine-tuned model weights (not in git, ~500 MB)
```

## Limitations

- **Small dataset.** 447 headlines is enough to show the idea works, but not enough for a real production system. You would need thousands.
- **Labeling noise.** Initial labels came from which site published the headline. I reviewed all of them manually, but some wrong labels probably still slipped through.
- **Slovak only.** Not tested on Czech or other similar languages.
- **No diacritics handling.** Did not test what happens with headlines that are missing diacritics (which is common in informal Slovak text).

## Acknowledgments

- [SlovakBERT](https://huggingface.co/gerulata/slovakbert) by [Gerulata](https://gerulata.com) / [KInIT](https://kinit.sk)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers), [FastAPI](https://fastapi.tiangolo.com), [PyTorch](https://pytorch.org)

## Author

Richard Simek | [GitHub](https://github.com/4Xplos1on)
