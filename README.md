# Slovak Clickbait Detector

Classifies Slovak news headlines as clickbait or legitimate using a fine-tuned [SlovakBERT](https://huggingface.co/gerulata/slovakbert) model. Runs on FastAPI with a simple web interface.

![App demo](screenshots/app_legit.png)

## Why I built this

There is no public clickbait dataset for Slovak. I wanted to see if I could build one from scratch and use it to fine-tune SlovakBERT for binary classification. I scraped 447 headlines from 6 Slovak news sites, went through every single one by hand, and trained a model on them.

Slovak is a low-resource language, so most NLP work skips over it. This project connects to the disinformation detection research at [KInIT](https://kinit.sk) (vera.ai, DisAI).

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 89% |
| F1 (macro) | 0.86 |
| Precision (clickbait) | 0.84 |
| Recall (clickbait) | 0.89 |

Tested on 90 held-out headlines (stratified split). Results shift between runs (F1 between 0.86 and 0.90) because the dataset is small. With 447 samples, that kind of variance is normal.

![Confusion Matrix](screenshots/confusion_matrix.png)

### Error analysis

The model got 10 out of 90 test headlines wrong. I went through each one and found that 3-4 of those were mislabeled in the dataset, not real model errors.

One example: *"Premier Fico sa v Bratislave stretol so srbskym ministrom obrany Gasicom"* was labeled clickbait because it came from a tabloid site. It's a normal news headline. The model called it legit, and it was right.

This happened because my first labeling pass was based on source (tabloid = clickbait, quality outlet = legit). I caught and fixed 31 of these during manual review, but a few wrong labels made it into the test set. Accounting for those, the real accuracy is closer to 92-93%.

### It learns language patterns, not topics

Same political topic, different phrasing, different result:

| Headline | Prediction | Confidence |
|----------|------------|------------|
| *Fico oznamil nove opatrenia proti inflacii* | LEGIT | 89% |
| *Fico oznamil nove opatrenia proti inflacii. Tento krok zmeni zivoty tisicov Slovakov* | CLICKBAIT | 89% |

Adding "this step will change the lives of thousands of Slovaks" flips the prediction. The model reacts to the phrasing, not the topic.

![App - Clickbait result](screenshots/app_clickbait.png)

### Where the model fails

I wrote 28 new headlines myself to test specific edge cases. None of them were in the training data.

**Missing diacritics break detection.** Slovak text online often shows up without diacritics (no s, c, z, t, etc.). SlovakBERT was trained on proper Slovak, so when diacritics are gone, the tokenizer splits words differently. The model can't find the patterns it learned.

| Headline | Prediction | Confidence | Correct? |
|----------|------------|------------|----------|
| *Tato zena z Bratislavy nasla sposob ako schudnut 20 kilo za mesiac* | LEGIT | 63% | No |
| *Nebudte si verili co sa stalo na dialnici pri Zilina* | LEGIT | 97% | No |

Both are obvious clickbait. The first one at least has low confidence (63%), so the model knows something is off. The second one is confidently wrong.

**Emotional manipulation without clickbait vocabulary goes undetected.** The model learned specific phrases that show up in clickbait ("pozrite sa", "konecne prehovoril o tom co sa stalo"). Headlines that use emotional framing without those exact phrases get through.

| Headline | Prediction | Confidence |
|----------|------------|------------|
| *Matka troch deti prisla o vsetko po jednej navsteve uradu* | LEGIT | 97% |
| *Slovenska nemocnica odmietla pacienta a ten zomrel pred vchodom* | LEGIT | 99% |
| *Ucitelia su na pokraji zrutenia a nikoho to nezaujima* | LEGIT | 98% |

These use emotional framing that is common in tabloids. Some could appear in quality outlets too, but the style is typical clickbait. The model has no reaction to emotional tone. It only responds to the specific words it saw during training.

**It catches some clickbait phrases but not others.**

| Headline | Prediction | Confidence |
|----------|------------|------------|
| *Pozrite sa co sa naslo v detskej vyzive predavanej v Lidli* | CLICKBAIT | 98% |
| *Znamy slovensky herec konecne prehovoril o tom co sa stalo* | CLICKBAIT | 96% |
| *Lekari tvrdia ze tato potravina sposobuje rakovinu a Slovaci ju jedia denne* | LEGIT | 99% |
| *Jedna vec ktoru by ste nikdy nemali robit rano na prazdny zaludok* | LEGIT | 95% |

"Pozrite sa" and "konecne prehovoril" trigger the detector. "Lekari tvrdia" and "jedna vec ktoru by ste nikdy nemali" do not, even though they are classic clickbait patterns. The model learned specific vocabulary, not what makes something clickbait.

**Same topic, different framing.**

| Headline | Prediction | Confidence |
|----------|------------|------------|
| *NBS zvysila zakladnu urokovu sadzbu na 4,5 percenta* | LEGIT | 99% |
| *Urokove sadzby letia hore. Vase hypoteky budu dramaticky drahsie* | LEGIT | 99% |
| *Pocet turistov na Slovensku vzrastol o 20 percent* | LEGIT | 99% |
| *Turisti zaplavili Slovensko. Pozrite sa kam sa uz neoplatí ist* | LEGIT | 87% |

The second headline in each pair uses sensational framing, but the model still calls them legit. The last one drops to 87% confidence (probably because of "pozrite sa"), but still gets it wrong.

**What works well:** short factual headlines (99% confidence), neutral question-style headlines, and headlines containing clickbait phrases from the training data. The model also handles legit headlines that sound surprising, like *"Vedci z SAV objavili novy druh jaskynneho chrobaka na Slovensku"* (LEGIT, 98%).

**Why this happens:** I only had 178 clickbait headlines to train on. The model picked up the phrases and patterns from those examples, but 178 is not enough variety to cover all the ways clickbait can look.

## Dataset

447 headlines scraped from 6 Slovak news websites using `requests` + `BeautifulSoup`, then reviewed one by one.

Clickbait sources: cas.sk, topky.sk
Legitimate sources: dennikn.sk, pravda.sk, aktuality.sk, tasr.sk

How I labeled them:

1. First pass: labeled by source (tabloid = clickbait, quality outlet = legit)
2. Went through all 447 headlines manually
3. Fixed 31 labels where the source-based label was wrong
4. Removed 18 truncated headlines (ones ending with "...")

Final split: 178 clickbait, 269 legitimate. I used weighted cross-entropy loss during training to handle the imbalance.

The full dataset is in [`headlines_clean.csv`](headlines_clean.csv).

## Model

**Base model:** [gerulata/slovakbert](https://huggingface.co/gerulata/slovakbert). A BERT-based transformer pre-trained on ~20 GB of Slovak text, built by Gerulata/KInIT.

**Fine-tuning setup:**

- Binary classification (clickbait vs. legit)
- 5 epochs, best model picked by validation F1 (usually epoch 2-4)
- Batch size: 16, learning rate: 2e-5, weight decay: 0.01
- Max token length: 64 (headlines are short, this covers all of them)
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
pip install -r requirements.txt
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

- **Small dataset.** 447 headlines shows the idea works, but a real system would need thousands.
- **Labeling noise.** My first pass labeled by source, not content. I fixed 31 mistakes during manual review, but some wrong labels are still in there.
- **Slovak only.** I haven't tested on Czech or other similar languages.
- **No diacritics handling.** Headlines without diacritics (common in informal Slovak online) break the model because SlovakBERT was trained on proper text.
- **Vocabulary-dependent.** The model catches clickbait phrases it saw in training but misses emotional manipulation and sensational framing with different wording (see "Where the model fails" above).

## Future work

- **Expand the dataset with targeted synthetic data.** The edge case analysis shows specific gaps: emotional manipulation without clickbait vocabulary, sensational framing, missing-diacritics variants. Instead of generating random clickbait headlines, I would generate examples targeting these failure categories, then validate them by hand before adding them to training. Cegin et al. ([EMNLP 2025](https://kinit.sk/publication/a-rigorous-evaluation-of-llm-data-generation-strategies-for-low-resource-languages/)) showed that LLM-generated data can help for low-resource languages like Slovak, but the generation strategy matters more than volume. Naive generation produces repetitive samples that don't improve generalization.
- **Test cross-lingual transfer** from Czech clickbait data. Czech and Slovak are close enough that shared clickbait patterns could help fill the gaps in my dataset.
- **Add explainability** using attention visualization or occlusion-based methods to show which words the model focuses on when making predictions.
- **Handle missing diacritics** by adding a diacritics restoration step before classification, or by including headlines without diacritics in the training data.
- **Benchmark on standardized Slovak NLP tasks** from skLEP (sentiment analysis, semantic similarity) to see how SlovakBERT fine-tuning performs across different text classification problems.

## Acknowledgments

- [SlovakBERT](https://huggingface.co/gerulata/slovakbert) by [Gerulata](https://gerulata.com) / [KInIT](https://kinit.sk)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers), [FastAPI](https://fastapi.tiangolo.com), [PyTorch](https://pytorch.org)

## Author

Richard Simek | [GitHub](https://github.com/4Xplos1on)
