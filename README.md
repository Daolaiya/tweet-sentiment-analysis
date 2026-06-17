# Tweet Sentiment Analysis

End-to-end sentiment-classification pipeline for short, informal English text (tweets).
The project ingests the [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
dataset of 1.6 M tweets, cleans and vectorises it, and benchmarks four
classical machine-learning models against four densely-connected neural networks.

## About
- LinkedIn: <https://www.linkedin.com/in/damilola-olaiya/>

## Hypothesis / motivation
> How can a company use tweets about itself, its products, or its employees to
> gauge overall sentiment in order to make better decisions about demand and
> marketing?

## Project structure
```
SA/
├── part_1_wrangling_eda_processing.ipynb   # Load, clean, EDA, TF-IDF, train/test split
├── part_2_1_modelling.ipynb                # Bernoulli NB, LinearSVC, Logistic Regression
├── part_2_2_modelling.ipynb                # Dense-NN baseline + size/regularisation/dropout
├── images/                                 # Pre-generated word clouds rendered in Part 1
├── models/                                 # Pickled fitted models (output of Parts 2-1 / 2-2)
├── raw_data/                               # (gitignored) Sentiment140 CSV goes here
├── data/                                   # (gitignored) intermediate pickles produced by Part 1
├── Final Project Report.pdf
├── Final Project Presentation.pdf
├── requirements.txt
└── README.md
```

`raw_data/` and `data/` are intentionally **not** committed; the raw CSV is
~228 MB and the intermediate TF-IDF matrices are several GB. They are
regenerated locally by running Part 1.

## Setup

1. Clone the repository and `cd` into it.
2. Create and activate a Python 3.10 virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the NLTK corpora used by Part 1 (one-time):
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
   ```
5. Download `training.1600000.processed.noemoticon.csv` from
   [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140),
   rename it to `data.csv`, and place it in a new `raw_data/` folder at the
   repository root.
6. Create empty `data/` and `models/` folders if they do not already exist —
   Part 1 and Part 2-1/2-2 write their pickles there.

## How to run

Run the notebooks **in order**:

| # | Notebook | What it does | Approx. runtime\* |
|---|----------|--------------|-------------------|
| 1 | `part_1_wrangling_eda_processing.ipynb` | Load CSV, decontract, regex-clean, lemmatise, plot, TF-IDF (uni- and bi-grams, max 500 000 features), 80/20 split, pickle everything into `data/`. | ~3–5 min |
| 2 | `part_2_1_modelling.ipynb` | Fit and evaluate Bernoulli NB, LinearSVC, and Logistic Regression on the pickled TF-IDF matrices. | ~5–10 min |
| 3 | `part_2_2_modelling.ipynb` | Convert the sparse matrix to a `tf.SparseTensor`, then train a baseline dense net and three variants (reduced size, L2-regularised, dropout). | ~2 min on CPU |

\* Times measured on a modern laptop; your mileage will vary, especially for Part 1's TF-IDF step.

## Results at a glance

| Model | Train accuracy | Test accuracy |
|-------|---------------:|--------------:|
| Bernoulli Naive Bayes              | 82.8 % | 80.0 % |
| Linear SVC                         | 91.0 % | 81.6 % |
| **Logistic Regression** (best)     | **87.2 %** | **82.6 %** |
| Dense NN — baseline (64-64-1)      | 85.2 % | 81.8 % |
| Dense NN — reduced (32-1)          | 78.2 % | 78.3 % |
| Dense NN — L2-regularised          | 76.9 % | 76.9 % |
| Dense NN — dropout                 | 77.4 % | 78.4 % |

Logistic Regression on TF-IDF (uni- + bi-grams) is the strongest model on this
preprocessing pipeline. The neural baseline is competitive but does not surpass
it, and the three regularisation strategies actually hurt performance —
consistent with the dataset being large relative to the network capacity.

## Further reading
- `Final Project Report.pdf` — full write-up of the methodology and results.
- `Final Project Presentation.pdf` — slide deck summary.
