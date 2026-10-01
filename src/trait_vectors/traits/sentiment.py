"""Phase 0 toy trait: sentiment. The only file that knows about movie reviews.

Extraction data is SST-2 (Stanford Sentiment Treebank, binary), a set of
short movie-review sentences labelled positive (1) or negative (0). Scoring
uses a small DistilBERT classifier fine-tuned on the same task, run on CPU.
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from trait_vectors import config

DATASET = "stanfordnlp/sst2"
MIN_WORDS = 8
N_PER_SIDE = 100
SCORER = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

DESCRIPTION = (
    f"{DATASET} train split, sentences with >= {MIN_WORDS} words, "
    f"{N_PER_SIDE} positive + {N_PER_SIDE} negative, shuffled with seed {config.SEED}"
)

# Neutral, and deliberately not about movies: a steering effect here is
# evidence the vector encodes sentiment rather than movie-ness. Phrased as
# writing tasks about a topic, not questions about the model's own experience:
# the first sweep showed that "Describe your commute" makes the model spend
# most of its reply on an "As an AI I don't have a body" disclaimer, leaving
# little text to carry any tone.
EVAL_PROMPTS = [
    "Write a short paragraph about a morning commute.",
    "Write a short paragraph about cooking dinner on a weeknight.",
    "Write a short paragraph about the weather in autumn.",
    "Write a short paragraph about a typical Tuesday at an office.",
    "Write a short paragraph about a quiet neighbourhood.",
    "Write a short paragraph about a weekend at home.",
    "Write a short paragraph about a small kitchen.",
    "Write a short paragraph about a walk in a city park.",
    "Write a short paragraph about the first hour of the day.",
    "Write a short paragraph about the view from a train window.",
    "Write a short paragraph about a public library.",
    "Write a short paragraph about waiting at a bus stop.",
    "Write a short paragraph about a grocery store on a Saturday.",
    "Write a short paragraph about a garden in spring.",
    "Write a short paragraph about a long car journey.",
    "Write a short paragraph about a school classroom.",
    "Write a short paragraph about a rainy afternoon.",
    "Write a short paragraph about a busy train station.",
    "Write a short paragraph about a family dinner.",
    "Write a short paragraph about a hospital waiting room.",
    "Write a short paragraph about a walk along a river.",
    "Write a short paragraph about a shared apartment.",
    "Write a short paragraph about a village in winter.",
    "Write a short paragraph about a coffee shop in the morning.",
    "Write a short paragraph about a day at the beach.",
    "Write a short paragraph about repairing a bicycle.",
    "Write a short paragraph about a night shift.",
    "Write a short paragraph about a farmers' market.",
    "Write a short paragraph about moving to a new city.",
    "Write a short paragraph about a Sunday afternoon.",
]


class SentimentTrait:
    """Satisfies the Trait protocol (see traits/base.py) by having its three methods."""

    def __init__(self):
        self._scorer = None  # (tokenizer, model, index of POSITIVE); loaded on first score()

    def extraction_pairs(self):
        """(P_plus, P_minus): positive and negative reviews as user messages."""
        ds = load_dataset(DATASET, split="train")
        ds = ds.filter(lambda ex: len(ex["sentence"].split()) >= MIN_WORDS)
        ds = ds.shuffle(seed=config.SEED)
        pos = ds.filter(lambda ex: ex["label"] == 1).select(range(N_PER_SIDE))
        neg = ds.filter(lambda ex: ex["label"] == 0).select(range(N_PER_SIDE))
        return (
            [_as_prompt(s) for s in pos["sentence"]],
            [_as_prompt(s) for s in neg["sentence"]],
        )

    def eval_prompts(self):
        return list(EVAL_PROMPTS)

    def score(self, prompt, response):
        """P(positive) of the response under the DistilBERT SST-2 classifier."""
        if self._scorer is None:
            tokenizer = AutoTokenizer.from_pretrained(SCORER)
            model = AutoModelForSequenceClassification.from_pretrained(SCORER)
            model.eval()
            self._scorer = (tokenizer, model, model.config.label2id["POSITIVE"])
        tokenizer, model, positive = self._scorer

        inputs = tokenizer(response, return_tensors="pt", truncation=True)
        with torch.inference_mode():
            logits = model(**inputs).logits  # shape (1, 2): scores for NEGATIVE, POSITIVE
        return torch.softmax(logits, dim=-1)[0, positive].item()


def _as_prompt(sentence):
    return f'Here is a movie review: "{sentence.strip()}"'
