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
# evidence the vector encodes sentiment rather than movie-ness.
EVAL_PROMPTS = [
    "Describe your commute this morning.",
    "Tell me about the last meal you cooked.",
    "What is the weather like where you are?",
    "Describe a typical Tuesday.",
    "Tell me about your neighbourhood.",
    "What did you do last weekend?",
    "Describe the room you are in right now.",
    "Tell me about a walk you took recently.",
    "How do you usually start your day?",
    "Describe the view from a train window.",
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
