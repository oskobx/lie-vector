"""Phase 2 spec §10: true-lead questions, and their hand spot-check.

    uv run python scripts/26_truelead_questions.py generate   # one API call per scenario
    uv run python scripts/26_truelead_questions.py check      # Oskar reads 20, blind to nothing (it's a wording check)

A true-lead question assumes the TRUE answer ("This car has been in a flood,
right?"), where the existing leading question assumes the false one. Written
from data/lying/truelead_prompt.md, filled with the
scenario's fact, neutral question, true answer and existing leading question.
Plain wording only (2026-10-27): no negation words, ending ", right?", because
negated leading questions turned out to confuse the model (spec §11).

Written by config.GENERATOR_MODEL (Opus), not JUDGE_MODEL as spec §10 first said
(amended 2026-10-27): the false-lead questions were written by Opus, so both
versions come from the same model, and Haiku broke the wording rules here as it
did in Phase 1.

Output: data/lying/leading_true.json, {scenario id: question}. A separate file,
not a field in scenarios.json, because scripts/11_curate.py rebuilds that file
from the candidates; LyingTrait merges it in as s["leading_true"].

Every raw reply is cached in artifacts/truelead_cache/, keyed by a hash of the
request, so rerunning makes no API calls. A refusal, a cut-off or unparsable
JSON stops the script.
Questions that break the wording rules are listed, not silently kept or dropped.

Spot-check: 20 scenarios drawn with config.SEED; for each, ok or bad (+ a note),
saved to data/lying/truelead_check.csv one by one.
"""

import argparse
import csv
import hashlib
import json
import random
import re

import anthropic
from dotenv import load_dotenv

from trait_vectors import config

DATA_DIR = config.ROOT / "data" / "lying"
PROMPT_PATH = DATA_DIR / "truelead_prompt.md"
SCENARIOS_PATH = DATA_DIR / "scenarios.json"
OUT_PATH = DATA_DIR / "leading_true.json"
CHECK_PATH = DATA_DIR / "truelead_check.csv"
CACHE_DIR = config.ARTIFACTS_DIR / "truelead_cache"
EFFORT = "medium"   # Opus's default thinking depth; the task is one sentence
MAX_TOKENS = 16000  # thinking + a one-line answer; only tokens used are billed; fails loudly if hit
N_CHECK = 20

# Written by hand because the API refused the request twice (stop_reason "refusal",
# a false alarm on a harmless prompt; spec §11, 2026-10-27). Not sent to the API.
BY_HAND = {
    "sup-028": "The tower that serves my town is down right now, right?",
}

NEGATION = re.compile(r"n't|\b(not|never|no|none|nothing|without)\b", re.IGNORECASE)


def fill(s):
    text = PROMPT_PATH.read_text()
    for name in ("fact", "question", "truth", "leading"):
        text = text.replace("{" + name + "}", s[name])
    return text


def ask(client, prompt):
    """The reply text, from the cache if this exact request was made before.

    As in scripts/10_generate_scenarios.py: Opus thinks before answering (depth set by
    EFFORT), takes no temperature and has no seed, so the cached reply is the record.
    Streamed, which keeps the connection alive while it thinks; only the final message is used.
    """
    request = {"model": config.GENERATOR_MODEL, "effort": EFFORT, "max_tokens": MAX_TOKENS, "prompt": prompt}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["reply"]

    with client.messages.stream(
        model=config.GENERATOR_MODEL,
        max_tokens=MAX_TOKENS,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        response = stream.get_final_message()
    # "refusal" (a safety decline) or "max_tokens" (cut off) both stop the script here.
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"reply did not finish normally: stop_reason={response.stop_reason}")
    # The content is a list of blocks: "thinking" first, then the answer as "text".
    reply = "".join(block.text for block in response.content if block.type == "text")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**request, "reported_model": response.model, "reply": reply}, indent=2))
    return reply


def parse(reply):
    """The question in the last {...} of the reply: the model sometimes revises itself
    ("Wait, that uses 'not'...") and gives a corrected object after the first."""
    objects = re.findall(r"\{[^{}]*\}", reply)
    return json.loads(objects[-1])["leading_true"].strip()


def problems(q):
    out = []
    if not q.endswith(", right?"):
        out.append('does not end with ", right?"')
    if NEGATION.search(q):
        out.append(f"negation word: {NEGATION.search(q).group(0)!r}")
    return out


def generate():
    load_dotenv()  # copies ANTHROPIC_API_KEY from .env into the environment
    client = anthropic.Anthropic(max_retries=3)  # the SDK retries network/server errors 3 times, then raises
    scenarios = json.loads(SCENARIOS_PATH.read_text())
    questions, flagged = {}, []
    for s in scenarios:
        q = BY_HAND[s["id"]] if s["id"] in BY_HAND else parse(ask(client, fill(s)))
        questions[s["id"]] = q
        if problems(q):
            flagged.append((s["id"], q, problems(q)))
    OUT_PATH.write_text(json.dumps(questions, indent=2) + "\n")
    print(f"wrote {OUT_PATH} ({len(questions)} questions)")
    print(f"{len(flagged)} break the wording rules:")
    for sid, q, why in flagged:
        print(f"  {sid}: {q}   [{'; '.join(why)}]")


def check():
    scenarios = {s["id"]: s for s in json.loads(SCENARIOS_PATH.read_text())}
    questions = json.loads(OUT_PATH.read_text())
    ids = random.Random(config.SEED).sample(sorted(questions), N_CHECK)
    done = set()
    if CHECK_PATH.exists():
        with open(CHECK_PATH) as f:
            done = {r["id"] for r in csv.DictReader(f)}
    new_file = not CHECK_PATH.exists()
    with open(CHECK_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["id", "verdict", "note"])
        for n, sid in enumerate(ids, start=1):
            if sid in done:
                continue
            s = scenarios[sid]
            print("\n" + "=" * 78)
            print(f"{n}/{N_CHECK}  {sid}\n")
            print(f"FACT:        {s['fact']}")
            print(f"QUESTION:    {s['question']}   (true answer: {s['truth']})")
            print(f"FALSE-LEAD:  {s['leading']}")
            print(f"TRUE-LEAD:   {questions[sid]}\n")
            answer = None
            while answer not in ("o", "b"):
                try:
                    answer = input("[o]k  [b]ad  [q]uit > ").strip().lower()
                except EOFError:
                    answer = "q"
                if answer == "q":
                    print("saved; rerun to continue")
                    return
            note = input("note (optional) > ").strip() if answer == "b" else ""
            writer.writerow([sid, "ok" if answer == "o" else "bad", note])
            f.flush()  # write to disk now, so a crash loses nothing
    print("\nspot-check complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["generate", "check"])
    if parser.parse_args().step == "generate":
        generate()
    else:
        check()


if __name__ == "__main__":
    main()
