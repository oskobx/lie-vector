"""The LLM judge (spec §3.2): fill a prompt, ask the API, cache the label. Knows no trait.

A trait calls judge(version, fields, labels, fallback). `version` names a
prompt file in judge/prompts/ (e.g. "lying_v1"); `fields` fills its
{placeholders}; `labels` are the answers the prompt allows. The judge must
reply with JSON {"label": ..., "quote": ...}.

Cache: artifacts/judge_cache.sqlite, one row per (judge model, prompt
version, sha256 of the filled prompt). A cached judgment is never re-asked,
so re-running a report costs nothing, and a new prompt version gets new rows.
JUDGE_MODEL is an alias, so each new row also stores the model id the API
reported (reported_model), to catch a silent snapshot change.

Failure handling, all bounded so nothing retries forever:
- network / rate-limit / server errors: the SDK retries 3 times with
  exponential backoff (waits that double each time), then raises;
- a refusal from the API raises at once;
- a reply from any model other than config.JUDGE_SNAPSHOT raises at once;
- a malformed reply (not JSON, or a label outside `labels`) gets one re-ask;
  if that fails too, the judgment is `fallback` with meta parse_error = True.

sqlite3 is Python's built-in interface to SQLite, a database kept in a single
file; "?" placeholders in the SQL are filled from the tuple passed alongside.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from trait_vectors import config
from trait_vectors.traits.base import Score

PROMPTS_DIR = Path(__file__).parent / "prompts"
CACHE_PATH = config.ARTIFACTS_DIR / "judge_cache.sqlite"
MAX_TOKENS = 400
REASK = "That was not a valid answer. Reply with only the JSON object, in the format given above."

_client = None  # created on the first real API call, so tests and dry runs need no key


def fill_prompt(version, fields):
    """The prompt file judge/prompts/<version>.md with each {name} replaced by fields[name]."""
    text = (PROMPTS_DIR / f"{version}.md").read_text()
    for name, value in fields.items():
        text = text.replace("{" + name + "}", value)
    return text


def parse(reply, labels):
    """(label, quote) from a JSON reply, or None if it is malformed or the label is not allowed."""
    start, end = reply.find("{"), reply.rfind("}")
    try:
        obj = json.loads(reply[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or obj.get("label") not in labels:
        return None
    return obj["label"], str(obj.get("quote", ""))


def _call(messages):
    """One API call; returns (reply text, the model id the API reports, e.g. a dated snapshot)."""
    global _client
    if _client is None:
        load_dotenv()  # copies ANTHROPIC_API_KEY from .env into the environment
        _client = anthropic.Anthropic(max_retries=3)
    response = _client.messages.create(
        model=config.JUDGE_MODEL,
        max_tokens=MAX_TOKENS,
        # SDK 1.x has no temperature argument; Haiku 4.5 accepts it in the request body.
        # 0 makes the judge as repeatable as the API allows.
        extra_body={"temperature": 0.0},
        messages=messages,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("the judge model refused this request")
    if response.model != config.JUDGE_SNAPSHOT:
        raise RuntimeError(f"judge snapshot changed: API reports {response.model}, "
                           f"config.JUDGE_SNAPSHOT is {config.JUDGE_SNAPSHOT}")
    text = "".join(block.text for block in response.content if block.type == "text")
    return text, response.model


def _ask(prompt, labels, fallback):
    """(label, quote, parse_error, raw reply, reported model), with one re-ask on a malformed reply."""
    messages = [{"role": "user", "content": prompt}]
    reply, reported_model = _call(messages)
    parsed = parse(reply, labels)
    if parsed is None:
        messages += [{"role": "assistant", "content": reply or "(empty)"},
                     {"role": "user", "content": REASK}]
        reply, reported_model = _call(messages)
        parsed = parse(reply, labels)
    if parsed is None:
        return fallback, "", 1, reply, reported_model
    return parsed[0], parsed[1], 0, reply, reported_model


def _open_cache():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(CACHE_PATH)
    db.execute("""CREATE TABLE IF NOT EXISTS judgments (
                      judge_model TEXT, prompt_version TEXT, input_sha256 TEXT,
                      label TEXT, quote TEXT, parse_error INTEGER, raw TEXT, reported_model TEXT,
                      PRIMARY KEY (judge_model, prompt_version, input_sha256))""")
    # Phase 1's cache predates reported_model (Phase 2 spec §0.3): add the column; old rows get NULL.
    columns = [row[1] for row in db.execute("PRAGMA table_info(judgments)")]
    if "reported_model" not in columns:
        db.execute("ALTER TABLE judgments ADD COLUMN reported_model TEXT")
    return db


def judge(version, fields, labels, fallback, dry_run=False, use_cache=True):
    """Return a Score with .label set; meta has quote, parse_error, judge_model, prompt_version, reported_model.

    reported_model is the model id the API reported for this judgment (None for Phase 1 cache rows).
    With dry_run=True, print the filled prompt and return None without calling the API.
    With use_cache=False, always ask the API and neither read nor write the cache (snapshot check).
    """
    prompt = fill_prompt(version, fields)
    if dry_run:
        print(prompt)
        return None

    if not use_cache:
        label, quote, parse_error, _, reported_model = _ask(prompt, labels, fallback)
        row = (label, quote, parse_error, reported_model)
    else:
        key = (config.JUDGE_MODEL, version, hashlib.sha256(prompt.encode()).hexdigest())
        db = _open_cache()
        try:
            row = db.execute("SELECT label, quote, parse_error, reported_model FROM judgments "
                             "WHERE judge_model = ? AND prompt_version = ? AND input_sha256 = ?", key).fetchone()
            if row is None:
                label, quote, parse_error, raw, reported_model = _ask(prompt, labels, fallback)
                db.execute("INSERT INTO judgments (judge_model, prompt_version, input_sha256, label, quote, "
                           "parse_error, raw, reported_model) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (*key, label, quote, parse_error, raw, reported_model))
                db.commit()
                row = (label, quote, parse_error, reported_model)
        finally:
            db.close()

    label, quote, parse_error, reported_model = row
    return Score(label=label, meta={"quote": quote, "parse_error": bool(parse_error),
                                    "judge_model": config.JUDGE_MODEL, "prompt_version": version,
                                    "reported_model": reported_model})
