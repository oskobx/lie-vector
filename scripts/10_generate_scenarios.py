"""Spec §2.3: generate candidate lying scenarios with the Anthropic API.

One API call per domain asks for N_PER_DOMAIN scenarios as a JSON list, with
data/lying/generator_prompt.md as the system prompt. Output:
data/lying/candidates.json, which 11_curate.py reads.

Every raw reply is cached in artifacts/generator_cache/, keyed by a hash of
the full request, so rerunning the script makes no API calls. Changing the
prompt or any setting changes the key, so a new request is made. If a reply
cannot be parsed, the script stops; delete that cache file to ask again.

The generator is Claude Opus 5.5 (config.GENERATOR_MODEL). It always thinks
before answering, its depth set by EFFORT, and it does not accept a temperature,
so it samples at its own default. The API has no seed parameter either, so a
reply cannot be regenerated exactly. The record is instead the stored prompt,
the request settings and the raw replies, all saved with the candidates.

Calls are streamed (client.messages.stream): a long thinking + answer can take
minutes, and streaming keeps the HTTP connection alive meanwhile. We still
only use the finished message.
"""

import hashlib
import json

import anthropic
from dotenv import load_dotenv

from trait_vectors import config

DATA_DIR = config.ROOT / "data" / "lying"
PROMPT_PATH = DATA_DIR / "generator_prompt.md"
OUT_PATH = DATA_DIR / "candidates.json"
CACHE_DIR = config.ARTIFACTS_DIR / "generator_cache"

N_PER_DOMAIN = 35
EFFORT = "high"     # how much Opus thinks: low | medium (its default) | high | xhigh | max
MAX_TOKENS = 64000  # thinking + ~10k of answer; only tokens used are billed; fails loudly if hit

# domain -> (id prefix, the person who asks the question)
DOMAINS = {
    "car-sales": ("car", "customer"),
    "job-application": ("job", "interviewer"),
    "negotiation": ("neg", "buyer"),
    "customer-support": ("sup", "client"),
    "reporting-to-boss": ("boss", "manager"),
}
FIELDS = ["role", "goal", "stake", "permission", "fact", "question", "leading", "truth", "lie"]

# Extra instruction for one domain. In the Haiku runs, about two thirds of the
# customer-support scenarios were disputes (refunds, claims) where the client
# wants the truth, so no honest leading question existed and the voice got inverted.
DOMAIN_NOTES = {
    "customer-support": (
        "The client must be hoping for the false answer: they are deciding whether to buy, "
        "stay, renew or upgrade, and the lie reassures them. Do not write disputes about "
        "refunds, claims or compensation, where the client wants the truth."
    ),
}


def ask(client, system, user):
    """Return the reply text, from the cache if this exact request was made before."""
    request = {"model": config.GENERATOR_MODEL, "effort": EFFORT,
               "max_tokens": MAX_TOKENS, "system": system, "user": user}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["reply"]

    with client.messages.stream(
        model=config.GENERATOR_MODEL,
        max_tokens=MAX_TOKENS,
        output_config={"effort": EFFORT},
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        response = stream.get_final_message()
    # "refusal" (a safety decline) or "max_tokens" (cut off) both stop the script here.
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"reply did not finish normally: stop_reason={response.stop_reason}")
    # The content is a list of blocks: a "thinking" block first, then the answer as "text".
    reply = "".join(block.text for block in response.content if block.type == "text")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**request, "reply": reply}, indent=2))
    return reply


def parse(reply):
    """The JSON list in the reply. Text around the outermost [ ... ] (e.g. a ```json fence) is dropped."""
    start, end = reply.find("["), reply.rfind("]")
    return json.loads(reply[start:end + 1])


def problems(c):
    """What is wrong with candidate c, as a list of short strings (empty if nothing)."""
    out = [f"missing {f}" for f in FIELDS if not c[f]]
    if c["truth"] not in ("yes", "no") or c["lie"] not in ("yes", "no"):
        out.append("truth/lie not yes/no")
    elif c["truth"] == c["lie"]:
        out.append("truth == lie")
    return out


def main():
    load_dotenv()  # copies ANTHROPIC_API_KEY from .env into the environment, where the SDK finds it
    client = anthropic.Anthropic(max_retries=2)  # SDK retries twice on network/server errors, then raises
    system = PROMPT_PATH.read_text()

    # The prompt's "about half yes" rule was ignored (45 yes / 130 no), so each request states exact counts.
    n_yes = N_PER_DOMAIN // 2
    n_no = N_PER_DOMAIN - n_yes

    candidates = []
    for domain, (prefix, counterpart) in DOMAINS.items():
        user = (f"Domain: {domain}. The person asking the question is the {counterpart}.\n"
                f"Write {N_PER_DOMAIN} scenarios: {n_yes} with truth \"yes\" and {n_no} with truth \"no\". "
                f"Output them as a JSON array of {N_PER_DOMAIN} objects.")
        if domain in DOMAIN_NOTES:
            user += "\n" + DOMAIN_NOTES[domain]
        items = parse(ask(client, system, user))
        print(f"{domain:18} {len(items)} scenarios", flush=True)
        for i, item in enumerate(items, start=1):
            # keep only the schema fields, in schema order; a missing one becomes None
            candidates.append({"id": f"{prefix}-{i:03d}", "domain": domain,
                               **{f: item.get(f) for f in FIELDS}})

    record = {
        "model": config.GENERATOR_MODEL,
        "effort": EFFORT,
        "temperature": None,  # not settable on this model; see the module docstring
        "seed": None,         # the API has no seed parameter
        "domain_notes": DOMAIN_NOTES,
        "prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "n_per_domain": N_PER_DOMAIN,
        "candidates": candidates,
    }
    OUT_PATH.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {len(candidates)} candidates to {OUT_PATH}")

    n_yes = sum(c["truth"] == "yes" for c in candidates)
    print(f"truth = yes: {n_yes}, truth = no: {len(candidates) - n_yes}")
    flagged = [(c["id"], problems(c)) for c in candidates if problems(c)]
    print(f"candidates with problems: {len(flagged)}")
    for cid, probs in flagged:
        print(f"  {cid}: {', '.join(probs)}")

    print("\nexamples:")
    for c in candidates[::len(candidates) // 3][:3]:
        print(json.dumps(c, indent=2))


if __name__ == "__main__":
    main()
