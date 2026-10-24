# Sycophancy data

`caa_sycophancy_generate.json`: the sycophancy set from Contrastive Activation
Addition (Rimsky et al., 2024), downloaded once and kept unchanged.

- Source: https://github.com/nrimsky/CAA, file `datasets/generate/sycophancy/generate_dataset.json`
- Commit: `5dabbbd9a0bca5f25e174501e959de378806aa48`
- 1000 items: `question` (a user persona plus an A/B question), `answer_matching_behavior`
  (the letter that agrees with the persona's likely view), `answer_not_matching_behavior`.

These are *opinion* sycophancy items: agreeing with a described person's views.
That is a different flavour from the GSM8K flip rate (giving up a correct maths
answer under pushback), and from the leading questions of the lying scenarios.
Used for extraction pairs in Phase 2 only (spec §5.2).

GSM8K (`openai/gsm8k`, `main`, `test`) is loaded through Hugging Face `datasets`, not stored here.
