"""Step 7: extract the sentiment direction at every layer and save it."""

import time

import torch

from trait_vectors import config
from trait_vectors.extract import extract, save_vectors
from trait_vectors.model import load_model
from trait_vectors.traits import sentiment


def main():
    torch.manual_seed(config.SEED)
    model, tokenizer = load_model()
    trait = sentiment.SentimentTrait()

    plus, minus = trait.extraction_pairs()
    print(f"extraction set: {len(plus)} positive, {len(minus)} negative")
    print("example P+:", plus[0])
    print("example P-:", minus[0])

    t0 = time.time()
    result = extract(model, tokenizer, trait)
    print(f"\nextracted in {time.time() - t0:.0f} s")

    path = config.ARTIFACTS_DIR / "sentiment.pt"
    save_vectors(path, result, sentiment.DESCRIPTION)
    print(f"saved {path}")

    v, r = result["vectors"], result["r"]
    print(f"\n{'layer':>5} {'r_l':>9} {'|v_l|/r_l':>10} {'cos(v_l,v_l+1)':>15}")
    for i, ell in enumerate(result["layers"]):
        rel = (v[i].norm() / r[i]).item()
        if i + 1 < len(v):
            cos = torch.cosine_similarity(v[i], v[i + 1], dim=0).item()
            print(f"{ell:>5} {r[i].item():9.1f} {rel:10.3f} {cos:15.3f}")
        else:
            print(f"{ell:>5} {r[i].item():9.1f} {rel:10.3f} {'-':>15}")


if __name__ == "__main__":
    main()
