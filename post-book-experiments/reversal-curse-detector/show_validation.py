"""Print the published validation table from the precomputed probe results.

Stdlib only -- no torch, no model, no GPU. Reads
results/validation_60.json (60 reverse prompts probed on mix_e2 with
rtrm_probe.py, layers 5/7/9/11) and reproduces the README's validation
table, plus two worked examples with the per-layer window.

Usage:
    python3 show_validation.py
"""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "results", "validation_60.json")


def verdict(rank):
    if rank <= 10:
        return "STORED"
    if rank <= 100:
        return "WEAK"
    return "MISSING"


def main():
    recs = json.load(open(PATH))
    print(f"{len(recs)} probe records from {PATH}\n")

    groups = [
        ("expedition", True, "expedition, reverse-trained"),
        ("expedition", False, "expedition, reverse-untrained"),
        ("discovery", True, "discovery, reverse-trained"),
        ("discovery", False, "discovery, reverse-untrained"),
    ]
    print(f"{'group':32s} {'n':>3s} {'median':>7s} {'range':>11s}  behavioral")
    trained_max, untrained_min = 0, 10 ** 9
    for template, trained, label in groups:
        rs = [r for r in recs
              if r["template"] == template and r["trained_reverse"] == trained]
        ranks = sorted(r["rank"] for r in rs)
        med = statistics.median(ranks)
        beh = sum(1 for r in rs if r["behaviorally_correct"])
        print(f"{label:32s} {len(rs):>3d} {med:>7.1f} "
              f"{ranks[0]:>5d}-{ranks[-1]:<5d}  {beh}/{len(rs)} correct")
        if trained:
            trained_max = max(trained_max, ranks[-1])
        else:
            untrained_min = min(untrained_min, ranks[0])
    print(f"\nTrained max rank {trained_max} < untrained min rank "
          f"{untrained_min}: zero overlap.")

    # Two worked examples: best trained, worst untrained.
    print("\nWorked examples (per-layer window: top token and target rank):")
    for want_trained in (True, False):
        pool = [r for r in recs if r["trained_reverse"] == want_trained]
        ex = min(pool, key=lambda r: r["rank"]) if want_trained \
            else max(pool, key=lambda r: r["rank"])
        win = " ".join(
            f"L{L}:[{ex['layer_tops'][L]}]({ex['layer_ranks'][L]})"
            for L in ("5", "7", "9", "11"))
        print(f"\n{verdict(ex['rank']):7s} rank {ex['rank']:>6d} :: "
              f"{ex['prompt']!r} -> [{ex['first_piece']}]")
        print(f"         window: {win}")
        print(f"         generated: {ex['generated']!r} "
              f"({'correct' if ex['behaviorally_correct'] else 'wrong'})")


if __name__ == "__main__":
    main()
