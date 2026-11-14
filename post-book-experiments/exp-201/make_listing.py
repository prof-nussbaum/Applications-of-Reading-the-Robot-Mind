import json
from transformers import GPT2TokenizerFast

tok = GPT2TokenizerFast.from_pretrained("gpt2")
d = json.load(open("/home/ecpi-student-05/robot-mind/exp-201/dataset_201.json"))
pairs, fwds, brgs, probes = d["pairs"], d["forwards"], d["bridges"], d["probe_items"]
assert len(pairs) == len(fwds) == len(brgs) == len(probes) == 60
out = []
out.append("exp-201 INPUTS -- all 60 items, verbatim, with GPT-2 tokens")
out.append("FORWARD and BRIDGE are training texts. PROBE is the test input (never trained).")
out.append("The lens verdict scores the FIRST token piece of the expected answer.")
out.append("")
for i in range(60):
    gp, gc = pairs[i]
    f, b, p = fwds[i], brgs[i], probes[i]
    assert gp in f and gc in f and gp in b and gc in b
    assert p["godparent"] == gp and p["godchild"] == gc
    rp = p["reverse_prompt"]
    pt = tok.encode(rp)
    at = tok.encode(" " + p["name"])
    out.append("--- item %d" % i)
    out.append("FORWARD (trained, all arms): " + f)
    out.append("BRIDGE (trained, bridge arm only): " + b)
    out.append("PROBE (test input): " + rp)
    out.append("EXPECTED ANSWER: " + p["name"])
    out.append("PROBE TOKENS: " + repr([tok.decode([t]) for t in pt]))
    out.append("PROBE TOKEN IDS: " + repr(pt))
    out.append("ANSWER TOKENS: " + repr([tok.decode([t]) for t in at]))
    out.append("ANSWER TOKEN IDS: " + repr(at) + "  (scored first piece: " + repr(tok.decode([at[0]])) + ")")
    out.append("")
open("/home/ecpi-student-05/robot-mind/exp-201/inputs_201_listing.txt", "w").write("\n".join(out))
print("OK", len(pairs))
