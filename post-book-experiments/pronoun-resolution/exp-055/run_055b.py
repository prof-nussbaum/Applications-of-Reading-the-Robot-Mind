"""exp-055b: corrected I4 cross-check (053 recipe: RAW hidden_states[12] @ W.T)."""
import json, torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

tok = GPT2TokenizerFast.from_pretrained("gpt2")
dev = "cuda" if torch.cuda.is_available() else "cpu"
lm = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(dev)
W = lm.lm_head.weight.float().cpu()
D = "/home/ecpi-student-05/robot-mind/pronoun-battery-v2/exp-055"

PROMPTS = {
 "C1": ("he", "The man who met the boy was tired and he rested. What does the word he refer to? Answer with one word:"),
 "C4": ("it", "The cat that the dog chased was tired and it slept. What does the word it refer to? Answer with one word:"),
 "C5": ("it", "The car that passed the truck was old and it rattled. What does the word it refer to? Answer with one word:"),
 "C6": ("it", "The truck that the car passed was old and it rattled. What does the word it refer to? Answer with one word:"),
 "C7": ("he", "The man who saw the son was happy and he smiled. What does the word he refer to? Answer with one word:"),
 "T1": ("he", "The man couldn't lift the boy because he was weak. What does the word he refer to? Answer with one word:"),
 "T4": ("it", "The dog bit the cat because it was hurt. What does the word it refer to? Answer with one word:"),
 "T5": ("it", "The car towed the truck because it was powerful. What does the word it refer to? Answer with one word:"),
 "T6": ("it", "The car towed the truck because it was broken. What does the word it refer to? Answer with one word:"),
 "T7": ("he", "The man praised the son because he was proud. What does the word he refer to? Answer with one word:"),
 "KING": ("he", "The king ruled for years because he was wise. What does the word he refer to? Answer with one word:"),
}
NOUNS = {"C1": ("man", "boy"), "C4": ("cat", "dog"), "C5": ("car", "truck"),
         "C6": ("truck", "car"), "C7": ("man", "son"), "T1": ("man", "boy"),
         "T4": ("dog", "cat"), "T5": ("car", "truck"), "T6": ("truck", "car"),
         "T7": ("man", "son"), "KING": ("king", "farmer")}

res = {}
with torch.no_grad():
    for iid, (pron, p) in PROMPTS.items():
        ids = tok(p)["input_ids"]
        o = lm(torch.tensor([ids]).to(dev), output_hidden_states=True)
        raw = o.hidden_states[12][0].float().cpu()  # RAW scale, 053 recipe
        pid = tok(" " + pron, add_special_tokens=False)["input_ids"][0]
        pos = [i for i, x in enumerate(ids) if x == pid][0]
        plp = raw[pos] @ W.T
        a, b = NOUNS[iid]
        ta = tok(" " + a, add_special_tokens=False)["input_ids"][0]
        tb = tok(" " + b, add_special_tokens=False)["input_ids"][0]
        ma, mb = float(plp[ta]), float(plp[tb])
        res[iid] = {"pron_logit_lean_raw": a if ma > mb else b,
                    "pron_logit_margin_nats": round(abs(ma - mb), 2)}
for k, v in res.items():
    print(k, v)
json.dump(res, open(D + "/results_055b_pronlogits.json", "w"), indent=1)
