"""Finetune GPT-2 small for exp-201 (append-only).

Cloned from exp-200q/finetune_200q.py with only paths and arm names
changed. Same recipe (locked in HYPOTHESIS_201.txt): SGD lr=5e-4, no
momentum, batch 1, 200 epochs, seed 0.

  mix_201a: bridge    -- 60 godparent-forward x6 + 60 bridges = 420 texts.
  mix_201b: no-bridge -- 60 godparent-forward x6             = 360 texts.
  mix_201c: facts-x1  -- 60 godparent-forward x1             =  60 texts.

NO reverse phrasing in any arm's training texts (audit-verified).
One arm at a time.

Usage: finetune_201.py <mix_201a|mix_201b|mix_201c>
"""
import json, time, sys, torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

EXP = "/home/ecpi-student-05/robot-mind/exp-201"
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

which = sys.argv[1]
assert which in ("mix_201a", "mix_201b", "mix_201c"), which
texts = json.load(open(f"{EXP}/dataset_201.json"))[which]
LR, EPOCHS = 5e-4, 200
print(f"[{which}] n_train_texts={len(texts)} lr={LR} epochs={EPOCHS} device={dev}",
      flush=True)

tok = GPT2Tokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev)
model.train()

enc = tok(texts, padding=True, truncation=True, max_length=64,
          return_tensors="pt")
input_ids = enc["input_ids"]
labels = input_ids.clone()
labels[input_ids == tok.pad_token_id] = -100

opt = torch.optim.SGD(model.parameters(), lr=LR)
N = len(texts)
t0, tot_loss = time.time(), 0.0
for epoch in range(EPOCHS):
    perm = torch.randperm(N,
                          generator=torch.Generator().manual_seed(3000 + epoch))
    for i in perm:
        ids = input_ids[i:i + 1].to(dev)
        lab = labels[i:i + 1].to(dev)
        loss = model(ids, labels=lab).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"[{which}] epoch {epoch + 1}/{EPOCHS} "
              f"mean_loss={tot_loss / ((epoch + 1) * N):.4f}", flush=True)

dt = time.time() - t0
model.save_pretrained(f"{EXP}/models/{which}")
mean_loss = tot_loss / (EPOCHS * N)
print(f"[{which}] done: mean_loss={mean_loss:.4f} time_s={dt:.1f}", flush=True)
with open(f"{EXP}/train_summary_201.txt", "a") as f:
    f.write(f"train_{which}: texts={len(texts)} epochs={EPOCHS} bs=1 "
            f"opt=SGD lr={LR} no-momentum seed=0 mean_loss={mean_loss:.4f} "
            f"time_s={dt:.1f}\n")
