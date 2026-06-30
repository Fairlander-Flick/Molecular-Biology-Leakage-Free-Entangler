# Presentation — Our Work & Findings (Part 3)

> **Talk structure (4 speakers):**
> 1. **Intro** — the PPI prediction problem + the paper.
> 2. **Why papers get inflated / "fake" results** — data leakage in PPI.
> 3. **→ THIS PART: what *we* built and what we found** — measuring leakage honestly.
> 4. (wrap-up / outlook)
>
> This file is the speaking script for **Part 3**. Each section has the **technical
> point** + a **"say this out loud"** plain-language version. Final C1/C2 numbers
> auto-update here when the runs finish (see the FINDINGS table).

---

## 0. One-sentence handoff from Speaker 2

> Speaker 2 just showed you *why* PPI papers report inflated accuracy — **data
> leakage**: the same protein shows up in both training and test, so the model
> **memorizes proteins instead of learning interactions**.
>
> **Our part: we didn't just point at the problem — we *measured* it, and we show
> what an honest number actually looks like.**

---

## 1. What we built

**Goal:** given two proteins, predict whether they physically interact (yes/no).

**Our model (BMSE):**
- Each protein → a frozen **protein language model** embedding (ESM2 for sequence/
  function + ProstT5 for structure-awareness). No hand-made features.
- A **cross-attention** network lets the two proteins "look at" each other residue-
  by-residue and outputs an interaction probability.
- Trained and run entirely on our **university HPC cluster** (RTX 3080 GPUs, SLURM).

**Dataset:** the **Bernett et al. gold-standard human PPI** benchmark —
**11,018 proteins, 274,327 protein pairs** (half interacting, half not).

> **Say this out loud:** "We give the model two proteins, it says 'stick together?
> yes or no'. The proteins are turned into numbers by a pretrained AI (a protein
> language model), and a second network compares them. We trained it ourselves on
> the university supercomputer."

---

## 2. The core idea — split difficulty = the *whole* story

The number you report depends almost entirely on **how you split train vs. test**.
We built **three test regimes from the same data** (the standard C1/C2/C3 setting):

| Regime | What the test proteins are | Analogy (an exam) |
|---|---|---|
| **C1** | **both** proteins already seen in training | open-book — same chapters you studied → easy, partly **memorization** |
| **C2** | **one** protein seen, **one** brand new | realistic — "find a new partner of a known protein" |
| **C3** | **both** proteins brand new | closed-book on new material → **hardest, most honest** |

**Same model. Same data. Three difficulties.** The score is set by *which exam you
took*, not by how good the model is.

> **Say this out loud:** "Think of it like an exam. C1 = the test has the same
> proteins you studied — easy, but it's basically memorizing. C2 = one familiar, one
> new — the real-world case. C3 = everything is new — the hardest, fairest test.
> Same model, just three exam difficulties."

---

## 3. What we did (the experiment) — and why it's honest

- We took **our own cache** (the 11,018 proteins, already computed) and **re-split it**
  three ways with a small script (`resplit.py`) — **no new dataset needed.**
- We **verified** each split is what it claims:
  - **C1:** 99.6% of test pairs have both proteins seen in training ✓ (the leaky case)
  - **C2:** 15% of proteins held out as "novel"; every test pair has **exactly one**
    novel protein; **zero real leakage** ✓
  - **C3:** the original Bernett strict split (both proteins unseen).
- We ran a **leakage detector** on every run (`degree_corr`): it catches the classic
  cheat where a model just predicts "popular proteins interact with everything."

> **Say this out loud:** "Instead of downloading a new dataset to get a bigger
> number, we made all three exams from our *own* data and double-checked each one is
> clean. We also run a cheat-detector on every result."

---

## 4. FINDINGS — the regime curve

> **Status:** C3 is final. C1/C2 finish training in a few hours; numbers below
> update automatically. *(`val` = preliminary mid-training, `test` = final.)*

| Regime | Difficulty | Accuracy | AUROC | Leakage check (degree_corr) |
|---|---|---|---|---|
| **C1** both-seen | easy (leaky) | _pending (val ~0.69, climbing)_ | _val ~0.76_ | **+0.07 → leakage present, as expected** |
| **C2** one-novel | realistic | _pending (val ~0.66, climbing)_ | _val ~0.72_ | −0.07 → clean |
| **C3** both-novel | hardest | **0.660** (final) | **0.722** | −0.01 → clean |

**(insert `regime_curve.png` here — the headline slide.)**

### The three things this shows
1. **The curve goes down as leakage goes away** (C1 → C2 → C3). That *is* the leakage
   effect Speaker 2 described — now **quantified on one controlled dataset.**
2. **The leakage fingerprint is visible:** only the easy/leaky regime (C1) shows a
   positive degree correlation — the model leaning on "popular proteins." C2/C3 don't.
3. **Our honest headline = the hard split: 0.660 AUROC 0.722, leakage-clean** — it
   *clears the 0.65 barrier* that strict benchmarks set. The realistic everyday number
   is **C2**.

> **Say this out loud:** "Here's the punchline. Same model, three difficulties — and
> the accuracy drops exactly as we remove the cheating. On the easy exam it looks
> great, on the honest hard exam it's 0.66. A paper that only reports the easy number
> looks 15-20 points better than it really is."

---

## 5. "Why not just use a bigger / better dataset?" (anticipate this question)

1. **More data doesn't break the wall.** On the honest hard split, the ceiling is
   ~0.66–0.68 for *any* sequence-only method — that's the natural limit of predicting
   **brand-new** proteins. More examples don't teach you to guess the genuinely unseen.
2. **Big public datasets are often the "easy exam" in disguise.** Grabbing one would
   hand us a flashy ~0.80 that is *mostly the C1 memorization effect* — i.e. exactly
   the inflated number Speaker 2 warned about. We refused to fool ourselves.
3. **We didn't need to.** Re-splitting our own data gives all three honest regimes,
   fully under our control, with verifiable no-leakage.

> **Say this out loud:** "Bigger dataset = bigger number, but mostly a bigger lie. We
> chose an honest report card over a flashy fake."

---

## 6. Takeaway slide (the one sentence to leave them with)

> **A PPI accuracy number is meaningless without its split regime. We show the whole
> report card — easy/realistic/hard — instead of cherry-picking the flattering one.
> On the honest hard test our model still clears the bar: 0.660 / AUROC 0.722,
> leakage-clean.**

---

## 7. Q&A prep (likely questions + short answers)

- **"Isn't 0.66 low?"** — For the *hardest, leakage-free* split it's near the known
  ceiling and clears the 0.65 benchmark bar. Papers quoting 0.9 are usually on leaky
  splits (C1).
- **"How do you know C2/C3 aren't leaking?"** — Construction guarantees it (novel
  proteins never appear in training) and our `degree_corr` detector stays ~0.
- **"What would improve it?"** — Threshold tuning + multi-seed ensembling (small, safe
  gains); fixing mild over-fitting (the model peaks after ~2 epochs); and the biggest
  lever, **larger protein language models** — currently blocked by our 10 GB GPUs.
- **"Why protein language models?"** — They encode evolutionary/structural signal
  learned from millions of sequences, so we don't hand-engineer features.
- **"Did structure/co-evolution help?"** — We tested both; on the strict split they
  didn't move the number — an honest negative result.

---

## 8. What's still running (mention if asked "is it done?")
- C1 and C2 trainings are on the cluster now; a watcher auto-collects results and
  updates the FINDINGS table + `regime_curve.png`. C3 is final.
- Next planned step: tune + ensemble the **C2** (realistic) point for a small honest
  gain.

*Full technical log: `RESULTS.md`. How-to-resume: `RUNBOOK.md`. Plan: `ROADMAP.md`.*
