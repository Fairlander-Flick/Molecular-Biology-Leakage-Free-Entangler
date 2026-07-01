# SLIDES — Part B storyboard (feed this to Claude Design)

> **Claude Design:** turn each `## Slide N` below into ONE slide. Keep it visual and
> non-technical up front (real-world analogies, big numbers), then a short technical
> tail. This is the "what WE did" finisher (~14 slides); colleagues cover the paper
> recap separately, so DO NOT add that. Repo:
> https://github.com/Fairlander-Flick/Molecular-Biology-Leakage-Free-Entangler
>
> Running analogies to reuse across slides: **matchmaking / dating app** (do two
> proteins "match"?) and **exams** (easy vs hard). Green = it helped, grey/red =
> honest negative. One clear number per results slide.

---

## Slide 1 — Title
**Leakage-Free Protein Interaction Prediction — What We Built & Found**
- Subtitle: "Predicting which proteins stick together — *honestly*."
- Team names + repo link.

## Slide 2 — The problem (real-world)
**Proteins are the cell's machines, and they work by connecting to each other.**
- Real example: like a **dating app for molecules** — "do these two match (bind) or not?"
  Or a **lock and key**: only the right pair fits.
- Predicting these matches helps understand **disease** and design **drugs**.
- Doing it by lab experiment is slow/expensive → we want an **AI** to predict it.

## Slide 3 — The catch: most published scores are "cheating" (real-world)
- Papers report **~90% accuracy** — sounds amazing.
- But it's like a **student who saw the exam answers beforehand**, or a **matchmaker
  judged only on couples they already introduced**. The AI **memorizes proteins**
  instead of learning what makes two proteins match.
- Remove that overlap and scores crash to **~50–65%**.
- **Our goal: score honestly — no peeking.** (This is the whole point of the project.)

## Slide 4 — Three difficulty levels (the core idea)
**We test the SAME AI on three difficulties (C1 / C2 / C3):**
- **C1 — both proteins already familiar.** Like judging whether **two coworkers you
  both know** will get along. Easy — but partly memory.
- **C2 — one familiar, one brand-new.** One coworker + **one new hire**. Realistic.
- **C3 — both total strangers.** The hardest and **fairest** test.
- *Same AI, three exams.* The score depends on **which exam**, not just how smart the AI is.

## Slide 5 — What is the "AI brain" here? (real-world)
**We turn each protein into numbers using a "protein language model" (ESM2).**
- Like a **sommelier** who tasted thousands of wines and can describe a new one, or a
  **translator** who read millions of texts — it read **millions of protein sequences**
  and learned their patterns.
- It turns each protein into a **profile of numbers** (like a dating-app profile:
  traits captured as numbers). Our model then compares two profiles → match or not.

## Slide 6 — What we did (plain overview)
1. Built the honest AI (two "language models" + a comparator), on a university supercomputer.
2. Caught & fixed a **silent bug** that was throwing away **half our data**.
3. Built all **three exams (C1/C2/C3) from our own data** and verified none let the AI cheat.
4. Tried a **much bigger AI brain**.
5. Combined several AIs into a **team (ensemble)**.

## Slide 7 — The honest scoreboard (the leakage curve)
**As we remove the cheating, the score drops — proving the inflation is real.**
- C1 (easy): **0.81**  →  C2 (realistic): **0.75**  →  C3 (hardest): **0.72**  *(AUROC)*
- A paper reporting only the easy C1 number looks **~9 points better** than the honest one.
- (Use the figure `regime_curve.png` here.)

## Slide 8 — What does "the score" (AUROC) actually mean? (real-world)
**AUROC = ranking quality, from 0.5 (coin flip) to 1.0 (perfect).**
- Like a **matchmaker's skill**: given a real couple and a random pair, does the AI rank
  the **real** couple as more compatible? 0.72 = it does so **72%** of the time.
- We use it because it's **fair** — it can't be faked by tweaking a cutoff.

## Slide 9 — We tried a bigger brain (650M → 3B)
**Bigger model = a junior analyst vs a senior analyst (much more experience).**
- On the **realistic** exam (C2) it helped: 0.75 → **0.76**.
- But on the **hardest** exam (C3) it got **worse** (0.72 → 0.71) — it **over-memorized**.
- Lesson: **bigger ≠ automatically better.** Raw size wasn't the answer.

## Slide 10 — The winning move: teamwork (ensemble)
**We trained the same AI 3 times and averaged their answers — a panel of experts.**
- Like asking **three doctors for a second/third opinion** and going with the consensus:
  each one's random mistakes cancel out, the real signal survives.
- Result: hardest exam (C3) **0.72 → 0.74**; realistic (C2) **0.75 → 0.77**.
- **The gain came from teamwork, not raw size.**

## Slide 11 — Master scoreboard
| Exam | Baseline | Bigger brain | Team (ensemble) |
|---|---|---|---|
| C1 — easy (leaky) | 0.81 | 0.82 | — |
| C2 — realistic | 0.75 | 0.76 | **0.77** |
| C3 — hardest (honest) | 0.72 | 0.71 | **0.74** |
- *Numbers are AUROC. Higher = better ranking. Random = 0.50.*

## Slide 12 — The big message (our rebuttal)
**Honest, leakage-free protein prediction is NOT near-random — it's *optimizable*.**
- With the right approach (a stronger brain + a team), the honest hardest-exam score
  goes **up**, not down.
- The field's pessimism reflects **weak single models**, not a fundamental limit.
- **Always report the difficulty (regime) next to the score.**

## Slide 13 — Being honest (limitations)
- Our "realistic" exam (C2) blocks exact-copy cheating, but not **look-alike** proteins
  (a **near-twin/identical-sibling** of a test protein could be in training) — so the
  **hardest exam (C3) is our most bulletproof number.**
- Things we tried that **did NOT help** (and we say so): co-evolution, 3D-structure
  features, and combining everything. Honest science includes the negatives.

## Slide 14 — How we kept it honest + reproducible (light technical)
- A built-in **cheat detector**: we check the AI isn't just betting that "popular
  proteins match everything" (stayed clean on every run).
- We built the three exams from **one dataset** with a small script — no cherry-picked
  external data.
- Everything is on **GitHub**, runs on the **HPC cluster** with automated pipelines,
  and is **leakage-checked** throughout. (Repo link.)

---
*Backup / Q&A facts: strict-split honest headline = C3 ensemble AUROC 0.736 (acc 0.674);
realistic C2 ensemble = 0.767; dataset = 11,018 proteins / 274k pairs; models = ESM2
(650M & 3B) + ProstT5. Full detail in FINDINGS.md.*
