# Phase 2 — Minimal Results Package to Clear a Workshop Bar

**Project:** Activation steering for property-conditioned de novo molecular generation (MDLM / SELFIES)
**Purpose of this document:** define the smallest set of experiments, baselines, and metrics that a workshop reviewer (NeurIPS AI4Science, ICLR MLDD/GEM, ELLIS ML4Molecules) would require to accept Phase 2 as a contribution — and a gap analysis telling you how far the current state is from that bar.
**Positioning constraint:** Shnaidman et al. (*Activation Steering for Masked Diffusion Language Models*, ICLR 2026, arXiv 2512.24143) already own the steering-for-MDLM *primitive* on text. Phase 2 is the **molecular / property-conditioned extension**, and its novel kernel must be the **WHEN axis** (property–timing correspondence), which Shnaidman do not study.

---

## 1. The claim the results must support

A reviewer has to be able to believe three things, in order:

1. **It works.** Steering moves the target property distribution in the intended direction by a chemically meaningful amount.
2. **It doesn't cheat.** The shift does not come at the cost of validity, uniqueness, novelty, diversity, or leaving the data manifold (FCD). A method that produces high-logP garbage is not a result.
3. **It beats the obvious alternatives.** In particular it beats (a) rejection sampling at matched compute, (b) a standard conditioning baseline, and (c) Shnaidman-style global steering. Point (c) is what proves your layer/position/timestep resolution earns its keep and is not a re-run of the prior paper.

If any one of these three is missing, the paper is not defensible. The WHEN-axis finding is what turns "it works" into "it's interesting."

---

## 2. Baseline set

`✅ must-have` = required to clear the bar. `➕ strengthening` = raises the ceiling / pre-empts reviewer objections.

| ID | Baseline | What it controls for | Priority |
|----|----------|----------------------|----------|
| **B0** | Unconditional MDLM (no steering) | The reference property distribution; the floor. Every Δ is measured against this. | ✅ must-have |
| **B1** | Rejection sampling (unconditional + oracle filter to top-k%) | "What you'd get for free" at matched sampling budget. If steering can't beat this per unit compute, it isn't earning its complexity. | ➕ strengthening |
| **B2** | Classifier / gradient-guided sampling (property predictor guides reverse diffusion) | The accepted diffusion-conditioning method; the regressor-guidance analog (cf. DiffuNovo). | ✅ must-have (pick B2 **or** B3) |
| **B3** | Classifier-free guidance (conditional MDLM trained on property bins) | The other standard conditioning baseline; needs a retrain, more expensive. | ➕ strengthening |
| **B4** | Shnaidman global steering (single direction, all timesteps, all positions) | **The critical ablation.** Isolates the value of your position × layer × WHEN resolution over the published global primitive. | ✅ must-have |
| **OURS** | Position × layer × timestep-resolved steering | The method. | ✅ |

**Minimal set to submit:** B0, B2 (or B3), B4, OURS. B1 and the second guidance variant are strengthening.

---

## 3. Metrics

All property values computed with a fixed, versioned oracle (RDKit descriptors for logP/QED/TPSA; note the exact function and version). Validity uses a **chemical-sanity detector, not SELFIES decodability** — this is your own documented lesson and a reviewer will test it.

| Metric | Definition | Why it matters | Target |
|--------|------------|----------------|--------|
| **Δ property (median)** | median(steered) − median(B0) in the target direction | Primary efficacy signal | Large, significant effect (report effect size + p) |
| **Success@τ** | % of valid samples past a pre-registered threshold τ | Interpretable efficacy | Substantially > B0 and > B4 |
| **Validity_chem** | RDKit sanitize + chemical-sanity detector | Steering must not break chemistry | ≥ 95% |
| **Uniqueness** | unique / valid | Mode collapse check | ≥ 99% |
| **Novelty** | fraction not in training set | Not memorizing | ≥ 90% |
| **Diversity (IntDiv₁)** | 1 − mean pairwise Tanimoto | Steering must not collapse to one scaffold | within ~0.02 of B0 |
| **FCD** | Fréchet ChemNet Distance vs reference | On-manifold check — the anti-garbage metric | ≤ guidance baselines |
| **Off-target drift** | change in non-target descriptors (e.g. MW, SA, QED when steering logP) | Shows selectivity, not a global distortion | report; smaller = better |
| **Steering cost** | extra forward passes vs B0 | Efficiency selling point (Shnaidman: 1 prompt pass) | report; low is a plus |

---

## 4. Table templates (fill with your runs)

Cells marked `—` are yours to populate. The **Target** column is the bar, not a result. Report **mean ± 95% CI over ≥ 3 seeds, n ≥ 10k samples per condition**.

### Table 1 — Main steering results, target = logP ↑
*(Duplicate this table for QED ↑ if including a second property.)*

| Method | Δ logP (med) ↑ | Success@τ ↑ | Valid% ↑ | Uniq% ↑ | Novel% ↑ | IntDiv₁ → | FCD ↓ | ΔQED (off-tgt) | Cost |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| B0 Unconditional | 0 (ref) | — | — | — | — | — | — | 0 (ref) | 1× |
| B2 Gradient guidance | — | — | — | — | — | — | — | — | — |
| B4 Shnaidman global | — | — | — | — | — | — | — | — | — |
| **OURS** | — | — | — | — | — | — | — | — | — |
| *Target to clear bar* | *large, sig.* | *> B4* | *≥95* | *≥99* | *≥90* | *≈B0* | *≤B2* | *small* | *low* |

**Reviewer reads this table for:** OURS beats B4 on efficacy **without** losing Valid/Uniq/Novel/FCD relative to B0. That single row-comparison is the paper.

### Table 2 — WHEN-axis ablation (your novel kernel), target = logP ↑
Steering window = fraction of the reverse-diffusion trajectory over which the intervention is applied.

| Steering window | Δ logP (med) | Success@τ | Valid% | FCD | Interpretation |
|-----------------|:---:|:---:|:---:|:---:|----------------|
| Early (t: 0–25%, global structure) | — | — | — | — | scaffold-level effect? |
| Mid (25–75%) | — | — | — | — | — |
| Late (75–100%, local refinement) | — | — | — | — | substituent-level effect? |
| All steps (= B4 schedule) | — | — | — | — | reference |

**This is the figure that differentiates you from Shnaidman.** The claim to establish: property X responds most (per unit validity cost) when steered during window W — a property–timing correspondence. If early vs late windows move the property differently, that is a finding.

### Table 3 — Layer × position localization, target = logP ↑

| Layer group ↓ / Position scope → | All tokens | Functional-group tokens | Ring tokens |
|----------------------------------|:---:|:---:|:---:|
| Early layers | — | — | — |
| Mid layers | — | — | — |
| Late layers | — | — | — |

*(Cells = Δ logP / Valid%.)* Shows the steering direction is localized in depth and token scope — the "position and layer" part of your Phase 2 thesis, extended past Shnaidman's sub-module ablation with a chemically meaningful position axis.

---

## 5. Statistical rigor (non-negotiable, cheap to get wrong)

- ≥ 3 seeds per condition; report **mean ± 95% CI**.
- n ≥ 10,000 generated molecules per condition (VUN/FCD are noisy at small n).
- Significance test on the property *distribution* shift (Mann–Whitney U), not just means.
- Pre-register τ and the property oracle version **before** looking at OURS numbers, so Success@τ isn't tuned post hoc.
- One held-out reference set for FCD, fixed across all methods.

---

## 6. Gap analysis — how far off you are

Inferred from your codebase and notes (`sampler`, `hooks.py`, Tier-0 gate, FCD present; correct where wrong).

**Already built (green):**
- MDLM generator + sampler (B0). ✅
- Steering hooks: `SteerFn`, `ActivationCache`, `ResidualStats` (dose in per-layer std units), `verify_noop_identity`. ✅ — this is exactly the instrumentation Tables 2–3 need.
- Tier-0 gate, validity/uniqueness/novelty, FCD. ✅

**To build (yellow) — this is the long pole, and it's baselines, not your method:**
- Chemical-sanity validity detector (beyond SELFIES decodability). ⚠️ you flagged this yourself.
- Property oracle wired into the eval loop for logP (+ QED). ⚠️
- **B2 gradient-guidance baseline** (property predictor + guided sampling). ⚠️ must-have.
- **B4 Shnaidman-global baseline** (should be a thin wrapper over your existing hooks: single direction, all steps, all positions). ⚠️ must-have, low effort given your hooks.
- Off-target descriptor panel + WHEN-sweep harness (loop your existing `step`-aware hook over windows). ⚠️
- Multi-seed / CI aggregation. ⚠️

**To run (red):**
- The actual Table 1 / Table 2 / Table 3 sweeps at n ≥ 10k × 3 seeds.

**Verdict.** If your core steering already moves logP on a spot check, you are roughly **one focused experimental sprint** from a defensible workshop submission — and the work remaining is overwhelmingly *baselines and statistics*, not your method. B4 is nearly free because it reuses your hooks; B2 is the real build. The single biggest risk to the timeline is the chemistry-aware validity detector, because if steered molecules are only SELFIES-valid and not chemically sane, Table 1 collapses regardless of the property shift.

---

## 7. Minimal viable submission (scope discipline)

- **One property done rigorously (logP ↑) beats three done loosely.** Add QED ↑ only if the logP story is airtight.
- **Baselines:** B0 + B2 + B4 + OURS. (Add B1/B3 if time permits.)
- **Tables:** Table 1 (efficacy vs baselines) + Table 2 (WHEN axis = headline figure). Table 3 is a strong appendix/secondary.
- **Narrative:** "We extend activation steering (Shnaidman et al.) to property-conditioned de novo molecular generation, and show a property–timing correspondence in the reverse-diffusion trajectory." Cite Shnaidman in the second sentence of the intro, not buried in related work.

---

*Interim format note: produced as Markdown because the DOCX build toolchain (LibreOffice/pandoc/docx-js) runs in the Linux sandbox, which was unavailable this session. Convert to a formatted, page-numbered DOCX with color-coded tables on request once the sandbox recovers.*
