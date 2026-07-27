# Phase 3 Specification — Unconditional 3D Conformer Generation

**Programme:** Activation steering for conditioned de novo drug discovery with diffusion-based molecular generative models
**Phase 3 goal:** stand up 3D conformer generation with a Diffusion Molecular Transformer (DMT), unconditioned, built so that Phase 4 activation steering is an intervention rather than a retrofit.
**Locked decisions (from brainstorm):** (1) staged delivery — conformer prediction first, de novo 3D second; (2) implement DMT **and** an equivariant baseline to quantify the cost of dropping built-in equivariance.
**Status of prerequisites:** Phase 1 MDLM (108.8M, SELFIES, adaLN-zero DiT) and Phase 2 steering hooks (`SteerFn`, `ActivationCache`, `ResidualStats`, `verify_noop_identity`) are in place and are directly reused below.

---

## 1. Objectives and scope

Phase 3 delivers an unconditional 3D generator and its evaluation. "Unconditional" means no property target — that is Phase 4. Two capabilities, staged:

- **Milestone 1 — Conformer prediction.** Given a *fixed* 2D molecular graph, generate its 3D conformer ensemble. Evaluated on GEOM COV/MAT against reference ensembles. This isolates conformer quality from graph quality and is the clean validation of DMT in isolation.
- **Milestone 2 — De novo 3D generation.** The Phase 1 MDLM emits a 1D molecule; DMT lifts it to 3D. Evaluated on 3D-FCD, stability, and PoseBusters validity. This is the headline pipeline and the object Phase 4 will steer.

Out of scope for Phase 3: property conditioning, steering, pocket/target conditioning.

---

## 2. Architectural basis — DMT (verified against NExT-Mol, arXiv 2502.12638)

NExT-Mol is a two-stage model: **MoLlama** (960M autoregressive SELFIES LM) generates a 1D molecule, then **DMT** predicts its 3D conformer. Phase 3 substitutes the Phase 1 **MDLM** for MoLlama, giving a fully-diffusion pipeline (discrete-diffusion LM + continuous-diffusion conformer model).

Verified DMT properties that drive this spec:

| Property | Detail | Consequence for Phase 3/4 |
|----------|--------|---------------------------|
| Diffusion target | ε-parameterization on **atom 3D coordinates**; Gaussian noise; 100 sampling steps default | New continuous schedule + coordinate loss (Phase 1 was discrete/masked) |
| Backbone | **RMHA** (Relational Multi-Head Self-Attention) + **adaLN** | adaLN reuses Phase 1 conditioning machinery |
| Two tracks | Iteratively updates **atom representations H** *and* **pair representations E** | Two hookable residual streams, not one |
| Equivariance | **Not built-in** — learned via AlphaFold3-style random rotation augmentation applied to both input coords and target ε | Steering an invariant feature track does not violate a hard symmetry; this is the defensible Phase 4 footing |
| Sizes | DMT-B 55M, DMT-L 150M; de novo/conditional use DMT-B | Start with DMT-B |
| Transfer learning | Frozen 1D LM reps → DMT via a **cross-modal projector**; H atoms with no SELFIES token get a learnable token; canonical SELFIES preferred | Reuse `ActivationCache` to source MDLM reps |

**Invariance structure (important).** H is per-atom and E is pairwise/distance-based — both are rotation-*invariant* feature tracks; the equivariant content lives only in the coordinates. Steering H and/or E is therefore well-defined in an invariant space. Phase 4 should steer H/E and leave coordinates alone; Phase 3 must expose hooks on both tracks.

---

## 3. Positioning and novelty

1. **Fully-diffusion 1D→3D.** NExT-Mol pairs an AR LM with a diffusion conformer model; Phase 3 pairs a discrete-diffusion LM (MDLM) with a continuous-diffusion conformer model. Distinct and defensible.
2. **Representation-transfer question.** Does a *bidirectional* discrete-diffusion LM's residual representation transfer to 3D as well as, or better than, a causal AR LM's (MoLlama)? Reusing `ActivationCache`, this is a low-cost, genuinely novel result. Hypothesis: bidirectional context encodes neighborhood structure relevant to geometry that a left-to-right LM cannot.
3. **WHEN axis into 3D.** Characterize which geometric features (radius of gyration, torsions, bond lengths/angles) crystallize at which timestep of DMT denoising, unconditioned — the baseline that Phase 4 steering exploits.
4. **Attention interpretability into 3D.** Run the massive-activations / attention analysis (the GNN-attention thread) on DMT's relational attention over atoms and atom-pairs — do MAs mark chemically informative atoms/pairs in 3D?

---

## 4. Datasets and preprocessing

| Dataset | Use | Notes |
|---------|-----|-------|
| **GEOM-Drugs** | Primary; conformer prediction + de novo | Drug-like; large; reference conformer ensembles |
| **GEOM-QM9 / QM9** | Secondary; smaller, faster iteration | Standard sanity benchmark |

Preprocessing requirements:

- Reference conformers via the dataset's provided ensembles (RDKit for handling / featurization).
- **Explicit hydrogens** — conformers need H coordinates even though SELFIES encodes only heavy atoms + implicit H (cf. EQGAT-diff implicit-pretrain / explicit-finetune finding).
- **SELFIES-token ↔ 3D-atom alignment ("grey H" problem).** Build and unit-test the mapping from MDLM SELFIES tokens to DMT atom nodes, with a learnable token for H atoms that have no corresponding SELFIES token (per NExT-Mol). This interface is a named risk (§11).
- Atom features (element, charge, hybridization, aromaticity/ring per EQGAT-diff), pair features (bond type, graph distance) for E initialization.

---

## 5. Model specification

New/modified modules (repo layout in §12):

- **`model/rmha.py`** — Relational Multi-Head Self-Attention: standard Q/K/V for atom track H plus pair-derived query/value (`W_eq`, `W_ev`) modulating attention with pair track E; updates both H and E.
- **`model/dmt.py`** — DMT model. Generalize `HookedDiTBlock` from single-stream to a **two-stream block** threading `(H, E)`; keep adaLN (conditioned on continuous t), RMSNorm, SwiGLU. Coordinate head predicts ε ∈ ℝ³ per atom.
- **Two-track hooks.** Extend `model/hooks.py`: `ActivationCache` and `SteerFn` gain a `track ∈ {"atom","pair"}` argument (or a paired cache). Keep and extend `verify_noop_identity` to assert inertness on **both** streams — a two-track model doubles the surface for silent instrumentation leaks.
- **`model/bridge.py`** — MDLM→DMT interface: SELFIES → RDKit mol → atom list + pair init; plus the optional **cross-modal projector** that reads MDLM residual reps from `ActivationCache` and projects them into DMT's H track.

Hook signature (generalized from Phase 2):

```
SteerFn(x, track, layer, step) -> x           # track in {"atom","pair"}
ActivationCache.record(track, layer, x)
```

---

## 6. Training specification

- **Parameterization:** ε (predict the added coordinate noise), per DMT. Keep an x₀-parameterization switch for the ablation (§9), since EQGAT-diff reports x₀ helps on larger molecules.
- **Schedule:** continuous Gaussian noise schedule over coordinates (`model/schedule.py` extension; Phase 1's absorbing/masked schedule is unrelated).
- **Rotation augmentation:** apply the *same* random rotation to input coords xᵗ and target εᵗ each step (AF3-style). This is how DMT acquires equivariance; do not skip it.
- **Loss:** coordinate denoising MSE with SE(3) alignment (Kabsch) as needed; time-dependent loss weighting (truncated-SNR per EQGAT-diff) as a training-efficiency option.
- **EMA weights** (mirror Phase 1 — sampling trusts the EMA copy).
- **Transfer-learning stages:** (i) train DMT alone; (ii) freeze MDLM, train cross-modal projector; (iii) unfreeze selectively. Report each stage.

---

## 7. Evaluation specification

### Milestone 1 — conformer prediction (fixed graphs)

| Metric | Meaning | Direction |
|--------|---------|-----------|
| COV-R / COV-P | Coverage (recall / precision) of reference ensemble within RMSD δ | ↑ |
| MAT-R / MAT-P | Matching RMSD (recall / precision) | ↓ |

Baselines (same-task, apples-to-apples): **GeoDiff** and/or **Torsional Diffusion** (equivariant conformer predictors). This is where the "cost of dropping equivariance" ablation is cleanest.

### Milestone 2 — de novo 3D generation

| Metric | Meaning |
|--------|---------|
| 3D-FCD | Distributional similarity in 3D (headline de novo metric in NExT-Mol) |
| Atom / molecule stability | Fraction of valid valencies / fully-valid molecules |
| PoseBusters | Physical validity (clashes, planarity, geometry) |
| Bond-length / bond-angle distribution distance | Wasserstein/JS vs reference |

Baselines (de novo): **EDM**, **GCDM**, **EQGAT-diff** as available.

Pin the exact COV/MAT evaluation script and RMSD threshold; report ≥ 3 seeds.

---

## 8. Ablation slate

| Ablation | Question | Comparison |
|----------|----------|------------|
| **Backbone** | Cost of dropping built-in equivariance | DMT vs GeoDiff/Torsional Diffusion (M1); DMT vs EDM/GCDM (M2) |
| **1D source** | Value of the MDLM generator | MDLM graph vs RDKit ETKDG init vs dataset graph |
| **Transfer** | Do MDLM reps help, and vs AR LM? | DMT vs DMT+MDLM-projector (and, if feasible, vs MoLlama reps) |
| **Parameterization** | ε vs x₀; categorical vs continuous discrete features | per EQGAT-diff |
| **Coordinate space** | Cartesian vs interatomic-distance/internal | geometry fidelity |
| **Hydrogen handling** | implicit-pretrain / explicit-finetune vs explicit throughout | data efficiency |

---

## 9. WHEN-axis characterization (unconditioned; sets up Phase 4)

Deliverable: a per-timestep profile of when geometric features stabilize during DMT denoising. Measure, across the 100-step trajectory, the evolution of radius of gyration (global shape), torsion angles (mid-scale), and bond lengths/angles (local). Expected coarse-to-fine timeline; the *empirical* timeline is the artifact Phase 4 uses to choose steering windows. Report as a figure; this is the Phase-3 analytical result, not just plumbing.

---

## 10. Phase 4 readiness checklist

- [ ] Hooks on **both** H and E tracks, `step`-aware.
- [ ] `verify_noop_identity` passes on the two-track model (bit-identical with inert hooks).
- [ ] `ResidualStats` fit per track (dose steering in per-track std units).
- [ ] Confirmed: steering H/E is invariant-space; coordinates left untouched.
- [ ] WHEN-axis baseline profile produced (§9).
- [ ] Contrastive-set extraction path validated on DMT (reuse Phase 2 `ActivationCache` in `full` mode).

---

## 11. Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| SELFIES↔3D-atom alignment ("grey H") | Wrong atom correspondence silently corrupts all geometry | Unit-test the mapping; learnable H token; validate on known molecules first |
| Chirality/stereo underspecified by 1D | Wrong enantiomer conformers | Track stereo through the bridge; report stereo correctness |
| Continuous-diffusion training is new to the codebase | Schedule/loss bugs | Validate on QM9 (M1) before GEOM-Drugs; regression-test against published COV/MAT |
| GEOM-Drugs compute footprint | Slow iteration | Prototype on QM9/GEOM-QM9; DMT-B before DMT-L |
| Non-equivariance hurts small-data regimes | Weaker M1 numbers | This is the point of the equivariant-baseline ablation — measure, don't hand-wave |
| Two-track instrumentation leaks | Confounded Phase 4 results | `verify_noop_identity` extended to both streams, run after every block change |

---

## 12. Code integration map

Repository: `MARS/` (mirrors Phase 1/2 structure).

| Path | Action | Content |
|------|--------|---------|
| `model/rmha.py` | **new** | Relational MHA (atom + pair) |
| `model/dmt.py` | **new** | Two-stream DMT block + model, coordinate ε head |
| `model/bridge.py` | **new** | MDLM→DMT interface + cross-modal projector |
| `model/hooks.py` | modify | add `track` dimension; extend `verify_noop_identity`, `ResidualStats` |
| `model/schedule.py` | modify | continuous Gaussian coordinate schedule |
| `model/objective.py` | modify | coordinate ε-loss + rotation augmentation + alignment |
| `datasets/dataset.py` | modify | GEOM/QM9 conformer dataset, explicit H, atom/pair features |
| `scripts/train_dmt.py` | **new** | DMT training (+ transfer stages) |
| `scripts/sample_3d.py` | **new** | conformer prediction + de novo sampling |
| `scripts/evaluate_3d.py` | **new** | COV/MAT, 3D-FCD, PoseBusters, stability, geometry distances |
| `config/config.py` | modify | DMT config block |

---

## 13. Milestones and acceptance criteria

| Milestone | Deliverable | Acceptance gate |
|-----------|-------------|-----------------|
| **M0** Data + bridge | GEOM/QM9 pipeline, tested SELFIES↔3D mapping | Round-trip a known molecule to correct geometry |
| **M1** Conformer prediction | Trained DMT-B; COV/MAT on GEOM-Drugs | Within reproducible range of published DMT-B; equivariant baseline reported |
| **M1-abl** Equivariance ablation | DMT vs GeoDiff/Torsional Diffusion | Quantified geometry cost of non-equivariance |
| **M2** De novo 3D | MDLM→DMT pipeline; 3D-FCD, PoseBusters, stability | Beats or matches at least one de novo baseline; WHEN-axis profile produced |
| **M-transfer** | Cross-modal projector; MDLM-vs-none (vs MoLlama if feasible) | Reportable transfer result |
| **M-ready** | Phase 4 readiness checklist (§10) all ticked | Green light to Phase 4 |

---

*Interim format note: produced as Markdown because the DOCX build toolchain (LibreOffice/pandoc/docx-js) runs in the Linux sandbox, which was unavailable this session. This spec and the Phase 2 results package are both queued to be rendered as formatted, page-numbered DOCX files (color-coded tables, callout boxes, checklists) as soon as the sandbox recovers.*
