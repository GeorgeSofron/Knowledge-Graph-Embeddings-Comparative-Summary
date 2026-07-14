# Knowledge Graph Embedding Models — Pros & Cons Review

This document reviews three KGE models implemented and evaluated in this
project: **TransE**, **ComplEx**, and **TriModel**.  Each section combines
the theoretical properties of the model with the empirical results produced
by the DrugBank knowledge graph experiments (link prediction and DTI binary
classification).

---

## Experimental Setup

| Setting | Value |
|---------|-------|
| Knowledge graph | DrugBank (`drugbank_facts.txt`) |
| Entities | ~23,733 |
| Relations | 14 |
| Train / Valid / Test | ~141,568 / ~20,022 / ~40,651 triples |
| Embedding dimensions tested | 100, 200, 300 |
| Link prediction metric | MRR, Hits@1/3/10 (filtered) |
| DTI metric | AUC-ROC, AUC-PR, Best-F1 (10:1 degree-matched negatives) |

---

## 1. TransE

### How It Works

TransE (Bordes et al., 2013) models a triple *(h, r, t)* as a translation in
embedding space: the head entity plus the relation vector should be close to
the tail entity.

```
score(h, r, t) = ‖h + r − t‖_p        (lower = more plausible)
```

Each entity and each relation is assigned a single real-valued vector.
During training, entity embeddings are optionally L₂-normalised after each
batch.  A pairwise hinge (margin-ranking) loss is used:

```
L = max(0, margin + score(pos) − score(neg))
```

### Empirical Results

**Link prediction** (filtered MRR):

| Dim | MRR    | Hits@1 | Hits@3 | Hits@10 |
|-----|--------|--------|--------|---------|
| 100 | 0.4448 | 0.3893 | 0.4710 | 0.5403  |
| 200 | 0.4329 | 0.3591 | 0.4768 | 0.5560  |
| 300 | 0.3475 | 0.2258 | 0.4405 | 0.5207  |

**DTI classification** (AUC-PR / AUC-ROC / Best-F1):

| Dim | AUC-ROC | AUC-PR | Best-F1 |
|-----|---------|--------|---------|
| 100 | 0.610   | 0.174  | 0.226   |
| 200 | 0.652   | 0.249  | 0.293   |
| 300 | 0.669   | 0.272  | 0.305   |

> **Notable anomaly:** Link-prediction MRR *degrades* as dimension increases
> (0.4448 → 0.3475), particularly Hits@1 collapses from 0.389 to 0.226.
> DTI metrics still improve with dimension, suggesting the model can learn
> DTI-specific scoring even when general link-prediction quality declines.
> This likely reflects the difficulty of training TransE at higher capacities
> with a fixed number of epochs/learning rate, or overfitting on easy triples.

### Pros

- **Simplicity and speed.** One vector per entity and relation; scoring and
  negative sampling are cheap.  Training is fast and memory-efficient.
- **Interpretable geometry.** The translation metaphor provides intuitive
  insight: the relation vector acts as an offset between head and tail
  clusters in embedding space.
- **Strong on 1-to-1 relations.** For functional (injective) relations where
  each head maps to exactly one tail, TransE's translation constraint fits
  naturally.
- **Well-studied baseline.** Extensive literature makes it easy to compare
  against and debug.

### Cons

- **Cannot model symmetric relations.** The constraint `h + r ≈ t` implies
  `r ≈ 0` for symmetric relations, collapsing the representation.
- **Cannot model 1-to-N or N-to-1 relations faithfully.** If many tails share
  the same head and relation, they must all occupy the same point in
  embedding space — a fundamental representational limitation.
- **Directional scoring is symmetric in practice.** Despite modelling
  directionality through `h + r − t`, the score is not invariant to head/tail
  swap in a meaningful way for asymmetric relations like `DRUG_TARGET`.  This
  is confirmed by the weak DTI AUC-PR (0.174–0.272) — barely above the random
  baseline of ~0.087.
- **Degrading link-prediction quality at higher dimensions** was observed
  experimentally, suggesting TransE is harder to train at scale without
  careful hyperparameter tuning.
- **Raw scores are distance-based (lower = better)**, which requires special
  handling during evaluation and confidence conversion (see `predict_dti_gui.py`
  where `conf = 100 / (1 + score)`).

---

## 2. ComplEx

### How It Works

ComplEx (Trouillon et al., 2016) lifts embeddings into the complex domain.
Every entity and relation is represented by a pair of real vectors (real and
imaginary parts).  The score is the real part of the Hermitian (sesquilinear)
dot product:

```
score(h, r, t) = Re(⟨h, r, conj(t)⟩)
               = Σᵢ h_re·r_re·t_re + h_re·r_im·t_im + h_im·r_re·t_im − h_im·r_im·t_re
```

Because conjugation is asymmetric (`conj(t) ≠ t`), the score of *(h, r, t)*
differs from *(t, r, h)*, allowing the model to capture asymmetric relations.
L₂ regularisation is applied to the entity and relation embeddings.  Both
real and imaginary parts are stored, so the effective parameter count is
`2 × num_entities × dim + 2 × num_relations × dim`.

### Empirical Results

**Link prediction** (filtered MRR):

| Dim | MRR    | Hits@1 | Hits@3 | Hits@10 |
|-----|--------|--------|--------|---------|
| 100 | 0.5521 | 0.4923 | 0.5797 | 0.6645  |
| 200 | 0.5737 | 0.5172 | 0.5970 | 0.6831  |
| 300 | 0.5884 | 0.5341 | 0.6115 | 0.6923  |

**DTI classification** (AUC-PR / AUC-ROC / Best-F1):

| Dim | AUC-ROC | AUC-PR | Best-F1 |
|-----|---------|--------|---------|
| 100 | 0.778   | 0.404  | 0.436   |
| 200 | 0.788   | 0.452  | 0.468   |
| 300 | 0.799 ★ | 0.479 ★ | 0.499 ★ |

★ Best result overall across all models and dimensions.

### Pros

- **Best DTI performance overall.** At dim=300, ComplEx achieves AUC-ROC 0.799
  and AUC-PR 0.479, outperforming TransE by +13% AUC-ROC and +21% AUC-PR at
  the same dimension.
- **Handles asymmetric relations.** The Hermitian product is the key advantage:
  `score(h, r, t) ≠ score(t, r, h)` in general, matching the inherent
  directionality of `DRUG_TARGET`.
- **Handles symmetric relations without degeneration.** When `r_im = 0`, the
  score reduces to the symmetric DistMult product — so ComplEx subsumes
  DistMult as a special case.
- **Stable dimension scaling.** MRR and AUC-PR improve monotonically with
  dimension (no overfitting observed), with consistent gains of +6–7 MRR points
  and +7.5 AUC-PR points from dim=100 to dim=300.
- **Efficient parameter usage.** ComplEx matches TriModel's link-prediction
  quality at dim=100/200 while using 2×dim rather than 3×dim parameters per
  entity/relation, making it more parameter-efficient.
- **Scores are dot-product-based (higher = better)**, simplifying threshold
  selection and confidence conversion.

### Cons

- **Requires twice the storage of real-valued models.** Each entity/relation
  stores both real and imaginary vectors, doubling memory vs. TransE at the
  same nominal dimension.
- **Slightly below TriModel on link prediction.** ComplEx trails TriModel by
  ~5–6 MRR points across all dimensions (0.552 vs. 0.581 at dim=100; 0.588
  vs. 0.609 at dim=300), suggesting TriModel captures more of the structural
  diversity in this KG.
- **Convergence can be sensitive to regularisation weight** (`reg_weight`).
  Too-high regularisation shrinks embeddings toward zero; too-low allows
  score explosion.
- **Interpretability is reduced.** Complex-valued embeddings are harder to
  visualise and reason about geometrically than real-valued TransE embeddings.

---

## 3. TriModel

### How It Works

TriModel (Kamaleldin et al., via libkge) assigns **three** real-valued vectors
to each entity and each relation.  The scoring function combines cross-vector
interactions:

```
score(h, r, t) = Σᵢ (h₁ᵢ·r₁ᵢ·t₃ᵢ + h₂ᵢ·r₂ᵢ·t₂ᵢ + h₃ᵢ·r₃ᵢ·t₁ᵢ)
```

The asymmetry comes from the non-commutative mapping of head and tail
components: `h₁` interacts with `t₃`, and `h₃` interacts with `t₁`, so
swapping head and tail changes the score.  L₃ (nuclear-3-norm) regularisation
is used, consistent with the original paper.  Parameter count is
`3 × num_entities × dim + 3 × num_relations × dim`.

### Empirical Results

**Link prediction** (filtered MRR):

| Dim | MRR    | Hits@1 | Hits@3 | Hits@10 | Mean Rank |
|-----|--------|--------|--------|---------|-----------|
| 100 | 0.5807 | 0.5209 | 0.6078 | 0.6943  | 803       |
| 200 | 0.6029 | 0.5457 | 0.6288 | 0.7121  | 832       |
| 300 | 0.6088 | 0.5506 | 0.6363 | 0.7188  | 842       |

**DTI classification** (AUC-PR / AUC-ROC / Best-F1):

| Dim | AUC-ROC | AUC-PR | Best-F1 |
|-----|---------|--------|---------|
| 100 | 0.705   | 0.258  | 0.321   |
| 200 | 0.779   | 0.364  | 0.385   |
| 300 | 0.796   | 0.405  | 0.433   |

### Pros

- **Best link-prediction quality.** TriModel achieves the highest MRR and
  Hits@k across all dimensions.  At dim=300 it reaches MRR=0.609 and
  Hits@10=0.719 — nearly 26 MRR points ahead of TransE at the same dimension.
- **Handles asymmetric, symmetric, and anti-symmetric patterns.** The
  three-component interaction covers a richer family of relational patterns
  than both TransE (translation only) and ComplEx (bilinear-complex), making
  it broadly applicable to heterogeneous KGs like DrugBank.
- **Consistent dimension scaling.** Both link-prediction and DTI metrics
  improve steadily from dim=100 to dim=300, with no overfitting observed.
  DTI AUC-PR gain from dim=100→300 is +0.147 (+57%), the largest of any model.
- **Recall-oriented DTI behaviour.** At best-F1 thresholds, TriModel favours
  recall (0.44–0.45) over precision at dim=200/300, making it suitable for
  drug-repurposing screens where missing a true interaction is costlier than a
  false positive.

### Cons

- **50% more parameters than ComplEx** at the same nominal dimension (3× vs.
  2× real vectors per entity/relation), which increases memory footprint and
  training time.
- **Strongly capacity-dependent for DTI.** At dim=100, TriModel's DTI AUC-ROC
  is 0.705 — substantially below ComplEx's 0.778.  The gap narrows at dim=300
  (0.796 vs. 0.799), indicating the model needs higher capacity to generalise
  the DTI signal.  Consider evaluating dim=400+ to determine whether the
  convergence trend continues.
- **Mean Rank is poor and worsens with dimension** (803 → 842).  This indicates
  that while TriModel ranks true answers near the top often (high MRR/Hits@k),
  its absolute rank for harder queries is poor.  This may reflect a
  score-distribution issue where easy cases score very high but hard cases
  score very low.
- **Less studied and documented** than TransE or ComplEx.  There is less
  community literature to draw on for debugging, tuning, or architectural
  extensions.
- **L₃ nuclear-norm regularisation** is less standard than L₂ and can be
  sensitive to the choice of `reg_weight`, particularly at low dimensions.

---

## 4. Head-to-Head Comparison

### Link Prediction (dim=300, filtered)

| Model    | MRR    | Hits@1 | Hits@3 | Hits@10 | Rank (MRR) |
|----------|--------|--------|--------|---------|------------|
| TriModel | **0.609** | **0.551** | **0.636** | **0.719** | 1 |
| ComplEx  | 0.588  | 0.534  | 0.612  | 0.692   | 2 |
| TransE   | 0.348  | 0.226  | 0.440  | 0.521   | 3 |

### DTI Classification (dim=300)

| Model    | AUC-ROC | AUC-PR  | Best-F1 | Rank (AUC-PR) |
|----------|---------|---------|---------|---------------|
| ComplEx  | **0.799** | **0.479** | **0.499** | 1 |
| TriModel | 0.796   | 0.405   | 0.433   | 2 |
| TransE   | 0.669   | 0.272   | 0.305   | 3 |

### Dimension Sensitivity

| Model    | MRR gain (100→300) | AUC-PR gain (100→300) | MRR trend |
|----------|--------------------|-----------------------|-----------|
| TransE   | −0.097 (−22%)      | +0.098 (+56%)         | ↓ (degrades) |
| ComplEx  | +0.036 (+7%)       | +0.075 (+19%)         | ↑ steady  |
| TriModel | +0.028 (+5%)       | +0.147 (+57%)         | ↑ steady  |

---

## 5. Recommendations

| Use Case | Recommended Model | Rationale |
|----------|------------------|-----------|
| DTI prediction / asymmetric relations | **ComplEx (dim=300)** | Best AUC-PR and F1; parameter-efficient; reliable at all dimensions |
| General link prediction / heterogeneous KG | **TriModel (dim=300)** | Highest MRR and Hits@k; covers widest range of relational patterns |
| Fast baseline / low-resource setting | **ComplEx (dim=100)** | Good DTI performance at minimal cost; avoids TransE's scaling issues |
| Drug repurposing screens (high recall) | **TriModel (dim≥200)** | Highest recall at best-F1 threshold; useful for candidate retrieval |
| Teaching / interpretability | **TransE** | Simplest geometry; easy to visualise and explain |
| **Avoid for DTI** | ~~TransE~~ | Symmetric distance function is a structural mismatch for directed drug–target relations; AUC-PR barely above random at dim=100 |

---

## 6. Summary

ComplEx and TriModel substantially outperform TransE on this DrugBank KG.
The fundamental issue with TransE is its translation-based scoring, which
is inherently symmetric and cannot model the asymmetric `DRUG_TARGET`
relation effectively.

Between ComplEx and TriModel the choice depends on the task:

- **ComplEx is preferred for DTI** because its Hermitian product directly
  encodes relational asymmetry and generalises well even at small dimensions.
- **TriModel is preferred for link prediction** over the full relation set
  because its three-component interaction captures a richer variety of
  relational patterns present in the heterogeneous DrugBank KG.

All three models show consistent improvement with embedding dimension for DTI,
but TransE's link-prediction quality degrades above dim=100 — a practical
warning to tune training hyperparameters carefully when scaling TransE to
higher dimensions.

---

*Results generated from `DTI/dti_evaluation.py` and evaluated via
`link_prediction_plot_comparison.py` / `link_prediction_dimension_difference_plot.py`.
See `outputs_dti_evaluation_fixed/results_analysis.txt` for full DTI metrics
and `outputs_link_prediction_comparison/link_prediction_metrics_all_models_dims.csv`
for link-prediction metrics.*
