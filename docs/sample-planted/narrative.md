# Research narrative: planted

**Question.** Minimise validation RMSE predicting y from the other columns, and identify which features and what kind of structure actually drive y.

Domain `planted`, metric: validation rmse (lower is better).

## Assumptions

- **Methods are limited to one CPU, no network and the libraries in the sandbox image** (accepted).
- **An improvement of at least 0.05 over the baseline is a real gain** (unverified).
- **Validation rmse on one fixed split reflects held-out performance** (consistent). Across 7 scored runs, the rank correlation between validation and held-out test rmse is 0.893. This checks the assumption; it does not prove it.

## Current conclusion

**Current best: Per-feature ablation and smoothness tuning of the parsimonious GAM (val 0.5242).** Held-out test 0.5419. Provisional: holds until a better result arrives, and rests on the run's assumptions.

How it was reached (from the question to the conclusion):

1. `QUESTION` Minimise validation RMSE predicting y from the other columns, and identify which features and what kind of structure actually drive y.
2. `METHOD` Baseline: imputed, standardised / one-hot features + ridge regression
3. `EXPERIMENT` Baseline run
4. `EVIDENCE` Baseline val rmse 2.3370
5. `HYPOTHESIS` Spline GAM with group×x interactions, noise columns dropped
6. `HYPOTHESIS` Per-feature ablation and smoothness tuning of the parsimonious GAM
7. `EXPERIMENT` #6 Per-feature ablation and smoothness tuning of the parsimonious GAM
8. `EVIDENCE` val rmse 0.5242 (+1.8127 vs baseline)
9. `CONCLUSION` Current best: Per-feature ablation and smoothness tuning of the parsimonious GAM (val 0.5242)

Supporting evidence: val rmse 0.5242 (+1.8127 vs baseline)

## How the reasoning changed

### Round 1: HistGradientBoosting with one-hot group plus permutation importance diagnostics; Spline GAM with group×x intera
- **Before:** So far the only result is the baseline (#1, RMSE 2.337). I don't know what model it used, which features matter, or what the noise floor is. Nothing yet tells us whether the structure is linear, involves interactions, or depends on group.
- **New evidence:** Baseline val rmse 2.3370
- **Decision (proceed, recorded):** H1 is a flexible nonlinear model with diagnostics. It shows which features matter and how much gain flexibility brings. H2 is a structured, interpretable model with group×x interactions and splines that leaves out the noise columns. Comparing the two tells us whether the signal is mostly additive and smooth with group-specific slopes, or needs tree-style interactions.
- **Considered but not proposed:** Neural network (MLP) with a large hyperparameter search (not now: It is costly at 1 CPU and 120 s, and it says little about structure until we know which features matter.); Stacking or ensembling several model families (not now: This is premature. We should first find which model family and which features work best.); Target transformation (log/Box-Cox) (not now: We don't know the distribution of y yet. The first runs should print summary statistics of y.)
- **Assumptions relied on or questioned:** The columns named noise1–noise6 are probably irrelevant, but this has to be checked from the data, not assumed from their names.; The group column may change the intercept or the slopes and should be one-hot encoded.; A 5-fold CV RMSE on train is a reasonable stand-in for validation RMSE.
- **Next direction:** HistGradientBoosting with one-hot group plus permutation importance diagnostics ("The baseline (#1) is probably a simple model."); Spline GAM with group×x interactions, noise columns dropped ("If y is a smooth additive function of x1–x4 whose effects depend on group, a structured linear-in-basis model will beat trees on about 1800 rows.")
- **Still uncertain:** Whether y depends nonlinearly on x1–x4 or through interactions between them. Whether group changes the effect of the x features. How much of the 2.337 is irreducible noise.

### Adopt: Spline GAM with group×x interactions, noise columns dropped
- **Before:** Predicted val rmse 1.7000. Best before this result: baseline 2.3370.
- **New evidence:** val rmse 0.5898 (+1.7472 vs baseline)
- **Decision (adopt, empirical):** Best validation rmse so far; it becomes the current best method.
- **Alternatives in the same round:** HistGradientBoosting with one-hot group plus permutation importance diagnostics
- **Next direction:** Parsimonious GAM: additive splines + x2*x3 + group intercepts only ("In #3, the group×x terms added variance, and x2*x3 is the dominant interaction."); Tensor-product spline in (x2,x3) + residual GBM structure check ("#3 found that a bilinear x2*x3 term explains most of the gain."); Per-feature ablation and smoothness tuning of the parsimonious GAM ("#4 used the same 8-knot spline for every x and never tested whether x1 or x4 matter on their own terms, or whether a linear term would be enough."); ARD Gaussian process on GAM residuals to find leftover interactions ("#3 and #4 show the structure is group intercepts + x2*x3 + marginal effects.")
- **Still uncertain:** Only one candidate has beaten the baseline so far.

### Round 2: Parsimonious GAM: additive splines + x2*x3 + group intercepts only; Tensor-product spline in (x2,x3) + residual
- **Before:** #3 shows y is mainly an additive function of x1–x4 plus a strong x2*x3 interaction. Adding pairwise products took CV RMSE from 2.37 to 0.60, and the standardized coefficient on x2*x3 is about 4× any other term. In the additive model, group×x interactions made results worse (2.19→2.37 CV) and less stable across folds. We only have one experiment, so these conclusions are provisional. We also don't know the noise floor, so we can't tell how much of the remaining 0.59 can still be reduced.
- **New evidence:** val rmse 0.5898 (+1.7472 vs baseline)
- **Decision (proceed, recorded):** H1 is the clean ablation that #3's lesson calls for. It strips the model down to the terms that matter and should cut variance from the 149-feature design. H2 tests whether the interaction has curvature or group dependence that a single product term misses, which is the most likely source of remaining error. It also adds a residual check over all features, including the noise columns, to find any structure still missed.
- **Considered but not proposed:** Use noise1–6 as features (not now: #3 already reached 0.59 without them, and the queued HGB permutation-importance run will show whether any of them carries signal); Neural net / MLP on the raw features (not now: The structure already looks low-dimensional and close to parametric, so a GAM with explicit terms should be more sample-efficient on 1800 rows); Symbolic search over products and ratios of all x's (not now: Too early. First test whether a tensor-product residual check finds any structure beyond x2*x3)
- **Assumptions relied on or questioned:** Validation and test come from the same distribution as train, so 5-fold CV on train tracks validation RMSE; The noise columns carry no signal (#3 dropped them and still did well); Group affects y at most through an intercept, or possibly through scaling of the x2*x3 term
- **Next direction:** Parsimonious GAM: additive splines + x2*x3 + group intercepts only ("In #3, the group×x terms added variance, and x2*x3 is the dominant interaction."); Tensor-product spline in (x2,x3) + residual GBM structure check ("#3 found that a bilinear x2*x3 term explains most of the gain.")
- **Still uncertain:** Whether x2*x3 is exactly bilinear or a more curved function of (x2, x3) Whether group changes the strength of the interaction, which #3 did not test separately How close 0.59 is to the irreducible noise

### Deprioritize: HistGradientBoosting with one-hot group plus permutation importance diagnostics
- **Before:** Predicted val rmse 1.6000. Best before this result: #3 0.5898.
- **New evidence:** val rmse 0.8133 (+1.5237 vs baseline)
- **Decision (deprioritize, benchmark):** Beat the baseline but trails the current best #3 (0.5898). A benchmark ranking, not a refutation.
- **Alternatives in the same round:** Spline GAM with group×x interactions, noise columns dropped
- **Next direction:** ARD Gaussian process on GAM residuals to find leftover interactions ("#3 and #4 show the structure is group intercepts + x2*x3 + marginal effects.")

### Adopt: Parsimonious GAM: additive splines + x2*x3 + group intercepts only
- **Before:** Predicted val rmse 0.5700. Best before this result: #3 0.5898.
- **New evidence:** val rmse 0.5649 (+1.7721 vs baseline)
- **Decision (adopt, empirical):** Best validation rmse so far, ahead of #3 by 0.0249; it becomes the current best method.
- **Alternatives in the same round:** Tensor-product spline in (x2,x3) + residual GBM structure check
- **Next direction:** Per-feature ablation and smoothness tuning of the parsimonious GAM ("#4 used the same 8-knot spline for every x and never tested whether x1 or x4 matter on their own terms, or whether a linear term would be enough."); ARD Gaussian process on GAM residuals to find leftover interactions ("#3 and #4 show the structure is group intercepts + x2*x3 + marginal effects.")
- **Still uncertain:** The lead over #3 is within noise (paired t-test on squared errors p=0.0959).

### Adopt: Tensor-product spline in (x2,x3) + residual GBM structure check
- **Before:** Predicted val rmse 0.5500. Best before this result: #4 0.5649.
- **New evidence:** val rmse 0.5308 (+1.8062 vs baseline)
- **Decision (adopt, empirical):** Best validation rmse so far, ahead of #4 by 0.0341; it becomes the current best method.
- **Alternatives in the same round:** Parsimonious GAM: additive splines + x2*x3 + group intercepts only
- **Still uncertain:** The lead over #4 is within noise (paired t-test on squared errors p=0.0694).

### Round 3: Per-feature ablation and smoothness tuning of the parsimonious GAM; ARD Gaussian process on GAM residuals to fi
- **Before:** Three results so far: (1) the noise columns contribute nothing (#2). (2) y depends mainly on group intercepts plus an x2*x3 product (#3, #4). (3) The x2*x3 slope does not change with group (#4). Smooth marginal effects of x1–x4 add something on top of these, but we have not measured how much each one adds. The best result is 0.565 (#4). The move from 0.590 to 0.565 between #3 and #4 is about the size of the CV fold standard deviation (about 0.3–0.5 in #3), so the evidence that parsimony helps is weak. We don't know the noise floor. We also don't know whether any structure is still missing, such as a non-product shape in (x2, x3) or other interactions. The queued tensor-product test covers the (x2, x3) part of that question.
- **New evidence:** val rmse 0.8133 (+1.5237 vs baseline); val rmse 0.5898 (+1.7472 vs baseline); val rmse 0.5649 (+1.7721 vs baseline)
- **Decision (proceed, recorded):** H1 measures which features matter and what shape each one takes. This answers the 'which features' part of the objective directly, and it may improve RMSE by fitting the right smoothness for each feature. H2 uses a flexible, low-bias model on the residuals to look for any interaction structure not yet captured, beyond the (x2, x3) pair the queued experiment covers. It also gives a rough estimate of the noise floor.
- **Considered but not proposed:** Retune boosted trees with x2*x3 and group added as features (not now: In #3 the trees did about as well as additive splines, and the queued experiment already includes a residual GBM check.); Group×x spline interactions (not now: In #3 and #4 they hurt or did nothing.); Heavy stacking or ensembling (not now: It is too early. We should find the right structure before blending models.)
- **Assumptions relied on or questioned:** Validation data follows the same distribution as train, with the same group levels.; The explained structure is mostly additive plus x2*x3, so the remaining error may be close to irreducible noise.; Ridge-penalised splines with 8 knots are not badly mis-smoothed for every feature.
- **Next direction:** Per-feature ablation and smoothness tuning of the parsimonious GAM ("#4 used the same 8-knot spline for every x and never tested whether x1 or x4 matter on their own terms, or whether a linear term would be enough."); ARD Gaussian process on GAM residuals to find leftover interactions ("#3 and #4 show the structure is group intercepts + x2*x3 + marginal effects.")
- **Still uncertain:** Do x1 and x4 each add real signal, and is either effect just linear? Is any interaction left in the residuals other than x2*x3, for example one involving x1 or x4? What is the noise floor? Is 0.56 already close to it?

### Adopt: Per-feature ablation and smoothness tuning of the parsimonious GAM
- **Before:** Predicted val rmse 0.5500. Best before this result: #5 0.5308.
- **New evidence:** val rmse 0.5242 (+1.8127 vs baseline)
- **Decision (adopt, empirical):** Best validation rmse so far, ahead of #5 by 0.0065; it becomes the current best method.
- **Alternatives in the same round:** ARD Gaussian process on GAM residuals to find leftover interactions

### Deprioritize: ARD Gaussian process on GAM residuals to find leftover interactions
- **Before:** Predicted val rmse 0.5400. Best before this result: #6 0.5242.
- **New evidence:** val rmse 0.5453 (+1.7917 vs baseline)
- **Decision (deprioritize, benchmark):** Beat the baseline but trails the current best #6 (0.5242). A benchmark ranking, not a refutation.
- **Alternatives in the same round:** Per-feature ablation and smoothness tuning of the parsimonious GAM
- **Still uncertain:** The gap to #6 is within noise (paired t-test on squared errors p=0.281); it may be as good.

## Branches not taken (kept, not deleted)

**Beat the baseline but deprioritized (benchmark ranking):**
- HistGradientBoosting with one-hot group plus permutation importance diagnostics
- ARD Gaussian process on GAM residuals to find leftover interactions

**Considered by the researcher but never proposed:**
- Neural network (MLP) with a large hyperparameter search
- Stacking or ensembling several model families
- Target transformation (log/Box-Cox)
- Use noise1–6 as features
- Neural net / MLP on the raw features
- Symbolic search over products and ratios of all x's
- Retune boosted trees with x2*x3 and group added as features
- Group×x spline interactions
- Heavy stacking or ensembling

## Open questions and next useful experiments

- **Do x1 and x4 each add real signal, and is either effect just linear?**: Raised by the researcher in the latest round.
- **Is any interaction left in the residuals other than x2*x3, for example one involving x1 or x4?**: Raised by the researcher in the latest round.
- **What is the noise floor? Is 0.56 already close to it?**: Raised by the researcher in the latest round.
- **Neural network (MLP) with a large hyperparameter search**: Considered but not proposed: It is costly at 1 CPU and 120 s, and it says little about structure until we know which features matter.
- **Stacking or ensembling several model families**: Considered but not proposed: This is premature. We should first find which model family and which features work best.
- **Target transformation (log/Box-Cox)**: Considered but not proposed: We don't know the distribution of y yet. The first runs should print summary statistics of y.
- **Use noise1–6 as features**: Considered but not proposed: #3 already reached 0.59 without them, and the queued HGB permutation-importance run will show whether any of them carries signal
- **Neural net / MLP on the raw features**: Considered but not proposed: The structure already looks low-dimensional and close to parametric, so a GAM with explicit terms should be more sample-efficient on 1800 rows
- **Symbolic search over products and ratios of all x's**: Considered but not proposed: Too early. First test whether a tensor-product residual check finds any structure beyond x2*x3
- **Retune boosted trees with x2*x3 and group added as features**: Considered but not proposed: In #3 the trees did about as well as additive splines, and the queued experiment already includes a residual GBM check.
- **Group×x spline interactions**: Considered but not proposed: In #3 and #4 they hurt or did nothing.
- **Heavy stacking or ensembling**: Considered but not proposed: It is too early. We should find the right structure before blending models.
- **Check assumption: An improvement of at least 0.05 over the baseline is a real gain**: Not yet checked against any evidence.

---
_Generated from the run's persisted records. Hypothesis rationales and next-step reasons are quoted from the researcher; everything else is derived from scores. Evidence strength uses an exact paired McNemar test where per-example predictions are on disk, otherwise an unpaired approximation, and always rests on a single validation split._
