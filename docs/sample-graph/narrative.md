# Research narrative: digits-cheap-vs-heavy

**Question.** Can cheap, fast models match a kernel SVM on 8x8 digits? Test each cheap family directly (Gaussian naive Bayes, depth-limited decision trees, kNN, plain linear models), each against the baseline and an RBF-SVM reference, rather than only chasing the best accuracy.

Domain `digits`, metric: validation accuracy (higher is better).

## Assumptions

- **Methods are limited to one CPU, no network and the libraries in the sandbox image** (accepted).
- **An improvement of at least 0.005 over the baseline is a real gain** (questioned). The threshold 0.005 is below ~2 standard errors (0.0241) of an unpaired difference on 539 validation examples, so a 'success' near the threshold can be noise.
- **Validation accuracy on one fixed split reflects held-out performance** (consistent). Across 7 scored runs, the rank correlation between validation and held-out test accuracy is 0.891. This checks the assumption; it does not prove it.

## Current conclusion

**Current best: RBF-SVM reference + same-fold CV table of every cheap family (val 0.9889).** Held-out test 0.9815. Provisional: holds until a better result arrives, and rests on the run's assumptions.

How it was reached (from the question to the conclusion):

1. `QUESTION` Can cheap, fast models match a kernel SVM on 8x8 digits? Test each cheap family directly (Gaussian naive Bayes, depth-limited decision trees, kNN, plain linear models), each against the baseline and an RBF-SVM reference, rather than only chasing the best accuracy.
2. `METHOD` Baseline: standardised features + multinomial logistic regression
3. `EXPERIMENT` Baseline run
4. `EVIDENCE` Baseline val accuracy 0.9592
5. `HYPOTHESIS` RBF-SVM reference + same-fold CV table of every cheap family
6. `EXPERIMENT` #2 RBF-SVM reference + same-fold CV table of every cheap family
7. `EVIDENCE` val accuracy 0.9889 (+0.0297 vs baseline)
8. `CONCLUSION` Current best: RBF-SVM reference + same-fold CV table of every cheap family (val 0.9889)

Supporting evidence: val accuracy 0.9889 (+0.0297 vs baseline)

## How the reasoning changed

### Adopt: RBF-SVM reference + same-fold CV table of every cheap family
- **Before:** Predicted val accuracy 0.9860. Best before this result: baseline 0.9592.
- **New evidence:** val accuracy 0.9889 (+0.0297 vs baseline)
- **Decision (adopt, empirical):** Best validation accuracy so far; it becomes the current best method.
- **Alternatives in the same round:** Direct kNN test: tuned k, weighting, metric and scaling vs SVM
- **Next direction:** Direct linear-model test: tuned LogReg / LinearSVC / Ridge vs SVM ("#2 ran LogisticRegression only on a coarse C grid as part of a broad table, and the kNN family is already queued."); Direct NB & shallow-tree test: PCA-whitened GaussianNB and depth-limited trees ("In #2, GaussianNB and decision trees only got minimal grids."); Shift-augmented and deskewed 1-NN vs gamma-tuned RBF-SVM ("In #3, 1-NN with Euclidean distance on raw pixels matched the SVM on validation (0.9889) and had a higher CV score (0.9791 vs 0.9763)."); Linear models on cheap nonlinear feature maps (Nystroem/RFF/poly2) vs SVM ("In #4, plain linear models plateaued at about 0.961 and the lesson pointed to nonlinear features.")
- **Still uncertain:** Only one candidate has beaten the baseline so far.

### Deprioritize: Direct kNN test: tuned k, weighting, metric and scaling vs SVM
- **Before:** Predicted val accuracy 0.9820. Best before this result: #2 0.9889.
- **New evidence:** val accuracy 0.9889 (+0.0297 vs baseline)
- **Decision (deprioritize, benchmark):** Beat the baseline but ties the current best #2 (0.9889); the earlier result stays current. A benchmark ranking, not a refutation.
- **Alternatives in the same round:** RBF-SVM reference + same-fold CV table of every cheap family
- **Next direction:** Shift-augmented and deskewed 1-NN vs gamma-tuned RBF-SVM ("In #3, 1-NN with Euclidean distance on raw pixels matched the SVM on validation (0.9889) and had a higher CV score (0.9791 vs 0.9763).")
- **Still uncertain:** The gap to #2 is within noise (paired McNemar p=1); it may be as good.

### Set aside: Direct linear-model test: tuned LogReg / LinearSVC / Ridge vs SVM
- **Before:** Predicted val accuracy 0.9700. Best before this result: #2 0.9889.
- **New evidence:** val accuracy 0.9629 (+0.0037 vs baseline)
- **Decision (set aside, empirical):** No detectable gain (improvement +0.0037 < 0.005). Inconclusive rather than refuted; set aside in favour of stronger leads.
- **Alternatives in the same round:** Direct NB & shallow-tree test: PCA-whitened GaussianNB and depth-limited trees
- **Next direction:** Linear models on cheap nonlinear feature maps (Nystroem/RFF/poly2) vs SVM ("In #4, plain linear models plateaued at about 0.961 and the lesson pointed to nonlinear features.")
- **Still uncertain:** Whether a small real effect exists: one split cannot resolve it. The difference from the baseline is within noise (paired McNemar p=0.754). The researcher predicted 0.9700 and measured 0.9629: its own prediction held. It is judged here against the loop's fixed 'beat the baseline' criterion, which may not be what this hypothesis asked.

### Set aside: Direct NB & shallow-tree test: PCA-whitened GaussianNB and depth-limited trees
- **Before:** Predicted val accuracy 0.9400. Best before this result: #2 0.9889.
- **New evidence:** val accuracy 0.9425 (-0.0167 vs baseline)
- **Decision (set aside, empirical):** Lower than the baseline (-0.0167), but not significantly (paired McNemar p=0.136). Inconclusive rather than refuted; set aside in favour of stronger leads.
- **Alternatives in the same round:** Direct linear-model test: tuned LogReg / LinearSVC / Ridge vs SVM
- **Still uncertain:** Whether a small real effect exists: one split cannot resolve it. The difference from the baseline is within noise (paired McNemar p=0.136). The researcher predicted 0.9400 and measured 0.9425: its own prediction held. It is judged here against the loop's fixed 'beat the baseline' criterion, which may not be what this hypothesis asked.

### Deprioritize: Linear models on cheap nonlinear feature maps (Nystroem/RFF/poly2) vs SVM
- **Before:** Predicted val accuracy 0.9860. Best before this result: #2 0.9889.
- **New evidence:** val accuracy 0.9796 (+0.0204 vs baseline)
- **Decision (deprioritize, benchmark):** Beat the baseline but trails the current best #2 (0.9889). A benchmark ranking, not a refutation.
- **Alternatives in the same round:** Shift-augmented and deskewed 1-NN vs gamma-tuned RBF-SVM
- **Still uncertain:** The gap to #2 is within noise (paired McNemar p=0.0625); it may be as good.

### Deprioritize: Shift-augmented and deskewed 1-NN vs gamma-tuned RBF-SVM
- **Before:** Predicted val accuracy 0.9917. Best before this result: #2 0.9889.
- **New evidence:** val accuracy 0.9870 (+0.0278 vs baseline)
- **Decision (deprioritize, benchmark):** Beat the baseline but trails the current best #2 (0.9889). A benchmark ranking, not a refutation.
- **Alternatives in the same round:** Linear models on cheap nonlinear feature maps (Nystroem/RFF/poly2) vs SVM
- **Still uncertain:** The gap to #2 is within noise (paired McNemar p=1); it may be as good.

## Branches not taken (kept, not deleted)

**Not supported: no detectable gain (inconclusive, not refuted):**
- Direct linear-model test: tuned LogReg / LinearSVC / Ridge vs SVM
- Direct NB & shallow-tree test: PCA-whitened GaussianNB and depth-limited trees

**Beat the baseline but deprioritized (benchmark ranking):**
- Direct kNN test: tuned k, weighting, metric and scaling vs SVM
- Shift-augmented and deskewed 1-NN vs gamma-tuned RBF-SVM
- Linear models on cheap nonlinear feature maps (Nystroem/RFF/poly2) vs SVM

## Open questions and next useful experiments

- **Do the threshold-level 'successes' survive a noise check?**: The success threshold is below the noise level of the validation split; successes close to it should be re-checked before being relied on.
- **Which is better, #2 or #3?**: Their validation accuracy differs by 0.0000 (paired McNemar p=1), within the noise of one split. Next useful experiment: compare both with repeated cross-validation on the training split, or several seeds.
- **Check assumption: An improvement of at least 0.005 over the baseline is a real gain**: The threshold 0.005 is below ~2 standard errors (0.0241) of an unpaired difference on 539 validation examples, so a 'success' near the threshold can be noise.

---
_Generated from the run's persisted records. Hypothesis rationales and next-step reasons are quoted from the researcher; everything else is derived from scores. Evidence strength uses an exact paired McNemar test where per-example predictions are on disk, otherwise an unpaired approximation, and always rests on a single validation split._
