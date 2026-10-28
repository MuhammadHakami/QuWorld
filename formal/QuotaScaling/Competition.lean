import Mathlib

/-!
# Decision rules for the two competition entries (Fusion Codabench 17739, Weak Lensing Codabench 18339)

Lean does not predict anything. It certifies which RULE is optimal, and the data then supply the rule's numbers. Each theorem
below names the decision it governs (runs/wl/NOTES.md, doc 35).

Weak Lensing (OoD detection; metric = mean TPR at 100 log-spaced FPRs in [0.001, 0.05]):
1. `threshold_set_invariant`: a strictly increasing transform of a score keeps every threshold set, so the ROC curve and the
   metric depend only on the ORDER of the scores. Calibration of a single detector cannot change its score; it matters only
   inside a fusion of several detectors.
2. `np_finite` (Neyman–Pearson, finite form): a likelihood-ratio upper set S (q ≥ c p inside, q ≤ c p outside) catches at
   least as much OoD mass as any set T with no more InD mass. Ordering by the likelihood ratio is therefore optimal at EVERY
   FPR at once, hence for the averaged metric.
3. `gauss_loglr_affine`: for detector statistics z that are Gaussian with one covariance (precision A = Σ⁻¹) and means 0 (InD)
   vs δ (OoD), twice the log-likelihood ratio is 2 (A δ)·z − δᵀ A δ: an increasing affine function of w·z with w = Σ⁻¹ δ.
   With 2, the optimal fusion of correlated detectors is the matched filter w = Σ⁻¹ δ, not hand-set weights.

Fusion (R² terms = 1 − SSE / SST per target):
4. `route_sse` / `route_le_iff`: replacing predictions on a known group R (reversed-Ip shots) changes the SSE by exactly the
   group's SSE difference; the router never loses iff the replacement is no worse on R. The §45o read-out measures exactly
   this quantity. The optimal two-model blend is `BandGate.gate_opt_le` (same quadratic).
-/

namespace Competition

open Finset

/-- 1. Rank invariance: a strictly increasing transform keeps every threshold set (hence every ROC point). -/
theorem threshold_set_invariant {ι : Type*} (U : Finset ι) (s : ι → ℝ) (f : ℝ → ℝ) (hf : StrictMono f) (t : ℝ) :
    U.filter (fun i => t < s i) = U.filter (fun i => f t < f (s i)) := by
  ext i; simp [hf.lt_iff_lt]

/-- 2. Neyman–Pearson on a finite sample space (InD weights p, OoD weights q, threshold c ≥ 0). -/
theorem np_finite {ι : Type*} [DecidableEq ι] (U S T : Finset ι) (p q : ι → ℝ) (c : ℝ) (hc : 0 ≤ c)
    (hT : T ⊆ U) (hin : ∀ i ∈ S, c * p i ≤ q i) (hout : ∀ j ∈ U \ S, q j ≤ c * p j)
    (hP : ∑ i ∈ T, p i ≤ ∑ i ∈ S, p i) :
    ∑ i ∈ T, q i ≤ ∑ i ∈ S, q i := by
  have hS1 := Finset.sum_inter_add_sum_diff S T q
  have hT1 := Finset.sum_inter_add_sum_diff T S q
  have hS2 := Finset.sum_inter_add_sum_diff S T p
  have hT2 := Finset.sum_inter_add_sum_diff T S p
  rw [Finset.inter_comm] at hT1 hT2
  have h1 : c * ∑ i ∈ S \ T, p i ≤ ∑ i ∈ S \ T, q i := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun i hi => hin i (Finset.mem_sdiff.mp hi).1
  have h2 : ∑ i ∈ T \ S, q i ≤ c * ∑ i ∈ T \ S, p i := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun j hj => hout j ?_
    exact Finset.mem_sdiff.mpr ⟨hT (Finset.mem_sdiff.mp hj).1, (Finset.mem_sdiff.mp hj).2⟩
  have h3 : 0 ≤ c * (∑ i ∈ S \ T, p i - ∑ i ∈ T \ S, p i) := mul_nonneg hc (by linarith)
  nlinarith

/-- 3. Gaussian detectors with a shared precision A (symmetric): the log-likelihood ratio is affine in (A δ)·z. -/
theorem gauss_loglr_affine {n : Type*} [Fintype n] (A : Matrix n n ℝ) (hA : A.transpose = A) (z δ : n → ℝ) :
    z ⬝ᵥ A.mulVec z - (z - δ) ⬝ᵥ A.mulVec (z - δ) = 2 * (A.mulVec δ ⬝ᵥ z) - δ ⬝ᵥ A.mulVec δ := by
  have hs : δ ⬝ᵥ A.mulVec z = A.mulVec δ ⬝ᵥ z := by
    rw [Matrix.dotProduct_mulVec, ← Matrix.mulVec_transpose, hA]
  have hc : z ⬝ᵥ A.mulVec δ = A.mulVec δ ⬝ᵥ z := dotProduct_comm _ _
  simp only [Matrix.mulVec_sub, sub_dotProduct, dotProduct_sub]
  rw [hs, hc]; ring

/-- The matched-filter score w·z (w = A δ) orders maps exactly as the Gaussian likelihood ratio does. -/
theorem gauss_loglr_mono {n : Type*} [Fintype n] (A : Matrix n n ℝ) (hA : A.transpose = A) (δ z z' : n → ℝ) :
    (z ⬝ᵥ A.mulVec z - (z - δ) ⬝ᵥ A.mulVec (z - δ) ≤ z' ⬝ᵥ A.mulVec z' - (z' - δ) ⬝ᵥ A.mulVec (z' - δ)) ↔
      A.mulVec δ ⬝ᵥ z ≤ A.mulVec δ ⬝ᵥ z' := by
  rw [gauss_loglr_affine A hA z δ, gauss_loglr_affine A hA z' δ]; constructor <;> intro h <;> linarith

/-- 4. Routing on a known group R ⊆ U: the routed SSE is the base SSE minus the group's SSE difference. -/
theorem route_sse {ι : Type*} [DecidableEq ι] (U R : Finset ι) (hR : R ⊆ U) (eb ec : ι → ℝ) :
    ∑ i ∈ U, (if i ∈ R then ec i else eb i) ^ 2 = ∑ i ∈ U, eb i ^ 2 - (∑ i ∈ R, eb i ^ 2 - ∑ i ∈ R, ec i ^ 2) := by
  have hsplit : ∀ f : ι → ℝ, ∑ i ∈ U, f i = ∑ i ∈ U \ R, f i + ∑ i ∈ R, f i := fun f =>
    (Finset.sum_sdiff hR).symm
  rw [hsplit (fun i => (if i ∈ R then ec i else eb i) ^ 2), hsplit (fun i => eb i ^ 2)]
  have hA : ∑ i ∈ U \ R, (if i ∈ R then ec i else eb i) ^ 2 = ∑ i ∈ U \ R, eb i ^ 2 :=
    Finset.sum_congr rfl fun i hi => by rw [if_neg (Finset.mem_sdiff.mp hi).2]
  have hB : ∑ i ∈ R, (if i ∈ R then ec i else eb i) ^ 2 = ∑ i ∈ R, ec i ^ 2 :=
    Finset.sum_congr rfl fun i hi => by rw [if_pos hi]
  rw [hA, hB]; ring

/-- The router never loses iff the replacement is no worse on the routed group. -/
theorem route_le_iff {ι : Type*} [DecidableEq ι] (U R : Finset ι) (hR : R ⊆ U) (eb ec : ι → ℝ) :
    ∑ i ∈ U, (if i ∈ R then ec i else eb i) ^ 2 ≤ ∑ i ∈ U, eb i ^ 2 ↔ ∑ i ∈ R, ec i ^ 2 ≤ ∑ i ∈ R, eb i ^ 2 := by
  rw [route_sse U R hR eb ec]; constructor <;> intro h <;> linarith

/-! ## Robustness over the unknown OoD mixture (user, Oct 17: optimise for robustness, not the best mean)

The decision threshold at each FPR depends on the InD scores only, so on a test set whose OoD part mixes families f with proportions π f,
the TPR at every threshold, and hence the averaged metric, is Σ π f · metric f (`mixture_metric_linear`). The public OoD mixture is
unknown, so the robust rules are:
5. `mixture_ge_min` / `mixture_worst_case`: the worst case of a configuration over ALL mixtures is its worst family; maximising the
   worst family (max-min) maximises the guaranteed score.
6. `dominance_any_mixture`: if no family loses more than ε (d f ≥ −ε), the change loses at most ε under EVERY mixture; with ε = 0 it
   never loses. This is the robust adoption gate (no family degrades), stronger than a better mean.
-/

/-- 5a. The averaged metric of a mixture is linear in the mixture weights (thresholds come from InD only). -/
theorem mixture_metric_linear {F G : Type*} (s : Finset F) (g : Finset G) (π : F → ℝ) (T : F → G → ℝ) :
    (∑ k ∈ g, ∑ f ∈ s, π f * T f k) / g.card = ∑ f ∈ s, π f * ((∑ k ∈ g, T f k) / g.card) := by
  rw [Finset.sum_comm, Finset.sum_div]
  refine Finset.sum_congr rfl fun f _ => ?_
  rw [← Finset.mul_sum, mul_div_assoc]

/-- 5b. Any mixture scores at least the worst family. -/
theorem mixture_ge_min {F : Type*} (s : Finset F) (π m : F → ℝ) (hπ : ∀ f ∈ s, 0 ≤ π f) (hsum : ∑ f ∈ s, π f = 1)
    (c : ℝ) (hm : ∀ f ∈ s, c ≤ m f) : c ≤ ∑ f ∈ s, π f * m f := by
  calc c = ∑ f ∈ s, π f * c := by rw [← Finset.sum_mul, hsum, one_mul]
    _ ≤ ∑ f ∈ s, π f * m f := Finset.sum_le_sum fun f hf => mul_le_mul_of_nonneg_left (hm f hf) (hπ f hf)

/-- 5c. The point mass on a family attains that family's metric, so the worst case over mixtures is exactly the worst family. -/
theorem mixture_worst_case {F : Type*} [DecidableEq F] (s : Finset F) (m : F → ℝ) (f₀ : F) (hf₀ : f₀ ∈ s) :
    ∑ f ∈ s, (if f = f₀ then (1 : ℝ) else 0) * m f = m f₀ := by
  simp [hf₀]

/-- 6. Robust adoption gate: if no family loses more than ε, the change loses at most ε under every mixture. -/
theorem dominance_any_mixture {F : Type*} (s : Finset F) (π d : F → ℝ) (hπ : ∀ f ∈ s, 0 ≤ π f) (hsum : ∑ f ∈ s, π f = 1)
    (ε : ℝ) (hd : ∀ f ∈ s, -ε ≤ d f) : -ε ≤ ∑ f ∈ s, π f * d f :=
  mixture_ge_min s π d hπ hsum (-ε) hd

end Competition
