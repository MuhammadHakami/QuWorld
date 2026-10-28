import Mathlib

/-!
# Utilization ceilings of an allocated co-processor

Formal backbone of `33-allocation-strategy-study.md`. The study measures, for every allocation strategy, the
*utilization ceiling* `U` (the share of the trunk's error energy on the allocated components) and the realized
*captured* share. These theorems state exactly what the measurements test. None asserts that any corrector
predicts well, and none is a claim of quantum advantage.

1. `gain_le_ceiling`: in any orthonormal basis, a correction supported on a component set `T` (arbitrary
   coefficients, e.g. produced by a quantum register or an MLP) reduces the error energy by at most the energy
   of the error on `T`. So captured ≤ U for every core and every allocation (protocol check P1).
2. `ceiling_mono`: nested allocations have nested ceilings, so a top-K rule's ceiling is non-decreasing in K
   (the routed half of prediction P3).
3. `pooled_gain_le_dc`: a broadcast (spatially constant) correction `c`, whatever produced it (any core, any
   number of atoms), reduces the error energy by at most the DC energy `(Σ e)² / n`. Under pooling, adding
   qubits cannot raise the ceiling (the pooled half of prediction P3).
-/

namespace Utilization

open Finset

section Basis

variable {ι E : Type*} [Fintype ι] [DecidableEq ι] [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem coeff_of_correction (b : OrthonormalBasis ι ℝ E) (T : Finset ι) (a : ι → ℝ) (e : E) (i : ι) :
    @inner ℝ _ _ (b i) (e - ∑ j ∈ T, a j • b j) = @inner ℝ _ _ (b i) e - if i ∈ T then a i else 0 := by
  have hv := b.orthonormal
  rw [inner_sub_right, inner_sum]
  simp only [inner_smul_right, orthonormal_iff_ite.mp hv]
  split_ifs with hi <;> simp [hi]

/-- Any correction supported on `T` removes at most the error energy on `T` (captured ≤ U). -/
theorem gain_le_ceiling (b : OrthonormalBasis ι ℝ E) (T : Finset ι) (a : ι → ℝ) (e : E) :
    ‖e‖ ^ 2 - ‖e - ∑ j ∈ T, a j • b j‖ ^ 2 ≤ ∑ i ∈ T, (@inner ℝ _ _ (b i) e) ^ 2 := by
  rw [← b.sum_sq_norm_inner e, ← b.sum_sq_norm_inner (e - ∑ j ∈ T, a j • b j)]
  simp_rw [coeff_of_correction b T a e, Real.norm_eq_abs, sq_abs]
  rw [← Finset.sum_sub_distrib, ← Fintype.sum_ite_mem T (fun i => (@inner ℝ _ _ (b i) e) ^ 2)]
  refine Finset.sum_le_sum (fun i _ => ?_)
  split_ifs with hi
  · nlinarith [sq_nonneg (@inner ℝ _ _ (b i) e - a i)]
  · simp

end Basis

/-- Nested allocations have nested ceilings. -/
theorem ceiling_mono {ι : Type*} (w : ι → ℝ) (hw : ∀ i, 0 ≤ w i) {S T : Finset ι} (h : S ⊆ T) :
    ∑ i ∈ S, w i ≤ ∑ i ∈ T, w i :=
  Finset.sum_le_sum_of_subset_of_nonneg h (fun i _ _ => hw i)

/-- A broadcast correction removes at most the DC energy, whatever core (and however many atoms) produced it. -/
theorem pooled_gain_le_dc {n : ℕ} (hn : 0 < n) (e : Fin n → ℝ) (c : ℝ) :
    ∑ i, e i ^ 2 - ∑ i, (e i - c) ^ 2 ≤ (∑ i, e i) ^ 2 / n := by
  have hnr : (0 : ℝ) < n := by exact_mod_cast hn
  have hexp : ∑ i, (e i - c) ^ 2 = ∑ i, e i ^ 2 - 2 * c * ∑ i, e i + n * c ^ 2 := by
    have : ∀ i, (e i - c) ^ 2 = e i ^ 2 - 2 * c * e i + c ^ 2 := fun i => by ring
    simp_rw [this, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
    simp
  rw [hexp, le_div_iff₀ hnr]
  nlinarith [sq_nonneg (∑ i, e i - n * c)]

end Utilization
