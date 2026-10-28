import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
Conditional real-arithmetic guarantees for the relative routing correction.
The implementation supplies sigmoid/tanh outputs bounded in absolute value by
one. This file assumes those bounds explicitly; it does not verify PyTorch,
floating-point arithmetic, the downstream model, or predictive usefulness.
-/
namespace QuotaRouting

noncomputable def correction (scale rms gate direction probability : ℝ) : ℝ :=
  (scale * rms) * (gate * direction * probability)

theorem correction_abs_bound {s r g d p : ℝ}
    (hs : 0 ≤ s) (hr : 0 ≤ r)
    (hg : |g| ≤ 1) (hd : |d| ≤ 1) (hp : |p| ≤ 1) :
    |correction s r g d p| ≤ s * r := by
  have hgd : |g * d| ≤ 1 := by
    rw [abs_mul]
    nlinarith [abs_nonneg g, abs_nonneg d,
      mul_nonneg (sub_nonneg.mpr hg) (sub_nonneg.mpr hd)]
  have hgdp : |g * d * p| ≤ 1 := by
    rw [abs_mul]
    nlinarith [abs_nonneg (g * d), abs_nonneg p,
      mul_nonneg (sub_nonneg.mpr hgd) (sub_nonneg.mpr hp)]
  unfold correction
  rw [abs_mul, abs_of_nonneg (mul_nonneg hs hr)]
  simpa using mul_le_mul_of_nonneg_left hgdp (mul_nonneg hs hr)

theorem zero_gate (s r d p : ℝ) : correction s r 0 d p = 0 := by
  simp [correction]

theorem zero_allocation (s r g d : ℝ) : correction s r g d 0 = 0 := by
  simp [correction]

theorem token_energy_bound {n : ℕ} (x g d : Fin n → ℝ) {s r p : ℝ}
    (hs : 0 ≤ s) (hr : 0 ≤ r) (hp : |p| ≤ 1)
    (hg : ∀ i, |g i| ≤ 1) (hd : ∀ i, |d i| ≤ 1)
    (hRMS : (n : ℝ) * r ^ 2 = ∑ i, x i ^ 2) :
    (∑ i, correction s r (g i) (d i) p ^ 2) ≤ s ^ 2 * ∑ i, x i ^ 2 := by
  have hpoint (i : Fin n) : correction s r (g i) (d i) p ^ 2 ≤ (s * r) ^ 2 := by
    have h := (sq_le_sq₀ (abs_nonneg (correction s r (g i) (d i) p))
      (mul_nonneg hs hr)).2 (correction_abs_bound hs hr (hg i) (hd i) hp)
    simpa using h
  calc
    (∑ i, correction s r (g i) (d i) p ^ 2) ≤ ∑ _i : Fin n, (s * r) ^ 2 :=
      Finset.sum_le_sum (fun i _ => hpoint i)
    _ = (n : ℝ) * (s * r) ^ 2 := by simp
    _ = s ^ 2 * ((n : ℝ) * r ^ 2) := by ring
    _ = s ^ 2 * ∑ i, x i ^ 2 := by rw [hRMS]

theorem active_qubit_cap (activeRegisters : ℕ) (h : activeRegisters ≤ 3) :
    10 * activeRegisters ≤ 30 := by omega

end QuotaRouting
