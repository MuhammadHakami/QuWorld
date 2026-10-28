import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# Conditional hybrid scaling identities

These theorems certify algebra and resource accounting, not an empirical
relationship between qubit count and prediction error. `classical`, `data`,
and `optimization` are nonnegative error terms supplied by a fitted model.
The signed gain parameter is deliberately not assumed positive globally.
-/
namespace QuotaScaling

noncomputable def utilization (rank scale : ℝ) : ℝ := rank / (rank + scale)

def hybridError (floor classical data optimization gain use : ℝ) : ℝ :=
  floor + classical * (1 - gain * use) + data + optimization

theorem utilization_nonneg {r s : ℝ} (hr : 0 ≤ r) (hs : 0 < s) :
    0 ≤ utilization r s := by
  unfold utilization
  exact div_nonneg hr (by linarith)

theorem utilization_lt_one {r s : ℝ} (hr : 0 ≤ r) (hs : 0 < s) :
    utilization r s < 1 := by
  unfold utilization
  apply (div_lt_one (by linarith : 0 < r + s)).2
  linarith

theorem utilization_mono {r₁ r₂ s : ℝ}
    (h₁ : 0 ≤ r₁) (h₁₂ : r₁ ≤ r₂) (hs : 0 < s) :
    utilization r₁ s ≤ utilization r₂ s := by
  unfold utilization
  apply (div_le_div_iff₀ (by linarith : 0 < r₁ + s)
    (by linarith : 0 < r₂ + s)).2
  nlinarith

theorem classical_limit (f a d o g : ℝ) :
    hybridError f a d o g 0 = f + a + d + o := by
  simp [hybridError]

theorem zero_gain (f a d o u : ℝ) :
    hybridError f a d o 0 u = f + a + d + o := by
  simp [hybridError]

theorem gain_identity (f a d o g u : ℝ) :
    (f + a + d + o) - hybridError f a d o g u = a * g * u := by
  unfold hybridError
  ring

theorem error_nonneg {f a d o g u : ℝ}
    (hf : 0 ≤ f) (ha : 0 ≤ a) (hd : 0 ≤ d) (ho : 0 ≤ o)
    (hg : g ≤ 1) (hu : 0 ≤ u) (hu₁ : u ≤ 1) :
    0 ≤ hybridError f a d o g u := by
  have hgu : g * u ≤ 1 := le_trans (mul_le_mul_of_nonneg_right hg hu) (by simpa using hu₁)
  have hterm : 0 ≤ a * (1 - g * u) := mul_nonneg ha (by linarith)
  unfold hybridError
  linarith

-- Monotonic improvement is CONDITIONAL on nonnegative fitted gain.
theorem error_antitone_in_use {f a d o g u v : ℝ}
    (ha : 0 ≤ a) (hg : 0 ≤ g) (huv : u ≤ v) :
    hybridError f a d o g v ≤ hybridError f a d o g u := by
  have h := mul_le_mul_of_nonneg_left huv (mul_nonneg ha hg)
  unfold hybridError
  nlinarith

-- Negative fitted gain permits harm rather than forcing an advantage.
theorem nonpositive_gain_no_benefit {f a d o g u : ℝ}
    (ha : 0 ≤ a) (hg : g ≤ 0) (hu : 0 ≤ u) :
    f + a + d + o ≤ hybridError f a d o g u := by
  have h : a * g * u ≤ 0 := mul_nonpos_of_nonpos_of_nonneg
    (mul_nonpos_of_nonneg_of_nonpos ha hg) hu
  rw [← gain_identity f a d o g u] at h
  linarith

-- Marginal saturation of the chosen rational utilization coordinate.
theorem utilization_increment {r s t : ℝ}
    (hr : 0 ≤ r) (hs : 0 < s) (ht : 0 ≤ t) :
    utilization (r + t) s - utilization r s =
      s * t / ((r + s) * (r + t + s)) := by
  have h₁ : r + s ≠ 0 := ne_of_gt (by linarith)
  have h₂ : r + t + s ≠ 0 := ne_of_gt (by linarith)
  unfold utilization
  field_simp
  ring

-- An abstract inverse-resource special case, not a claim that measured loss
-- has this form. It yields a certified allocation lower bound.
theorem allocation_bound {a b x y : ℝ} (hx : 0 < x) (hy : 0 < y) :
    (a + b)^2 / (x + y) ≤ a^2 / x + b^2 / y := by
  apply (div_le_iff₀ (by linarith : 0 < x + y)).2
  have hid : (a^2 / x + b^2 / y) * (x + y) - (a + b)^2 =
      (a * y - b * x)^2 / (x * y) := by
    field_simp
    ring
  have hn : 0 ≤ (a * y - b * x)^2 / (x * y) :=
    div_nonneg (sq_nonneg _) (le_of_lt (mul_pos hx hy))
  linarith

-- Counting convention: total amplitude entries across independent registers,
-- without autograd/buffers. This is not peak memory or a runtime theorem.
def stateEntries (registers width : ℕ) : ℕ := registers * 2 ^ width

theorem independent_registers_linear (k n : ℕ) :
    stateEntries (k + 1) n = stateEntries k n + 2 ^ n := by
  simp [stateEntries, Nat.add_mul]

theorem register_width_doubles (k n : ℕ) :
    stateEntries k (n + 1) = 2 * stateEntries k n := by
  simp [stateEntries, pow_succ]
  ring

theorem thirty_qubit_accounting : stateEntries 3 10 = 3072 := by norm_num [stateEntries]
theorem monolithic_thirty : stateEntries 1 30 = 1073741824 := by norm_num [stateEntries]

end QuotaScaling
