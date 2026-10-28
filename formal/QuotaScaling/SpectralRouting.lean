import Mathlib

/-!
# Spectral routing of the analog part

Formal backbone of `32-dynamic-register-configuration.md`. Every statement is a consequence of the definitions
and explicit hypotheses below. None of them asserts predictive usefulness of any model, and none is a claim of
quantum advantage.

1. `pool_blind`, `mode_zero_mean`, `pool_blind_mode`, `broadcast_orthogonal`: mean-pooling cannot see any
   zero-mean perturbation, in particular any non-DC discrete Fourier mode, and a broadcast (spatially constant)
   correction is orthogonal to every zero-mean field. A pool → core → broadcast bottleneck therefore reads and
   writes only the DC spatial mode of the latent.
2. `band_energy_split`: for any orthonormal basis (discrete Fourier or orthonormal wavelets) and any assignment of
   basis vectors to bands, the error energy is exactly the sum of the band energies (Parseval + fibres).
3. `residual_after_band_removal`: removing the error's components on a coefficient set `T` leaves exactly the
   energy of the components outside `T`. This is the ceiling for any module that corrects a budget of `|T|`
   bands or wavelet tiles.
4. `total_eq`, `pointwise_min_optimal`, `swap_improves`, `swap_card`: with separable band costs, the per-band
   argmin is optimal without a budget. Under a budget, exchanging a band for one with larger gain keeps the
   budget and never increases the total, so optimal budgeted assignments are top-k by gain.
5. `reupload_degree`: if each of `L` encoding layers, multiplied by `z = e^{iφ}`, is a polynomial of degree at
   most 2 (a Laurent polynomial with exponents in {-1, 0, 1}), the product has degree at most `2L`, so the model's
   frequency spectrum in `φ` lies in `[-L, L]`. The circuit-to-Laurent-polynomial step is the standard
   Schuld–Sweke–Meyer argument and is not formalised here.
6. `vdw_hasDerivAt`, `vdw_sensitivity_anti`: the van der Waals phase `C r⁻⁶` has derivative `-6 C r⁻⁷`, whose
   magnitude strictly decreases with distance. An input-dependent geometry is therefore a frequency modulation:
   nearer atoms respond with a higher local frequency.
7. `lattice_sep`: distinct sites of the lattice `d ℤ³` are at least `d` apart, so any occupancy pattern on a
   fixed 3-D site grid satisfies a minimum-distance hardware constraint by construction.
8. `dip_inplane`, `dip_axial`, `dip_magic`: the resonant dipolar coupling is proportional to
   `1 - 3 z² / (x² + y² + z²)` for a separation `(x, y, z)` with the quantization axis along `z`. Every pair in a
   plane perpendicular to the axis gets `+1`, a pair stacked along the axis gets `-2`, and the coupling vanishes at
   the magic angle `z² = (x² + y²) / 2`. Signed or zero couplings therefore require the third dimension.
-/

namespace SpectralRouting

open Finset

/-! ## 1. Pooling sees only the DC mode -/

/-- Mean over `n` spatial positions. -/
noncomputable def pool {K : Type*} [Field K] {n : ℕ} (x : Fin n → K) : K := (∑ i, x i) / n

theorem pool_blind {K : Type*} [Field K] {n : ℕ} (x g : Fin n → K) (hg : ∑ i, g i = 0) :
    pool (x + g) = pool x := by
  unfold pool
  simp [Finset.sum_add_distrib, hg]

/-- Every non-DC discrete Fourier mode (`ζ ^ n = 1`, `ζ ≠ 1`) has zero mean. -/
theorem mode_zero_mean {K : Type*} [Field K] {n : ℕ} (ζ : K) (hn : ζ ^ n = 1) (h1 : ζ ≠ 1) :
    ∑ j ∈ range n, ζ ^ j = 0 := by
  rw [geom_sum_eq h1, hn, sub_self, zero_div]

/-- Adding any multiple of a non-DC mode leaves the pooled value unchanged. -/
theorem pool_blind_mode {K : Type*} [Field K] {n : ℕ} (x : Fin n → K) (a ζ : K)
    (hn : ζ ^ n = 1) (h1 : ζ ≠ 1) :
    pool (fun j : Fin n => x j + a * ζ ^ (j : ℕ)) = pool x := by
  have hsum : ∑ j : Fin n, a * ζ ^ (j : ℕ) = 0 := by
    rw [← Finset.mul_sum, Fin.sum_univ_eq_sum_range (fun j => ζ ^ j) n, mode_zero_mean ζ hn h1, mul_zero]
  have := pool_blind x (fun j : Fin n => a * ζ ^ (j : ℕ)) hsum
  simpa [Pi.add_apply] using this

/-- A spatially constant (broadcast) correction is orthogonal to every zero-mean field. -/
theorem broadcast_orthogonal {n : ℕ} (c : ℝ) (g : Fin n → ℝ) (hg : ∑ i, g i = 0) :
    ∑ i, c * g i = 0 := by
  rw [← Finset.mul_sum, hg, mul_zero]

/-! ## 2–3. Energy bookkeeping in any orthonormal basis (Fourier or orthonormal wavelets) -/

section Energy

variable {ι E : Type*} [Fintype ι] [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem band_energy_split {κ : Type*} [Fintype κ] [DecidableEq κ]
    (b : OrthonormalBasis ι ℝ E) (band : ι → κ) (e : E) :
    ∑ j, ∑ i ∈ univ.filter (fun i => band i = j), ‖@inner ℝ _ _ (b i) e‖ ^ 2 = ‖e‖ ^ 2 := by
  rw [Finset.sum_fiberwise univ band (fun i => ‖@inner ℝ _ _ (b i) e‖ ^ 2)]
  exact b.sum_sq_norm_inner e

/-- Coefficient of `e` minus its projection on the coefficient set `T`. -/
theorem coeff_after_removal [DecidableEq ι] (b : OrthonormalBasis ι ℝ E) (T : Finset ι) (e : E) (i : ι) :
    @inner ℝ _ _ (b i) (e - ∑ j ∈ T, (@inner ℝ _ _ (b j) e) • b j)
      = if i ∈ T then 0 else @inner ℝ _ _ (b i) e := by
  have hv := b.orthonormal
  rw [inner_sub_right, inner_sum]
  simp only [inner_smul_right, orthonormal_iff_ite.mp hv]
  split_ifs with hi
  · simp [hi]
  · simp [hi]

/-- Ceiling for a budget of `|T|` corrected bands / tiles: the residual energy is exactly the energy outside `T`. -/
theorem residual_after_band_removal [DecidableEq ι] (b : OrthonormalBasis ι ℝ E) (T : Finset ι) (e : E) :
    ‖e - ∑ j ∈ T, (@inner ℝ _ _ (b j) e) • b j‖ ^ 2
      = ∑ i ∈ univ.filter (fun i => i ∉ T), ‖@inner ℝ _ _ (b i) e‖ ^ 2 := by
  rw [← b.sum_sq_norm_inner (e - ∑ j ∈ T, (@inner ℝ _ _ (b j) e) • b j)]
  simp_rw [coeff_after_removal b T e]
  rw [Finset.sum_filter]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  split_ifs <;> simp

end Energy

/-! ## 4. Budgeted assignment of bands to the register -/

section Assignment

variable {κ : Type*} [Fintype κ] [DecidableEq κ]

/-- Total cost when the bands in `S` go to the register (cost `q`) and the rest stay classical (cost `c`). -/
def total (c q : κ → ℝ) (S : Finset κ) : ℝ := ∑ x, if x ∈ S then q x else c x

/-- Gain of moving a band from the classical path to the register. -/
def gain (c q : κ → ℝ) (x : κ) : ℝ := c x - q x

theorem total_eq (c q : κ → ℝ) (S : Finset κ) :
    total c q S = ∑ x, c x - ∑ x ∈ S, gain c q x := by
  unfold total gain
  rw [← Fintype.sum_ite_mem S (fun x => c x - q x), ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl (fun x _ => ?_)
  split_ifs <;> ring

/-- Without a budget, taking the cheaper path per band is optimal. -/
theorem pointwise_min_optimal (c q : κ → ℝ) (S : Finset κ) :
    ∑ x, min (c x) (q x) ≤ total c q S := by
  unfold total
  refine Finset.sum_le_sum (fun x _ => ?_)
  split_ifs
  · exact min_le_right _ _
  · exact min_le_left _ _

/-- Exchanging a register band for an outside band with at least the same gain never increases the total. -/
theorem swap_improves (c q : κ → ℝ) (S : Finset κ) {x y : κ} (hx : x ∈ S) (hy : y ∉ S)
    (hg : gain c q x ≤ gain c q y) :
    total c q (insert y (S.erase x)) ≤ total c q S := by
  have hy' : y ∉ S.erase x := fun h => hy (Finset.mem_of_mem_erase h)
  rw [total_eq, total_eq, Finset.sum_insert hy']
  have := Finset.add_sum_erase S (gain c q) hx
  linarith

omit [Fintype κ] in
/-- … and the exchange preserves the qubit budget. -/
theorem swap_card (S : Finset κ) {x y : κ} (hx : x ∈ S) (hy : y ∉ S) :
    (insert y (S.erase x)).card = S.card := by
  have hy' : y ∉ S.erase x := fun h => hy (Finset.mem_of_mem_erase h)
  rw [Finset.card_insert_of_not_mem hy', Finset.card_erase_add_one hx]

end Assignment

/-! ## 5. Frequency spectrum of an L-layer re-uploading register -/

open Polynomial in
theorem reupload_degree {R : Type*} [CommRing R] (L : ℕ) (layer : Fin L → R[X])
    (h : ∀ l, (layer l).natDegree ≤ 2) :
    (∏ l, layer l).natDegree ≤ 2 * L := by
  calc (∏ l, layer l).natDegree ≤ ∑ l, (layer l).natDegree := Polynomial.natDegree_prod_le _ _
    _ ≤ ∑ _l : Fin L, 2 := Finset.sum_le_sum (fun l _ => h l)
    _ = 2 * L := by simp [mul_comm]

/-! ## 6. Geometry as frequency modulation: the van der Waals phase -/

theorem vdw_hasDerivAt (C r : ℝ) (hr : r ≠ 0) :
    HasDerivAt (fun s : ℝ => C * s ^ (-6 : ℤ)) (C * (-6 * r ^ (-7 : ℤ))) r := by
  have h := (hasDerivAt_zpow (-6) r (Or.inl hr)).const_mul C
  have e : ((-6 : ℤ) : ℝ) * r ^ ((-6 : ℤ) - 1) = -6 * r ^ (-7 : ℤ) := by norm_num
  rwa [e] at h

/-- `|d/dr (C r⁻⁶)| = 6C / r⁷` strictly decreases with distance: nearer atoms, higher local frequency. -/
theorem vdw_sensitivity_anti {C a b : ℝ} (hC : 0 < C) (ha : 0 < a) (hab : a < b) :
    6 * C / b ^ 7 < 6 * C / a ^ 7 := by
  have h6 : 0 < 6 * C := by positivity
  have ha7 : 0 < a ^ 7 := by positivity
  have hab7 : a ^ 7 < b ^ 7 := by gcongr
  exact div_lt_div_of_pos_left h6 ha7 hab7

/-! ## 7. Hardware validity of occupancy patterns on a 3-D site grid -/

theorem lattice_sep {d : ℝ} (hd : 0 < d) (p q : Fin 3 → ℤ) (hpq : p ≠ q) :
    d ≤ Real.sqrt (∑ i, (d * p i - d * q i) ^ 2) := by
  obtain ⟨i, hi⟩ : ∃ i, p i ≠ q i := by
    by_contra h
    push_neg at h
    exact hpq (funext h)
  have h1 : (1 : ℝ) ≤ |(p i : ℝ) - q i| := by
    have : (1 : ℤ) ≤ |p i - q i| := Int.one_le_abs (sub_ne_zero.mpr hi)
    exact_mod_cast this
  have hsq : 1 ≤ ((p i : ℝ) - q i) ^ 2 := by
    have := abs_mul_abs_self ((p i : ℝ) - q i)
    nlinarith [abs_nonneg ((p i : ℝ) - q i)]
  have h2 : d ^ 2 ≤ (d * p i - d * q i) ^ 2 := by
    have e : (d * p i - d * q i) ^ 2 = d ^ 2 * ((p i : ℝ) - q i) ^ 2 := by ring
    rw [e]
    nlinarith [sq_nonneg d]
  have h3 : (d * p i - d * q i) ^ 2 ≤ ∑ j, (d * p j - d * q j) ^ 2 :=
    Finset.single_le_sum (f := fun j => (d * p j - d * q j) ^ 2) (fun j _ => sq_nonneg _)
      (Finset.mem_univ i)
  calc d = Real.sqrt (d ^ 2) := (Real.sqrt_sq hd.le).symm
    _ ≤ Real.sqrt (∑ j, (d * p j - d * q j) ^ 2) := Real.sqrt_le_sqrt (h2.trans h3)

/-! ## 8. Why z matters: the anisotropic dipolar factor -/

/-- Angular factor `1 - 3 cos² θ` of the resonant dipole-dipole coupling, for separation `(x, y, z)`, axis `z`. -/
noncomputable def dip (x y z : ℝ) : ℝ := 1 - 3 * z ^ 2 / (x ^ 2 + y ^ 2 + z ^ 2)

/-- All pairs in a plane perpendicular to the axis share the same (positive) factor. -/
theorem dip_inplane (x y : ℝ) : dip x y 0 = 1 := by
  simp [dip]

/-- A pair stacked along the axis has the opposite sign. -/
theorem dip_axial {z : ℝ} (hz : z ≠ 0) : dip 0 0 z = -2 := by
  have h : z ^ 2 ≠ 0 := pow_ne_zero 2 hz
  unfold dip
  field_simp
  ring

/-- At the magic angle the coupling vanishes. -/
theorem dip_magic {x y z : ℝ} (hz : z ≠ 0) (hm : z ^ 2 = (x ^ 2 + y ^ 2) / 2) : dip x y z = 0 := by
  have hsum : x ^ 2 + y ^ 2 + z ^ 2 = 3 * z ^ 2 := by linarith
  have h : z ^ 2 ≠ 0 := pow_ne_zero 2 hz
  unfold dip
  rw [hsum]
  field_simp

end SpectralRouting
