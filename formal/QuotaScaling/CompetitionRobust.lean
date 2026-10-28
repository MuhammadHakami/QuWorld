import QuotaScaling.Competition

/-!
Bounds for two competition decisions when the evaluated population differs from
the public proxy or validation population. These are conditional arithmetic
statements. They do not estimate the unknown mixture mass or score change.
-/

namespace Competition

/-- If the represented OoD mixture loses at most `ε`, an unrepresented mass `η`
can lose the full unit range of a [0,1] detection metric. Thus the overall
guarantee degrades to `-(1-η)*ε-η`. -/
theorem mixture_with_unknown_mass (ε η represented_delta unknown_delta : ℝ)
    (hη0 : 0 ≤ η) (hη1 : η ≤ 1)
    (hrepresented : -ε ≤ represented_delta)
    (hunknown : -1 ≤ unknown_delta) :
    -((1 - η) * ε + η) ≤ (1 - η) * represented_delta + η * unknown_delta := by
  have h1 : (1 - η) * (-ε) ≤ (1 - η) * represented_delta :=
    mul_le_mul_of_nonneg_left hrepresented (by linarith)
  have h2 : η * (-1) ≤ η * unknown_delta :=
    mul_le_mul_of_nonneg_left hunknown hη0
  linarith

/-- An absolute metric lower bound `m` on the represented families becomes
`(1-η)*m` when the unrepresented mass is `η` and its metric is merely known
to lie in [0,1]. This remains conditional on an upper bound for `η`. -/
theorem metric_with_unknown_mass (η m represented_metric unknown_metric : ℝ)
    (hη0 : 0 ≤ η) (hη1 : η ≤ 1)
    (hrepresented : m ≤ represented_metric)
    (hunknown : 0 ≤ unknown_metric) :
    (1 - η) * m ≤ (1 - η) * represented_metric + η * unknown_metric := by
  have h1 : (1 - η) * m ≤ (1 - η) * represented_metric :=
    mul_le_mul_of_nonneg_left hrepresented (by linarith)
  have h2 : 0 ≤ η * unknown_metric := mul_nonneg hη0 hunknown
  linarith

/-- If a pooled target SST is the sum of two contributions, its *unclipped* R²
improvement weights each subset's SSE reduction by that pooled SST. The inputs
`sstA` and `sstB` here must be contributions to that same denominator, not
independently centred SSTs when group means differ. -/
theorem pooled_r2_delta_sst_weighted
    (sstA sstB oldA newA oldB newB : ℝ)
    (hA : 0 < sstA) (hB : 0 < sstB) :
    ((oldA - newA) + (oldB - newB)) / (sstA + sstB) =
      (sstA / (sstA + sstB)) * ((oldA - newA) / sstA) +
      (sstB / (sstA + sstB)) * ((oldB - newB) / sstB) := by
  have hsum : sstA + sstB ≠ 0 := ne_of_gt (by linarith)
  field_simp
  ring

/-- If `sstA` and `sstB` are separately centred subset SSTs, the pooled SST
also has a nonnegative between-group mean term. This equation keeps that term
in the denominator. It applies to raw R² before the competition's zero clip. -/
theorem pooled_r2_delta_with_between
    (sstA sstB between oldA newA oldB newB : ℝ)
    (hA : 0 < sstA) (hB : 0 < sstB) (hbetween : 0 ≤ between) :
    ((oldA - newA) + (oldB - newB)) / (sstA + sstB + between) =
      (sstA / (sstA + sstB + between)) * ((oldA - newA) / sstA) +
      (sstB / (sstA + sstB + between)) * ((oldB - newB) / sstB) := by
  have hsum : sstA + sstB + between ≠ 0 := ne_of_gt (by linarith)
  field_simp
  ring

end Competition
