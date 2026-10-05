# VoidFormer mathematical foundation

## Token state

A token's representation is treated as a superposition of two semantic states:

```
|T⟩ = α |S_c⟩ + β |S_v⟩
```

where `S_c` is the classical (deterministic) state and `S_v` the void (latent
probabilistic) state. The mixing coefficients `α, β` are produced
context-adaptively.

## Layer update

For layer ℓ, given previous classical and void states `c_{ℓ-1}, v_{ℓ-1}`:

```
c'_ℓ = c_{ℓ-1} + ClassicalAttn(c_{ℓ-1})
v'_ℓ = v_{ℓ-1} + VoidAttn(v_{ℓ-1})
i_+, i_- = InterferenceAttn(c'_ℓ, v'_ℓ)
combined_ℓ = α_ℓ c'_ℓ + β_ℓ φ(v'_ℓ) + γ⁺_ℓ i_+ - γ⁻_ℓ i_- + δ_ℓ (c'_ℓ ⊙ φ(v'_ℓ))
```

`φ` is a learned projection `R^{d_v} → R^{d_m}`. The minus on `i_-` provides a
phase-like antisymmetric component reminiscent of constructive/destructive
interference.

## Stochastic void attention

```
K' = K + ε,  ε ~ N(0, σ² I)
A   = softmax((Q K'ᵀ / √d) · 1/τ_h)
A   = (1 - π) A + π U
Out = A V
```

where `π` is a fixed uniform-floor mass (latent probability field) and `τ_h` a
per-head temperature. Entropy of `A` is propagated to the collapse engine.

## Adaptive collapse

```
τ_ℓ = σ( g(combined_ℓ, cos(c'_ℓ, combined_ℓ), H[A]) / T  + 4(ℓ/L - 0.5) )
y_ℓ = (1 - τ_ℓ) combined_ℓ + τ_ℓ c'_ℓ
```

Mode controls the binarisation:
* soft     — τ as-is
* hard     — straight-through `τ → 𝟙[τ > 0.5]`
* coexist  — τ ← 0.3 τ (caps collapse strength)

## Uncertainty memory (recurrent over depth)

```
z_ℓ = σ(W_z [v'_ℓ, m_{ℓ-1}])
r_ℓ = σ(W_r [v'_ℓ, m_{ℓ-1}])
ĉ_ℓ = tanh(W_c [v'_ℓ, r_ℓ ⊙ m_{ℓ-1}])
m_ℓ = (1 - z_ℓ) m_{ℓ-1} + z_ℓ ĉ_ℓ
v''_ℓ = LN(v'_ℓ + W_o m_ℓ)
```

## Loss decomposition

```
L = L_lm
  + λ_e Σ_ℓ (1 - depth_ℓ) (-H[A_ℓ])
  + λ_s (1 - cos(c_L, e_pred))
  + λ_c Σ_ℓ w_mid(ℓ) ‖τ_ℓ - 0.5‖²
  + λ_i Σ_ℓ ‖coef_ℓ‖²
  + λ_a Σ_ℓ (1 - depth_ℓ) τ_ℓ
```

`w_mid(ℓ) = 1 - 2|ℓ/L - 0.5|` peaks at the network's middle and is zero at
extremes.

## Riemannian geodesic on the semantic manifold

We now lift the architecture onto a statistical manifold. Treat each token's
collapsed trajectory through the network as a curve

```
x : [0, 1] → ℝ^{d_model},        x(ℓ/L) := combined_ℓ
```

equipped with a positive-definite Fisher-like metric tensor `I_ℓ(x)` whose
conformal factor is driven by the local void-attention entropy.

### Continuous form

```
d(I, U) = ∫₀¹ √( Σ_ij I_ij(x(t)) ẋⁱ(t) ẋʲ(t) ) dt
```

### Discrete form used in code

```
d(I, U) ≈ Σ_{ℓ=0}^{L-1} √( Δx_ℓᵀ I_ℓ(x̄_ℓ) Δx_ℓ )
Δx_ℓ   = x_{ℓ+1} - x_ℓ
x̄_ℓ    = ½ (x_ℓ + x_{ℓ+1})           (midpoint, ~trapezoidal accuracy)
```

### Metric parametrisation

Low-rank + diagonal + conformal:

```
I_ℓ(x) = κ_ℓ(H_ℓ) · ( U_ℓ(x)ᵀ U_ℓ(x) + diag(d_ℓ(x)) + ε I )
```

* `U_ℓ(x) ∈ ℝ^{k × D}` — learned low-rank factor (default k = 16)
* `d_ℓ(x) ∈ ℝ^D₊`        — softplus diagonal correction
* `κ_ℓ(H_ℓ) ∈ ℝ₊`        — conformal factor; small MLP of the per-token
                            void-attention entropy `H_ℓ`

This guarantees PSD-ness by construction and gives `O(k · D)` cost per token
per layer for the Mahalanobis norm via `‖UΔx‖² + Σ d_i Δx_i²`.

### Loss term

```
L_geodesic = ‖d(I, U) − τ_geo‖² + 0.1 · Var_t[d(I, U)]
```

— pulls the per-token path length toward a target `τ_geo` (so the model
neither over- nor under-travels on the manifold) while smoothing across the
sequence. As entropy grows, `κ` raises locally, the metric is "stretched",
and ambiguous tokens accumulate larger geodesic distance — a quantitative
proxy for how much semantic work the network performed.

