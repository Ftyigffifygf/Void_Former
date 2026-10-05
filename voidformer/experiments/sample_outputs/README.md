# Sample diagnostic outputs

Produced by `python -m voidformer.experiments.sanity_run
--config configs/tiny.yaml --prompt "the cell divided into two daughter cells"`
on a freshly initialised (un-trained) tiny model.

| File                          | What it shows                                                |
|-------------------------------|--------------------------------------------------------------|
| `entropy_heatmap.png`         | Void-attention entropy per (layer, token)                    |
| `collapse_trajectory.png`     | Mean τ across depth                                          |
| `interference_coefs.png`      | α, β, γ⁺, γ⁻, δ — interference engine coefficients           |
| `route_weights.png`           | DynamicRouter weights (classical / void / merged) per layer  |
| `geodesic_path.png`           | Riemannian arc-length `‖Δx‖_{I_ℓ}` per layer + cumulative + per-token total |
| `geodesic_distribution.png`   | Histogram of per-token geodesic lengths `d(I, U)`            |

## Riemannian geodesic interpretation

For each token, the trajectory of its collapsed state through the L layers
is treated as a curve x(t) on a statistical manifold. The Fisher-information
metric tensor `I_ℓ` is locally inflated by void-attention entropy, so
ambiguous tokens accumulate a longer arc-length

    d(I, U) = ∫₀¹ √( Σ_ij I_ij(x(t)) ẋⁱ ẋʲ ) dt

This number is the model's *Riemannian semantic distance* — a quantitative
measure of how much manifold-distance each token traversed before settling on
its meaning. After real training, you should see:

* high-entropy / polysemous tokens (e.g. *cell*, *bank*, *light*) → larger d
* unambiguous tokens (e.g. articles, punctuation) → smaller d
* the distribution skew shifts as context resolves polysemy
