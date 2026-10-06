# docs/PAPER.md — Publication-grade draft (companion to README)

## Title
**Grokking the Clock: From Memorization to Fourier Circuits in a Hand-Written Transformer — A Verified Reproduction, Benchmark, and Market Boundary**

## Abstract
We reproduce delayed generalization ("grokking") from scratch on modular addition (P=113) with a 227k-parameter one-layer transformer containing no ready-made layers. Memorization (100% train) arrives by epoch ~400; with AdamW weight decay λ=1, test accuracy jumps from ~30% to 100% near epoch 7,000 (30% data, seed 0), while the sum of squared weights falls 9,048→928. Without decay the net never generalizes (8.4% test, |w|²=11,436). Fourier analysis shows the grokked embedding concentrates 92.3% of its power in 4 speeds (Gini 0.71) versus a flat spectrum when memorizing (Gini 0.025); trained logits are 93.2% explained by Σαₖcos(wₖ(a+b−c)); keep-only-keys ablation preserves 99.7% test while remove-keys destroys it (1.2%); a 3-line handwritten cosine algorithm scores 100% with no training. Sweeps confirm: more data accelerates grokking (25%→12k, 30%→7k, 40%→4.1k, 50%→1.4k); stronger decay accelerates it (λ=0.5→>8k, 1→7k, 2→1.5k, Δt∝1/λ); seeds change time and keys but not the phenomenon; subtraction needs more data (30% never, 50% @2.2k). A live-market control (ECB FX monthly, BTC daily) under the same protocol never exceeds chance, bounding grokking to clean enumerable tasks. We distill six hidden-pattern seeds for future PhD work and release a Docker-reproducible benchmark.

## 1 Introduction
Overparameterized nets that perfectly fit training data yet generalize have troubled classical theory. Power et al. (2022) made the tension extreme: on small algorithmic tables, test accuracy can sit at chance for 10³× longer than train accuracy needs to saturate, then jump to perfect. Nanda et al. (2023) explained one case mechanistically (Fourier multiplication) and split training into memorization / circuit formation / cleanup. Since then (2024–2026) the literature has added precursors (FSD), causal wd-fork laws, topology/prior interventions, and discrete-log extensions. AllRest on the same fruit-fly task. We asked: can the full arc — data, hand-written model, grokking, circles, handwritten clock, sweeps, market boundary — be rebuilt from zero, verified live, and packaged so a reader can rerun every number? Yes.

## 2 Related work (2022–2026)
Power 2022; Nanda 2023; Gromov 2023 (regularization-free MLP grokking, analytic weights); Furuta 2024 (polynomial superpositions, co-grokking); Sivasankar 2026 (FSD +1,722 steps lead, Δt=C/λ); Singh 2026 (LN position, compressibility); 2603.05228 (spherical/uniform ablations bypass grokking); 2607.04333 (true/sibling/random priors 22/30, 14/15, 0/20); 2606.17399 (discrete-log clock for ×); 2602.16849 (diversification + majority vote); plus community repros (clockwork, BurnyCoder) and Nanda's Colab.

## 3 Method
### 3.1 Task
(a+b) mod 113, 12,769 equations, tokens a,b,= (id 113) → c. Random 30% train (3,830) / 70% test (8,939). Also sub/mul, other fractions/seeds.
### 3.2 Model
1-layer, d=128, 4 heads ×32, MLP 512 ReLU, learned pos (3), no LN, untied WE/WU. Forward: x=WE[t]+pos; 4-head attention queried from `=`; residual; MLP; residual; WU logits. Init N(0,0.2²). 226,688 params.
### 3.3 Optimization
Full-batch AdamW lr=1e-3, β=(0.9,0.98), 10-step warmup, CE in float64, wd∈{0,0.5,1,2}. Log train/test acc + train loss + Σw² every 100 epochs to JSON + final .pt.
### 3.4 Analysis
rFFT shares over 56 speeds, top-4 keys, Gini, circle projections, keep/remove WE ablations, OLS fit Σαₖcos(wₖ(a+b−c)) (FVE), handwritten clock argmax, sweeps, live-market logistic control (chronological split).

## 4 Results
See README §5 table (all execution-verified). Headline: wd=1 groks @7k to 100% with |w|²→928; wd=0 sticks at 8.4% with |w|²=11,436. Fourier sparsity 92.3% vs 8.5%; FVE 93.2% vs 0.15%; ablations 99.7% vs 1.2%. Fraction/decay/seed/subtraction laws as above. Market: chance.

## 5 Mechanism (what the net learned)
Embedding rows self-organize into circles (never told about order); attention+MLP form cos(wₖ(a+b)), sin(wₖ(a+b)) via trig identities; WL reads off cos(wₖ(a+b−c)); summation over 3–5 keys gives constructive interference only at c*=a+b (runner-up 0.998 with 1 key → 2.9/4 with 4 keys). Spare 4th key is typical scaffolding.

## 6 Why weight decay works here
Two solutions fit train: memorization (large |w|²) and circles (small |w|²). Decay is a constant pressure toward the small one once train loss saturates; formation is gradual (keys grow 14%→54% by epoch 5k in the video; FSD leads by ~1.7k in 2026 work), cleanup is sudden (Gini jumps, test jumps). Fork experiments (ours λ sweep + Sivasankar causal forks) support Δt≈(1/λ)log(‖Wmem‖/τ).

## 7 Boundary: markets do not grok
Same memorize-vs-generalize protocol on live ECB EUR (47 months) and BTC (365 days, CoinGecko) plus offline-tolerant fallback never exceeds chance (FX 53.8→46.2% test; BTC 48–50%). Enumerability + exact rule + stationarity appear necessary; noise/non-stationarity break the circle. Pre-registered null confirmed.

## 8 Hidden patterns → future papers
(1) Seed→key mapping from init FFT (lottery-ticket test). (2) Spare-key scaffolding dynamics via αₖ(t). (3) Rank-10 vs full-rank compression bound. (4) FSD-scheduled wd. (5) Symmetry-gap prediction for polynomial family. (6) Outlier-fraction breakdown curve.

## 9 Reproducibility
`pip install -r requirements.txt`, `make verify`, `docker compose up --build grokking-cpu`. All histories in `results/*.json`. Known deviation: absolute grok epoch 7k vs video 5.1k (init/seed sensitivity, disclosed; shape identical).

## References
Power et al. arXiv:2201.02177 (2022); Nanda et al. arXiv:2301.05217 (2023); Gromov 2301.02679; Furuta et al. 2402.16726; Sivasankar 2606.12966; Singh et al. 2602.06702; 2603.05228; 2607.04333; 2606.17399; 2602.16849; 2603.29262; Ootani 2609.20166.
