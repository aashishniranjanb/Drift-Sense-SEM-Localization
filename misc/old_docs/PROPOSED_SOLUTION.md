Yes. Based on the results you just obtained, I would change the proposed solution before continuing.

Your benchmark has exposed something important:

The problem is not currently “we need more sophisticated verification.” The problem is that our first-stage retrieval is not reliably finding the correct region.

FFT-NCC is already ~4× faster than ZNCC with essentially the same ≤5 px accuracy, while the full Drift-Sense++ is ~657 ms and does not improve that accuracy. So adding more PC/Radon processing now is the wrong direction.

The challenge itself defines the task very simply: given a 100× reference and 10× search image, find the reference's location in the search image and return (x,y); the difficult cases are periodic layouts, independent noise, 10× scale difference, and realistic degradation.

1. First, define the problem correctly
INPUT

Two images:

Reference Image
1000 × 1000
100× magnification
~1 nm/pixel

and

Search Image
1000 × 1000
10× magnification
~10 nm/pixel

The reference represents approximately:

1000 nm/10 nm/pixel=100 pixels

inside the search image.

So conceptually:

REFERENCE
1000 × 1000
     │
     │ 10× physical scale conversion
     ▼
100 × 100 TEMPLATE
     │
     │ search
     ▼
SEARCH IMAGE
1000 × 1000
OUTPUT

Only:

(x, y)

For example:

(642.37, 381.82)

Everything else is internal.

1. The better solution I recommend now

I would simplify the architecture to:

Robust Multi-Scale Structural Registration
        REFERENCE + SEARCH
                │
                ▼
       1. SCALE NORMALIZATION
                │
                ▼
       2. STRUCTURAL FEATURES
                │
                ▼
       3. FAST GLOBAL RETRIEVAL
                │
                ▼
           TOP-K CANDIDATES
                │
                ▼
       4. LOCAL REGISTRATION
                │
                ▼
       5. PERIODICITY CHECK
                │
                ▼
       6. CENTER-BASED TIE BREAK
                │
                ▼
       7. SUBPIXEL REFINEMENT
                │
                ▼
              (x,y)

Notice what is gone:

no giant neural network
no GNN
no full-image Phase Congruency
no expensive adaptive loop
no unnecessary image warping
no complicated AI model
3. Step 1 — Scale normalization

This is absolutely required.

Take:

1000 × 1000 reference

and generate a search-resolution template:

100 × 100

using proper anti-aliasing.

But because the actual test may contain small scale variation, don't assume exactly 100×100.

Use a small bounded scale bank:

95 × 95
98 × 98
100 × 100
102 × 102
105 × 105

This corresponds to roughly ±5%.

The challenge explicitly expects scale variation, so this is justified.

1. Step 2 — Use TWO structural representations

I would not make Phase Congruency the main representation.

Use:

Channel A — normalized intensity

Useful for:

large structural differences
low-frequency layout
Channel B — gradient magnitude/orientation

Useful for:

edges
fins
gates
contacts
brightness changes

So:

Reference
 ├── intensity
 └── gradient

Search
 ├── intensity
 └── gradient

Then combine their correlation scores.

For example:

S=w
I
 ​

S
I
 ​

+w
G
 ​

S
G
 ​

where the weights are calibrated experimentally.

This is much cheaper than full Phase Congruency.

1. Step 3 — Fast global retrieval

This is the heart of the algorithm.

For every scale candidate:

100×100 template
        ↓
FFT correlation
        ↓
1000×1000 correlation map

Then:

98×98 template
        ↓
FFT correlation

etc.

We don't immediately select one result.

We extract:

Top 10 candidates

across all scale candidates.

For example:

Candidate   x      y       scale    score
-------------------------------------------

C1          521    433     1.00     0.941
C2          310    611     0.98     0.938
C3          522    433     1.02     0.936
C4          700    221     1.00     0.932
...

Now we have a much more useful representation of the problem.

1. This is the experiment we desperately need now

Your current benchmark only tells us:

Top-1 is wrong.

We need:

Top-K Recall

Suppose:

Top-1 accuracy = 36%
Top-3 = 61%
Top-5 = 76%
Top-10 = 94%

Then our algorithm is actually doing something valuable.

It means:

The correct location is being retrieved, but our ranking is wrong.

Then we improve the verification stage.

But if:

Top-1 = 36%
Top-10 = 41%

then:

The retrieval itself is broken.

Then Radon, PSR and spatial priors won't save us.

This is the most important diagnostic experiment now.

1. Step 4 — Local registration

Once we have Top-K, don't immediately compare raw pixels.

For each candidate:

Reference 100×100
        VS
Candidate patch 100×100

perform a local registration.

We want to determine:

Is this candidate actually the same structure?

Use:

Gradient correlation
S
G
 ​

=NCC(G
R
 ​

,G
C
 ​

)
Phase correlation for local translation

Phase correlation is useful here because it estimates small translational shifts efficiently.

So:

candidate
   ↓
local phase correlation
   ↓
δx, δy

Then refine:

candidate location
+
δx, δy

This is better than trying to perform an expensive global affine warp on every candidate.

1. Why phase correlation is different from Phase Congruency

This distinction is important.

We were previously talking about:

Phase Congruency

A feature representation.

image → PC map

It is expensive.

I'm now suggesting:

Phase Correlation

A registration technique.

reference + candidate
        ↓
relative translation

It is much more directly related to the actual problem:

"Are these two patches aligned?"

So I would remove Phase Congruency from the core algorithm for now and test phase correlation as a local registration tool.

1. Step 5 — Candidate scoring

Now each candidate gets a score:

S
i
 ​

=w
1
 ​

S
NCC
 ​

+w
2
 ​

S
Grad
 ​

+w
3
 ​

S
Phase
 ​

where:

S
NCC
 ​

: structural similarity
S
Grad
 ​

: edge/geometry similarity
S
Phase
 ​

: local registration quality

Don't choose the weights arbitrarily.

Use the validation dataset to calibrate them.

 1. Step 6 — Handle periodicity

Now we address the actual semiconductor problem.

Suppose:

C1 = 0.941
C2 = 0.940
C3 = 0.939

and they're separated by approximately one fin pitch.

That means:

HIGH PERIODICITY

We don't want the algorithm blindly choosing C1.

Instead:

if candidates are clearly different:
       choose highest score

if candidates are nearly identical:
       identify periodic ambiguity
       apply center preference

The challenge explicitly says that when multiple matching regions exist, the center-nearest region is the desired selection rule.

This is where the spatial rule belongs.

 1. We don't need a complicated periodicity metric initially

Start with something very simple:

ΔS=S
1
 ​

−S
2
 ​

If:

ΔS large

→ unique match.

If:

ΔS small

→ ambiguous.

Then add spectral/autocorrelation periodicity only if necessary.

This is another example of simplifying the system.

 1. Step 7 — Subpixel refinement

Only after we have selected the correct candidate:

(642,382)

we look at the local correlation surface:

5 × 5 neighborhood

and refine:

(642.31,381.76)

This is cheap.

Keep it.

 1. Final proposed algorithm

So our actual final inference could be:

                   INPUT
             Reference + Search
                     │
                     ▼
             SCALE NORMALIZATION
               95–105 px
                     │
                     ▼
           INTENSITY + GRADIENT
                     │
                     ▼
            FFT GLOBAL SEARCH
                     │
                     ▼
                 TOP-10
                     │
                     ▼
          LOCAL REGISTRATION
         NCC + Gradient + Phase
                     │
                     ▼
              CANDIDATE SCORE
                     │
                     ▼
             AMBIGUITY CHECK
                /        \
              NO          YES
              │             │
              │        Center rule
              │             │
              └──────┬──────┘
                     ▼
             SUBPIXEL REFINEMENT
                     │
                     ▼
                  (x,y)

That is what I would implement next.

 1. Why this is better than our current V10

Your current V10 is approximately:

Scale/rotation search
       ↓
Phase Congruency
       ↓
Top-K
       ↓
Radon
       ↓
Subpixel

and produces:

35.83% ≤5 px
657 ms

The proposed version is:

Scale bank
       ↓
Intensity + Gradient FFT
       ↓
Top-10
       ↓
Local registration
       ↓
Ambiguity handling
       ↓
Subpixel

The goal is:

Better accuracy

because we're explicitly separating:

retrieval
vs
ranking
Lower latency

because we're removing expensive full-canvas PC.

Better explainability

because every candidate has:

NCC
gradient score
phase-registration score
peak separation
periodicity
distance from center
15. Comparison with other methods
Method Search Scale Rotation Periodicity Compute Explainability
ZNCC Good Poor Poor Poor Medium Excellent
FFT-NCC Good Poor Poor Poor Low Excellent
SIFT Keypoints Good Good Weak on repetitive layouts Medium Good
ORB Keypoints Limited Good Weak Low Good
CNN Learned Good Good Potentially good High Low
ViT Learned Good Good Potentially good Very high Low
Proposed Global + local Bounded Bounded Explicit Low–Medium High

The challenge specifically identifies periodic DRAM/FinFET structures as the core difficulty and notes that traditional template matching can generate false positives across the array.

 1. But there is an even more important issue

Before implementing this, we need to answer:

Is the generator correct?

Your physics validator says:

PASS

but that only means:

the numerical invariants we programmed are internally consistent.

It does not prove:

the generated reference actually appears at the declared coordinate in the search image.

Those are different things.

 1. Validation should have 4 levels
LEVEL 1 — Ground-truth validation

Create a completely clean case:

noise = 0
charging = 0
rotation = 0
scale = 1
drift = 0
distortion = 0

Expected:

ZNCC ≈ 100%
FFT ≈ 100%

If not:

Generator/coordinate problem.
18. LEVEL 2 — Retrieval validation

Calculate:

Top-1
Top-3
Top-5
Top-10

Example:

                 Top-10 Recall
Easy                  99%
Medium                96%
Hard                  89%
Adversarial           72%

This tells us whether the global search works.

 1. LEVEL 3 — Ranking validation

If Top-10 recall is high:

Top-10 = 94%
Top-1  = 36%

then:

Ranking is the problem.

We then test:

NCC only
NCC + gradient
NCC + phase registration
NCC + periodicity
NCC + center rule

This is exactly where our new solution should outperform plain FFT-NCC.

 1. LEVEL 4 — Robustness validation

Then run controlled stress tests:

Clean
   ↓
Noise
   ↓
Blur
   ↓
Dose
   ↓
Charging
   ↓
Rotation
   ↓
Scale
   ↓
Periodicity
   ↓
Combined adversarial

Report:

Condition Top-1 Top-5 Top-10 Error
Clean    
Noise    
Blur    
Charging    
Rotation    
Scale    
Periodic    
Combined    

Now we know exactly where the algorithm works.

 1. The benchmark that will decide our architecture

I want you to add these metrics to benchmark_120_harness.py:

retrieval_top1
retrieval_top3
retrieval_top5
retrieval_top10

ranking_accuracy

mean_error
median_error
p95_error

runtime_mean
runtime_p95

Then generate:

Example
                 RETRIEVAL       FINAL
Method          Top-10      ≤5 px
--------------------------------------

ZNCC              42%         36%
FFT-NCC           45%         36%
FFT+Gradient      51%         32%
Proposed          93%         87%

That would be a real breakthrough.

Because then we can say:

FFT retrieves candidates efficiently, while our structural registration and periodicity-aware ranking converts candidate recall into accurate localization.

That's a much stronger story than:

"We added Phase Congruency and Radon."

 1. What we should NOT do next

Don't:

❌ add a CNN

❌ add a Transformer

❌ add a GNN

❌ increase the number of scale/rotation templates

❌ optimize PC from 212 ms to 150 ms

❌ tune random thresholds

❌ generate 10,000 samples yet

❌ make the PPT claim 90% accuracy

The current benchmark has not earned those claims.

 1. What we DO next
Phase A — Diagnose
20 clean samples
       ↓
ground-truth visual check
       ↓
Top-K recall
       ↓
correlation maps
Phase B — Build the better core
FFT-NCC

+

gradient
+
local phase registration
+
Top-K ranking
Phase C — Add periodicity
Δscore
+
spectral/autocorrelation periodicity
+
center rule
Phase D — Subpixel
5×5 local refinement
Phase E — Benchmark
120 cases
+
1000-case stress set
Phase F — Optimize

Only after accuracy is established.

 1. The final story for the PPT becomes very simple
INPUT

High-resolution reference SEM + low-resolution search SEM

↓

SCALE NORMALIZATION

Convert 100× reference to approximately 100×100 search-scale template

↓

GLOBAL SEARCH

FFT-based structural correlation

↓

CANDIDATE RETRIEVAL

Keep Top-K possible locations

↓

LOCAL VERIFICATION

Compare structural alignment and local translation

↓

PERIODICITY HANDLING

Detect repeated candidates and apply center-selection rule

↓

SUBPIXEL REFINEMENT

Refine the winning correlation peak

↓

OUTPUT
(x,y)
 ​

That is the solution I would now pursue.

Most importantly, our next code change should not be the new algorithm itself. First add Top-1/3/5/10 recall + correlation-map visualization + clean-case oracle validation. Those three diagnostics will tell us whether our current 36% result is a retrieval problem or a ranking problem. That determines whether the proposed v4 architecture will actually improve anything.
