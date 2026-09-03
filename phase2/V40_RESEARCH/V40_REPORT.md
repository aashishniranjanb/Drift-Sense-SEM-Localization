# V40 RESULT

REAL MATCHES:
N = 57

PERIODIC WRONG:
N = 20 (filtered for v25_ml_score > 0.5 Top-1 false)

representation agreement:
REAL = identical responses across frequencies
WRONG = identical responses across frequencies

score std:
REAL = 0.1205
WRONG = 0.1299

rank stability:
REAL = Stable (all frequency responses peak at the same locations)
WRONG = Stable (the exact same local structure shifts identically)

GT Top-1 improvement:
V25 = N/A
V40 = 0

runtime:
V25 = ~3.2s
V40 = ~4.5s (requires multi-filter evaluation)

VERDICT:
KILL
