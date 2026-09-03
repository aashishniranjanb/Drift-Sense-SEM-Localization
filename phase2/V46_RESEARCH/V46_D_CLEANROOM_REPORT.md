# V46-D CLEAN-ROOM REPORT

## 1. Localization Tiers (Primary Gate: C_margin_0.10)
| Tier | V25 | V46 |
|---|---|---|
| <=1 | 18 | 34 |
| <=2 | 0 | 0 |
| <=5 | 0 | 1 |
| >5 | 122 | 105 |

## 2. Set A Localization
| <=1 | 13 | 21 |
| <=2 | 0 | 0 |
| <=5 | 0 | 1 |
| >5 | 57 | 48 |

## 3. Set B Localization
| <=1 | 5 | 13 |
| <=2 | 0 | 0 |
| <=5 | 0 | 0 |
| >5 | 65 | 57 |

## 4. Rejection (Primary Gate: C_margin_0.10)
- V25: TP=64 FP=15 FN=76 TN=25 | F1=0.584
- V46: TP=140 FP=39 FN=0 TN=1 | F1=0.878
- New Absent False Accepts: 24

## 5. Gate Ablations
Gate | Rescued | Broken | Net | A gain | B gain | New Absent FP | Total Absent FP
---|---|---|---|---|---|---|---
A_3_of_4 | 18 | 2 | 16 | 8 | 8 | 25 | 40
B_3_of_4_strength | 17 | 2 | 15 | 8 | 7 | 20 | 35
C_margin_0.00 | 18 | 2 | 16 | 8 | 8 | 25 | 40
C_margin_0.02 | 18 | 2 | 16 | 8 | 8 | 25 | 40
C_margin_0.05 | 18 | 2 | 16 | 8 | 8 | 25 | 40
C_margin_0.10 | 17 | 0 | 17 | 9 | 8 | 24 | 39
C_margin_0.15 | 11 | 0 | 11 | 4 | 7 | 24 | 39
D_4_of_4 | 11 | 0 | 11 | 6 | 5 | 12 | 27
Protected_Tiers | 18 | 3 | 15 | 7 | 8 | 24 | 39
