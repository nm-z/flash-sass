# Measured checkpoint layer maxima and event counts

All measurements in these tables were collected at `6da421bccb68e613ecbdc8142104935deecccb29`.

Each K or V group contains 32 components; each row covers 16 groups per position. Maxima are pre-clamp and pre-floor. Counts are groups, not individual clipped components.

## 128 positions

time: 8348.5 ms for 128 positions run (0 from the snapshot; 65.22 ms per position run; first token 8348.5 ms; 1154 phases; programs 4 ms; 0 paired positions)

| Layer | Die | K maximum | V maximum | Underflow K / V | Clipped K / V | Would overflow K / V |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1 | 6.86173916 | 4.26786804 | 0 / 0 | 0 / 0 | 0 / 0 |
| 7 | 1 | 6.66231155 | 10.6577444 | 0 / 0 | 0 / 0 | 0 / 0 |
| 11 | 1 | 8.29654598 | 16.9111023 | 0 / 0 | 0 / 0 | 0 / 0 |
| 15 | 1 | 8.39248753 | 8.010993 | 0 / 0 | 0 / 0 | 0 / 0 |
| 19 | 4 | 8.86703205 | 20.0474186 | 0 / 0 | 0 / 0 | 0 / 0 |
| 23 | 4 | 8.47018147 | 10.7908268 | 0 / 0 | 0 / 0 | 0 / 0 |
| 27 | 4 | 9.43380547 | 34.5206795 | 0 / 0 | 0 / 0 | 0 / 0 |
| 31 | 4 | 9.2677536 | 8.64970016 | 0 / 0 | 0 / 0 | 0 / 0 |
| 35 | 5 | 8.82504749 | 8.68432045 | 0 / 0 | 0 / 0 | 0 / 0 |
| 39 | 5 | 10.2237282 | 14.9922609 | 0 / 0 | 0 / 0 | 0 / 0 |
| 43 | 5 | 8.98132229 | 5.82264662 | 0 / 0 | 0 / 0 | 0 / 0 |
| 47 | 5 | 15.7117481 | 7.10601521 | 0 / 0 | 0 / 0 | 0 / 0 |

Coverage: 49152 groups. Other totals: zero_groups=0, subnormal_scale_groups=0, nonfinite_input_groups=0, nonfinite_stored_scales=0, scale_mismatches=0, missing_slots=0.

## 32768 positions

time: 1498564.5 ms for 32768 positions run (0 from the snapshot; 45.73 ms per position run; first token 1498564.4 ms; 1154 phases; programs 4 ms; 0 paired positions)

| Layer | Die | K maximum | V maximum | Underflow K / V | Clipped K / V | Would overflow K / V |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1 | 6.80025387 | 5.39559078 | 0 / 0 | 0 / 0 | 0 / 0 |
| 7 | 1 | 8.81341553 | 10.6924686 | 0 / 0 | 0 / 0 | 0 / 0 |
| 11 | 1 | 8.9118681 | 17.5899868 | 0 / 0 | 0 / 0 | 0 / 0 |
| 15 | 1 | 9.4591465 | 13.6026497 | 0 / 0 | 0 / 0 | 0 / 0 |
| 19 | 4 | 10.6280537 | 31.7153568 | 0 / 0 | 0 / 0 | 0 / 0 |
| 23 | 4 | 9.08883095 | 10.6557798 | 0 / 0 | 0 / 0 | 0 / 0 |
| 27 | 4 | 9.80697823 | 37.5961761 | 0 / 0 | 0 / 0 | 0 / 0 |
| 31 | 4 | 10.3292809 | 17.8680458 | 0 / 0 | 0 / 0 | 0 / 0 |
| 35 | 5 | 12.156189 | 34.4803658 | 0 / 0 | 0 / 0 | 0 / 0 |
| 39 | 5 | 12.0009727 | 16.5589466 | 0 / 0 | 0 / 0 | 0 / 0 |
| 43 | 5 | 10.0597868 | 7.46698904 | 0 / 0 | 0 / 0 | 0 / 0 |
| 47 | 5 | 15.6265059 | 8.37063599 | 0 / 0 | 0 / 0 | 0 / 0 |

Coverage: 12582912 groups. Other totals: zero_groups=0, subnormal_scale_groups=0, nonfinite_input_groups=0, nonfinite_stored_scales=0, scale_mismatches=0, missing_slots=0.

## 65536 positions

time: 3419972.0 ms for 65536 positions run (0 from the snapshot; 52.18 ms per position run; first token 3419971.9 ms; 1154 phases; programs 5 ms; 0 paired positions)

| Layer | Die | K maximum | V maximum | Underflow K / V | Clipped K / V | Would overflow K / V |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1 | 7.15722179 | 6.54948282 | 0 / 0 | 0 / 0 | 0 / 0 |
| 7 | 1 | 7.69565868 | 10.5231466 | 0 / 0 | 0 / 0 | 0 / 0 |
| 11 | 1 | 8.42552567 | 17.3760719 | 0 / 0 | 0 / 0 | 0 / 0 |
| 15 | 1 | 10.5841303 | 13.0314732 | 0 / 0 | 0 / 0 | 0 / 0 |
| 19 | 4 | 10.5774193 | 31.9472828 | 0 / 0 | 0 / 0 | 0 / 0 |
| 23 | 4 | 8.89301682 | 10.1083755 | 0 / 0 | 0 / 0 | 0 / 0 |
| 27 | 4 | 9.41734219 | 37.5779762 | 0 / 0 | 0 / 0 | 0 / 0 |
| 31 | 4 | 10.4936094 | 18.3954048 | 0 / 0 | 0 / 0 | 0 / 0 |
| 35 | 5 | 11.183321 | 34.4219017 | 0 / 0 | 0 / 0 | 0 / 0 |
| 39 | 5 | 11.4628191 | 13.7157984 | 0 / 0 | 0 / 0 | 0 / 0 |
| 43 | 5 | 10.0971851 | 7.42768478 | 0 / 0 | 0 / 0 | 0 / 0 |
| 47 | 5 | 15.6141577 | 8.34198666 | 0 / 0 | 0 / 0 | 0 / 0 |

Coverage: 25165824 groups. Other totals: zero_groups=0, subnormal_scale_groups=0, nonfinite_input_groups=0, nonfinite_stored_scales=0, scale_mismatches=0, missing_slots=0.

