# Final participant-level bootstrap 95% CIs (representative configurations)

2,000 stratified (by outcome class) participant-level bootstrap resamples, seed 12345, on the frozen OOF predictions of the representative configuration of each arm. **These are CIs for representative configurations, not for the sweep medians.** ECE is reported as optional/secondary.

| Horizon | Model | N | pMCI | AUC [95% CI] | PR-AUC [95% CI] | Brier [95% CI] | ECE [95% CI] (optional) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2y | No-context (PCA) | 186 | 18 | 0.713 [0.593, 0.824] | 0.273 [0.146, 0.473] | 0.083 [0.077, 0.088] | 0.018 [0.008, 0.059] |
| 3y | No-context (PCA) | 172 | 36 | 0.635 [0.528, 0.731] | 0.338 [0.246, 0.481] | 0.160 [0.155, 0.165] | 0.011 [0.006, 0.072] |
| 5y | No-context (PCA) | 125 | 48 | 0.627 [0.531, 0.724] | 0.511 [0.421, 0.632] | 0.230 [0.216, 0.244] | 0.078 [0.056, 0.129] |
| 2y | Sequence context | 186 | 18 | 0.811 [0.721, 0.889] | 0.308 [0.195, 0.539] | 0.079 [0.068, 0.089] | 0.040 [0.020, 0.072] |
| 3y | Sequence context | 172 | 36 | 0.677 [0.576, 0.773] | 0.429 [0.313, 0.573] | 0.150 [0.138, 0.162] | 0.064 [0.041, 0.125] |
| 5y | Sequence context | 125 | 48 | 0.690 [0.596, 0.782] | 0.614 [0.511, 0.725] | 0.213 [0.195, 0.230] | 0.057 [0.056, 0.141] |
| 2y | Seq+Bio context | 186 | 18 | 0.797 [0.687, 0.889] | 0.324 [0.204, 0.560] | 0.082 [0.073, 0.091] | 0.046 [0.024, 0.083] |
| 3y | Seq+Bio context | 172 | 36 | 0.681 [0.592, 0.772] | 0.363 [0.277, 0.541] | 0.158 [0.147, 0.169] | 0.045 [0.024, 0.103] |
| 5y | Seq+Bio context | 125 | 48 | 0.680 [0.594, 0.778] | 0.577 [0.479, 0.704] | 0.219 [0.197, 0.239] | 0.072 [0.058, 0.144] |

*Lower Brier/ECE = better. CIs are wide at 2y/3y because positive (pMCI) events are scarce (18, 36, 48).*
