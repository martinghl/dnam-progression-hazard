"""Siamese Hazard: context-conditioned paired-methylation progression model.

The package implements the headline model of the project: a Siamese
discrete-time hazard network that maps two longitudinal DNA-methylation draws
``(X0, X1, dt)`` to calibrated, monotone cumulative risks ``(p2, p3, p5)`` for
conversion from MCI to AD at 2, 3 and 5 years.

Key modules
-----------
- ``model``              : ``ContextSiameseHazardNet`` (residual context projector + hazard core)
- ``context``            : context-feature loading, z-scoring and block scaling (the "E" matrix)
- ``data``               : design-file loading, beta->M conversion, masked hazard labels
- ``calibration_layer``  : per-fold multi-head Platt calibration
- ``metrics``            : AUC / PR-AUC / Brier / ECE / decision-curve analysis
- ``train``              : 5-fold CV training + calibration + composition CLI
- ``explain``            : per-CpG permutation-importance interpretability panel
"""

__version__ = "1.0.0"
