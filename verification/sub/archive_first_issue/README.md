# First-issue verification

`impl_from_spec_first_issue.py` is the implementation written from the **first issue**
of `ops/sub.md` (before the V1–V4 amendments of revision 2026-10-03), together with the
run recorded in `../../results.md`.  It is kept as the evidence for the two
divergences found against ONNX Runtime: the float overflow band (V1) and the
undefined NaN payload and NaN of the invalid operation (V2).

It implements the fourth case of the float case analysis, which the amended document
has removed, so it must not be used for the verification of the amended document.
