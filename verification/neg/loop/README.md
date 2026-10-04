# The verification loop for **Neg**

One iteration, converged. Run on 2026-10-04 by the multi-agent system (`agents/`), which wrote
the document and verified it in the same run: `ops/neg.md`
(md5 `2d7670dd97df1792549cc668b81cb4a0`, 327 lines).

| | |
|---|---|
| specification | written in round 1 (13 180 characters), clean under `assets/lint_spec.py`, and `ok` on the first compliance review — no blocking finding, no amendment |
| iteration 001 | `iter-001-impl.py` — a fresh agent, given `ops/neg.md` and nothing else |
| test | `iter-001-report.json`, from `assets/specloop.py` |
| amendment | **none** — the document the implementation read is the document published as `ops/neg.md`, and it was not changed afterwards |

## Result

```
cases : 145 run, 145 passed, 0 mismatched, 0 open, 3 NaN-payload
sweep : the document's worked examples                             3 examples, as documented vs ORT vs implementation
sweep : float special values (document vs ORT vs implementation)    float16 23 values, float32 23 values, float64 23 values
sweep : int domain (document vs ORT vs implementation)              4 types: int8 exhaustive, int16 208 values, int32 208 values, int64 208 values
sweep : real section (exact rationals, no ONNX Runtime)             6 checks against fractions.Fraction
OK: no mismatch
```

Coverage, as the case module declares it over the document's own Contents anchors:
`{"real": 6, "float": 81, "int": 74}` — no section without a case. The three sections of the
document are the mathematical family, the three IEEE 754 types `float16`, `float`, `double`,
and the four signed types `int8` … `int64`; **Neg** admits no unsigned type in ONNX, so there
is no `uint` section, and `bfloat16` — which ONNX does admit — is outside the SONNX profile
and is named in the float section as such.

The three NaN-payload differences are the signalling-NaN operand of each float type: the
specification's `E_NEG_FLOAT_FUNC_0010` states that "every quiet NaN of the type of $X$ is a
conforming result" and that the sign and the payload of that NaN are not specified, so they
are not mismatches.

## The blind reader's `DECISION:` points, classified

Ten points. None of them is about a **value**, so none of them is a defect of the document;
listed here so the next reader can see they were considered.

1. the linked "[General restrictions](./../common/general_restrictions.md)", which is another
   file and was out of scope for the task — the same line `ops/add.md`, the guidelines and the
   template carry, and the reader's only remark about incompleteness;
2. the tensor representation and the API: numpy arrays in, a numpy array of the same dtype out
   (the document speaks of a tensor and names no library);
3–4. the types the document gives no section to — unsigned integers, `bool`, complex — and the
   NaN payload: no validation and no canonicalization added;
5. the signed integer overflow realized by the host type's own wraparound, and the language's
   overflow warning left unsuppressed, the document calling the overflow nominal and saying
   nothing about warnings (the same choice as the second blind reader of `ops/mul.md`, and the
   opposite of one of the two readers of `ops/add.md` — see `../../add/loop/README.md`);
6–10. "No error condition", "no attribute", "performs no division", "none of the operations of
   IEEE 754 section 7.2" — read as constraints on the implementation and honoured by not adding
   anything.

The single `return np.negative(X)` is where all ten land: the document fixes the values, and
the language's unary negation produces them, wraparound included.

## Detector check

The case set was checked against the misreading the integer section invites — that
$-(-2^{n-1}) = 2^{n-1}$ is not representable, so `int8` `-128` negates to `127` (saturation)
instead of to `-128`. It reports

```
cases     : 145 run, 136 passed, 9 mismatched, 0 open, 3 NaN-payload
SWEEP MISMATCH int Example (int8): documented [6, 0, -100, -128], implementation [6, 0, -100, 127]
SWEEP MISMATCH int8:  ONNX Runtime vs the implementation: x=-128        want=-128        got=127
SWEEP MISMATCH int16: ONNX Runtime vs the implementation: x=-32768     want=-32768     got=32767
SWEEP MISMATCH int32: ONNX Runtime vs the implementation: x=-2147483648 want=-2147483648 got=2147483647
SWEEP MISMATCH int64: ONNX Runtime vs the implementation: x=-9223372036854775808 want=… got=9223372036854775807
NOT OK: the specification needs an amendment
```

All four signed types are caught, in the exhaustive `int8` sweep and on the boundary value of
the wider three, and the document's own Example catches it as well — which is the point of
replaying the examples: a number written wrongly in the document is caught even when the
implementation agrees with it.

## The five attempts before this one

This document is the sixth run of the loop on **Neg**. The five earlier ones stopped on
defects of the *system*, not of the document: a writer heading rule that named a family twice,
an ONNX definition that was the documentation site's navigation, a cut-off writer answer
written to disk as a document, a verifier that invented a `uint` family for an operator that
admits none, and a coverage reader that read the literal of a computed `COVERAGE` dictionary
and called a 156-case set vacuous. Each cost an attempt and each is now a guard in
`agents/README.md` with a scenario in `agents/tests/test_graph_offline.py`. The drafts and the
records are archived in `agents/runs/Neg/attempt-1/` … `attempt-5/`.

## Re-running

```
~/Venvs/sonnx/bin/python .claude/skills/operator-spec/assets/specloop.py \
    --spec ops/neg.md --op Neg --cases verification/neg/cases.py \
    --impl verification/neg/loop/iter-001-impl.py
```

The cases are in `../cases.py`; the `int` and `float` sections also carry sweeps there, and the
`real` section is checked against exact rationals because ONNX Runtime cannot be asked about a
real number — a real-number case put in `cases()` is recorded as *open*, which is not a check.
A section added to `ops/neg.md` needs its cases in `../cases.py` before the next run, or the run
passes vacuously for that section (the loop itself stops with `uncovered-section`). A pass that
follows an amendment to `ops/neg.md` is not convergence: re-spawn the blind agent every time
(see `../../../.claude/skills/operator-spec/references/verification.md`, step 4).
