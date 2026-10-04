# The verification loop for **Add**

One iteration, converged. Run on 2026-10-03 against `ops/add.md`
(md5 `9dfe1947ed325b2a3eb3082ec3cb06eb`, 550 lines), which had already been through the loop by
hand.

| | |
|---|---|
| iteration | `iter-001-impl.py` — a fresh agent, given `ops/add.md` and nothing else |
| test | `iter-001-report.json`, from `assets/specloop.py` |
| amendment | **none** — the run that follows an unamended step 1 passed, which is convergence |

## Result

```
cases : 205 run, 205 passed, 0 mismatched, 0 open, 3 NaN-payload
sweep : the document's worked examples  5 examples, as documented vs ORT vs implementation
sweep : int domain (document vs ORT vs implementation)
        8 types: int8 exhaustive, int16/int32/int64 249 pairs each,
                 uint8 exhaustive, uint16/uint32/uint64 216 pairs each
sweep : real section (exact rationals, no ONNX Runtime)  2 examples
OK: no mismatch
```

Coverage: the eight machine types of ONNX **Add**, both 8-bit domains exhaustively and the wider
six over their boundary values plus a random sample; 205 individual cases spanning the float value
families (signed zeros, infinities, NaN, the two overflow thresholds and the tie above them,
subnormals, one-ulp and half-way rounding) and the thirteen shape families (rank-0, broadcasting,
zero-sized). The real section, which has no ONNX type, against exact rationals.

The three NaN-payload differences are the signalling-NaN operand of each float type: the
specification's `E_ADD_FLOAT_FUNC_0010` states that every quiet NaN of the operand type is a
conforming result, so they are not mismatches.

## The blind reader's `DECISION:` points, classified

Thirteen points. None of them is about a **value**, so none of them is a defect of the document;
listed here so the next reader can see they were considered.

Interface choices, which the document neither fixes nor needs to:

1. how the language discovers the operand type (numpy dtype kind);
2. how the `real` family is represented (object dtype, falling back to `np.add`, exact for `Fraction`);
3. the return object (a numpy array);
4. how the operands are converted (`np.asarray`);
5. `bool`, `complex`, `longdouble` — the document gives them no section, so no validation is added;
6. the "General restrictions" link, which is another file and was out of scope for the task;
7–8. that the document's mathematics is realized by the language's arithmetic (IEEE 754 add; the
   modulo-$2^n$ of both integer families).

Gaps the document leaves and the *harness* fixes, not the document: mixed operand types and
incompatible shapes are constraints with no error condition, so the language's own promotion and
`ValueError` decide — the case set contains none of these, and the runtime refuses such a model,
which `specloop.py` records as an "open" case rather than a mismatch.

One point worth recording: **warnings**. This reader did not suppress numpy's overflow/invalid
warnings; the earlier blind reader of the same text (`../impl_from_spec.py`) did, citing the
document's disposition table ("every condition of the list is part of the nominal behavior of the
operator; none of them is an error"). Two readers of the same sentence, opposite decisions. It is
not a specification gap: the document specifies the **result**, and both readers produce it — how an
implementation reports a nominal condition is not part of the operator's semantics. No amendment.

## Detector check

The driver was checked against a plausible misreading of the integer rule (reducing the exact sum
modulo $2^n$ unconditionally, as one would for the unsigned family). It reports
`205 run, 177 passed, 28 mismatched` and `NOT OK`, and both sweeps fail — so a wrong reading of the
document is caught, on values and not only on shape.

## Re-running

```
~/Venvs/onnxcheck/bin/python .claude/skills/operator-spec/assets/specloop.py \
    --spec ops/add.md --op Add --cases verification/add/cases.py --impl verification/add/impl_from_spec.py
```

`impl_from_spec.py` is the first blind implementation (the one that went through the by-hand loop);
`loop/iter-001-impl.py` is the one of this automated run. Both pass. The cases are in
`../cases.py`; a section added to `ops/add.md` needs its cases there before the next run, or the run
passes vacuously for that section.
