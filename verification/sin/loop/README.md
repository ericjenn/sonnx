# The verification loop for **Sin**

One iteration, converged. Run on 2026-10-03 against `ops/sin.md`
(md5 `7ec516fe07ca48c586f95925fdcb630c`, 283 lines, 16 596 bytes), written the same day.

| | |
|---|---|
| iteration 001 | `iter-001-impl.py` — a fresh agent, given `ops/sin.md` and nothing else |
| test | `iter-001-report.json`, from `assets/specloop.py` |
| amendment | **none** — the run that follows an unamended step 1 passed, which is convergence |

The document was frozen while the blind agent read it, and the md5 above is the md5 of the text
that agent read: the run of `iter-001` is the run of the current text, so one iteration is the whole
loop here. Two wording changes that had been considered are **not** applied for that reason — a
pass that follows an amendment proves nothing, and neither of them was a value or a case.

## Result

```
cases : 49 run, 49 passed, 0 mismatched, 0 open, 11 NaN-payload
sweep : the document's rules for special values, and the identity of a small value
        ONNX Runtime and the implementation, per type: 49 values, small ones within
        0 units (float16, float32) and 1 unit (float64, ONNX Runtime) of their own sine
sweep : the document's worked examples
        float Example 2: 3 documented values, faithful to the exact sine
        float Example 3: 4 documented values, faithful to the exact sine
sweep : the real section (60-digit arithmetic, no ONNX Runtime)
        real Example 1: 6 values; real Example 2: 3 values
sweep : the whole float16 domain (65536 values)
        ONNX Runtime: 4 of 63488 ordinary values depart, worst 0.5 units
        the implementation: 4 of 63488 ordinary values depart, worst 0.5 units
sweep : the correctness of a result (reported, not required: see V1)
        ONNX Runtime: 106 of 540 differ (20%), worst 0.5 units of 2^-23, 1 unit in
        the last place of the result; 387 of 540 differ (72%), worst 2.0 units of
        2^-52, 54 units in the last place of the result
        the implementation (numpy): 94 of 540 (17%) and 0 of 540 (0%)
sweep : the oddness of the result (bit for bit, no reference)
        3 types: float16 3206 pairs, float32 3206 pairs, float64 3206 pairs
OK: no mismatch
```

Coverage: the three floating-point types of ONNX **Sin** (opset 14, ONNX Runtime 1.30, CPU
provider, x86-64, numpy 2.5.3), the whole `float16` domain bit pattern by bit pattern and the wider
two over 540 values drawn uniformly from $[0, 10]$ and their opposites plus 40 values of larger
magnitude; 49 individual cases — the signed zeros, the infinities, NaN operands of three payloads
per type, subnormal operands, the smallest subnormal of both signs, a null and a subnormal
together, the document's `double` Example 1, and the six shape families (rank-0, `(1,)`, `(2,3)`,
`(0,3)`, `(0,)`, `(2,0)`) per type. The real section, which has no ONNX type, against the exact
sines of its two examples computed in 60-digit decimal arithmetic.

Two things in that block are deliberately not passes to be counted. The **correctness sweep** is
reported and never required: the document's warning V1 states that the value of the operator is the
correctly rounded sine and refers a departure from it to the accuracy guidelines, so a count of
departures measures an implementation and not the rule; the driver fails it only beyond `SLACK` of
four units in the last place of the value 1, which no implementation of the three examined comes
near except at a small result. The **real section** is the one part ONNX Runtime cannot be asked
about; the driver replays its examples against `sin_oracle.py`, whose sine is Decimal arithmetic
(π by Machin's formula, the argument reduced modulo $2\pi$ at a working precision wider than its
magnitude, rounded to the binary format by exact rational arithmetic) and whose comparison is
skipped when the exact sine lies within $10^{-30}$ of a rounding midpoint.

The eleven NaN-payload entries are the invalid operation of an infinity and the NaN operands of
every type: `E_SIN_FLOAT_FUNC_0010` states of the NaN it produces that "its sign and its payload are
not specified … every quiet NaN of the type of $A$ is a conforming result", so they are not
mismatches.

## The blind reader's `DECISION:` points, classified

Fourteen points. None of them is about a **value**, so none of them is a defect of the document;
listed here so the next reader can see they were considered.

Interface choices, which the document neither fixes nor needs to:

1. one entry point serving both sections, the two signatures being identical and the document
   silent on dispatch (the reader applied the float rule to a floating operand);
2. no exact or symbolic representation of a real operand — the reader's operands are floating-point
   values, and the document names no representation for the real family (the harness's own
   convention, an object array of `Fraction`, is not in the document and is not told to the reader);
3. how the three float cases are realized — one `numpy.sin` call rather than an explicit
   isinf/isnan/iszero branch, the document giving results and not method;
4. how the type of a result is preserved and how an operand type outside the document's list is
   treated (no validation, no cast);
5. what object holds a rank-0 result (the document prescribes no container);
6. NaN sign and payload taken from the host;
7. roundTiesToEven taken as the ambient rounding mode, the document naming the attribute but not
   a way to select it;
8. `float16` taken through the host's own path, with no upcast and round-back;
9. array-likes converted by the host, the document speaking of tensors.

Two points are worth a line each, because they look semantic and are not.

**Point 3 of the reader's list — the accuracy of the result.** The reader implemented
`numpy.sin`, which is not correctly rounded for `float32` and `float64`, and cited V1 to justify it.
That is exactly what V1 was written for: the document states the value (the correctly rounded
sine), V1 states that implementations of the sine do not all return it, and it points at
`other/accuracy.md` for the difference. It is the decision the document leaves *and says it leaves*,
which is what the skill asks for — not a gap. The one place it is sharp is `float64`, where ONNX
Runtime departs on 72% of a uniform sample while `numpy.sin` under the same sweep departs on none;
the document's value is the one numpy attains there, and no case in `cases.py` is decided by the
question.

**Point 11 of the reader's list — warnings.** The reader left numpy's `RuntimeWarning` for the
invalid operation of an infinity to propagate, on the ground that the document names the condition
and says nothing about signalling. This is the same point, decided the same way, as both readers of
`ops/mul.md`, and decided the other way by one reader of `ops/add.md` — see `../../add/loop/README.md`
and `../../mul/loop/README.md`. It is not a gap: the document specifies the result, and how an
implementation reports a nominal condition is not part of the operator's semantics. No amendment.

## Detector check

The driver was checked against the two misreadings of `ops/sin.md` a reader is most likely to make.
They are in `detector-impl.py`, selected by the environment variable `SIN_DETECTOR`, with the
implementation of iteration 001 replaced by the wrong one:

- `SIN_DETECTOR=zero` — **the null rule of the sum**, carried over from `ops/add.md` and `ops/mul.md`,
  where a result whose exact value is null is $+0$. The rule is right there and wrong here: the sine
  of $-0.0$ is $-0.0$, the document requiring the sign bit of the operand to pass to the result. It
  reports `49 run, 39 passed, 10 mismatched, 0 open` and `NOT OK`; the ten are the signed zeros, the
  null-and-subnormal case and the shape cases of all three types, and the document's own `double`
  Example 1, which contains a $-0.0$. Four sweeps fail as well — the per-type special-value rules
  ("a null operand does not give that zero"), the whole `float16` domain, and the oddness sweep,
  which catches it without any reference implementation at all.
- `SIN_DETECTOR=period` — **the periodicity read on the values of the type**: reduce the operand
  modulo the value of the type nearest $2\pi$ and take the sine of the remainder. This is what the
  paragraph "**The sine is not periodic on the values of a floating-point type**" exists to prevent.
  It reports `49 run, 39 passed, 10 mismatched, 0 open` and `NOT OK`; the meaningful failures are
  float Example 2 (a departure of 1 099 754 units in the last place of the value 1, the sine of the
  nearest `float` to $2\pi$ computed as $0$ instead of $1.7484555\text{e-}07$), the document's
  examples sweep, the ordinary-domain rule of the `float16` sweep ("a non-null operand gives a null
  result"), the correctness sweep on both wider types, and the oddness sweep.

So both the sign of a null result and the periodicity of the sine are held by the case set on
values, and not only on shape.

## Re-running

```
~/Venvs/onnxcheck/bin/python .claude/skills/operator-spec/assets/specloop.py \
    --spec ops/sin.md --op Sin --cases verification/sin/cases.py \
    --impl verification/sin/loop/iter-001-impl.py \
    --json verification/sin/loop/iter-001-report.json
```

The cases are in `../cases.py`, the reference sine in `../sin_oracle.py`. A section added to
`ops/sin.md` needs its cases there before the next run, or the run passes vacuously for that
section. Any amendment to `ops/sin.md` voids this pass: re-spawn the blind agent every time (see
`../../../.claude/skills/operator-spec/references/verification.md`, step 4).
