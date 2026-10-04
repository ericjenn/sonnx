# The verification loop for **Mul**

Two iterations. Run on 2026-10-03 against `ops/mul.md`, written the same day.

| | |
|---|---|
| iteration 001 | `iter-001-impl.py` — a fresh agent, given `ops/mul.md` and nothing else |
| test | `iter-001-report.json` — **passed, no amendment** |
| iteration 002 | `iter-002-impl.py` — a second fresh agent, after the amendment below |
| test | `iter-002-report.json` — **passed, no amendment** |

Convergence is the second line of that table: the amendment below was made after iteration 001, so
the run of 001 proves nothing about the current text, and the text that is now verified
(md5 `e4b4b3d7966c947ecfc66554592bd5b1`, 577 lines) is the text iteration 002 read.

## What the amendment was

Two defects of the document were found by reading the signed integer section for the case that
matters most there — that the product of two large positive values can be negative. Both are in the
explanatory prose inside the red span, and neither changes the congruence, which is why the two
implementations are identical and the second run is a convergence and not a re-check of a patched
implementation.

1. **The exact product of two signed values was bounded wrongly.** The document gave
   $[-2^{2n-2}+2^{n-1}, 2^{2n-2}-2^n+1]$. The upper bound is the largest product of two *positive*
   values, $(2^{n-1}-1)^2$; the true maximum is $(-2^{n-1})^2 = 2^{2n-2}$, reached by the two most
   negative values. For `int8` the document said $16129$ where the product reaches $16384$. Fixed,
   with the two pairs that reach the bounds named, since that the extreme product comes from the
   two smallest values is exactly the counter-intuitive part of the paragraph.
   Checked by brute force over `int8` (`[-16256, 16384]`) and, for `int16`, `int32` and `int64`,
   over the four corners of the domain, the product being bilinear on a box.
2. **`float` Example 2 named values that do not appear in it**: "the product of the two negative
   values $-1$ and $-2$" where the operands are $-\text{inf}$ and $-2.0$. Fixed.

The behavioural rule the amendment was looked for is **not** one of the defects: the document
already states, in `E_MUL_INT_FUNC_0010`, that "the multiplication of two positive values may give
a negative result, and conversely", illustrates it twice ($11 \cdot 13 = 143$ and $127 \cdot 127 =
16129$, both giving a negative or small result from two positive operands), and carries the same
example in the overflow row of the error-conditions table and in Example 1. The exhaustive `int8`
sweep holds it to ONNX Runtime on all 65 536 pairs.

## Result

```
cases : 199 run, 199 passed, 0 mismatched, 0 open, 3 NaN-payload
sweep : the document's worked examples  6 examples, as documented vs ORT vs implementation
sweep : int domain (document vs ORT vs implementation)
        8 types: int8 exhaustive, int16/int32/int64 249 pairs each,
                 uint8 exhaustive, uint16/uint32/uint64 216 pairs each
sweep : real section (exact rationals, no ONNX Runtime)  2 examples
OK: no mismatch
```

Coverage: the eight machine types of ONNX **Mul** (opset 14, ONNX Runtime 1.30, CPU provider,
numpy 2.5.3), both 8-bit domains exhaustively and the wider six over their boundary values plus a
random sample; 199 individual cases — 29 per float type (the signed zeros, the infinities, the
invalid operation `0 × ±inf`, NaN, the exact product, overflow to infinity, the largest finite
value and the value just below it, subnormal results, underflow to zero, and products needing
rounding), 14 per integer type (the extreme products, the sign flips, and the products that are
null although no operand is), and the thirteen shape families (rank-0, broadcasting, zero-sized
dimensions) across all eleven types. The real section, which has no ONNX type, against exact
rationals.

The two sweeps that carry the weight are the ones the document cannot be checked without. The
integer sweep compares a **literal transcription** of the congruence — `doc_int` in `../cases.py`,
written from the sentence of each of the two integer sections and not from the implementation —
with ONNX Runtime over the whole 8-bit domains and over the boundary values of the wider types; it
is a second reading of the document, so a disagreement there is a defect of the document. The
examples sweep replays every worked example of every section, as documented, against ONNX Runtime
*and* against the implementation, so a number written wrongly in the document is caught even when
the implementation is right.

The three NaN-payload differences are the signalling-NaN operand of each float type: the
specification's `E_MUL_FLOAT_FUNC_0010` states that every quiet NaN of the operand type is a
conforming result, so they are not mismatches.

## The blind readers' `DECISION:` points, classified

Eighteen points over the two iterations. None of them is about a **value**, so none of them is a
defect of the document; listed here so the next reader can see they were considered. The second
reader's list (iteration 002) is the one that goes with the verified text:

1. the `real` family has no host type, so no code path is dedicated to it — the same interface
   choice as the first reader's, deciding the other way (the first used the caller's dtype, the
   second let a floating host array fall under the float clause); the real sweep's `Fraction`
   operands are exact under both readings;
2. type-family selection by host dtype kind, the document being silent on run-time dispatch;
3. `np.asarray` on the operands, the document speaking only of tensors;
4. the congruence realized by the host type's own multiplication and its wraparound, the document
   stating a congruence and not a mechanism;
5. the whole float clause delegated to IEEE 754 multiplication — the document specifies values,
   not mechanisms;
6. the host's warning state left untouched, the document calling overflow nominal but saying
   nothing about warnings (the same point, the same way, as in the first round, and the opposite
   way from one of the two readers of `ops/add.md` — see `../../add/loop/README.md`);
7. the rank-0 result passed through `np.asarray` so that `C` is always a tensor, the host returning
   a scalar;
8–9. the two constraints treated as preconditions with no validation added, and the operands
   multiplied in the order given although the document says they play the same role;
10. broadcasting, rank-0 operands and zero-sized dimensions delegated to the host, whose rules the
    document describes verbatim.

Point 4 is worth one line. For the integer families the document fixes the type of the result (`C`
has the type of `A` and `B`) *and* gives the product as a congruence, and those two together leave
exactly one value: there is no way to be wrong about the residue and still return an element of the
type. Every attempt at a wrong-value mutant of the integer rule ended as an out-of-range value or an
error, never as a different number — see the detector check below. The clause that carries the
content is therefore not the congruence but the two sentences that say what a reader is likely to
get wrong: that the reduction may be applied several times, and that the product of two non-zero
values may be zero.

## Detector check

The driver was checked against the reading a reader of `ops/add.md` is most likely to carry over:
the null rule of the *sum*, that a result whose exact value is null is `+0`, applied to the product.
It reports

```
cases : 199 run, 193 passed, 6 mismatched, 0 open, 3 NaN-payload
SWEEP MISMATCH float Example 2: documented [1.0, nan, inf, -0.0, -0.0, 0.0], implementation [1.0, nan, inf, 0.0, 0.0, 0.0]
SWEEP MISMATCH float Example 4: documented [5.562684646268003e-309, -5.562684646268003e-309, -0.0, 0.0], implementation [5.562684646268003e-309, -5.562684646268003e-309, 0.0, 0.0]
NOT OK: the specification needs an amendment
```

The six cases are `float16/float32/float64: signed zeros` and `…: underflow to zero`, and the
differences are `-0.0` against `0.0`, bit pattern against bit pattern, on all three types. Both
worked examples that a null product with one negative operand appear in are caught as well, so the
document's own numbers carry the rule too.

This check also found a defect in the driver: `specloop.py` printed a failing sweep's summary but
not the reason, so a failure was visible in the exit status and invisible on the console. It now
names each failing sweep entry. The check above is the run that shows it.

## Re-running

```
~/Venvs/onnxcheck/bin/python .claude/skills/operator-spec/assets/specloop.py \
    --spec ops/mul.md --op Mul --cases verification/mul/cases.py \
    --impl verification/mul/loop/iter-002-impl.py \
    --json verification/mul/loop/iter-002-report.json
```

The cases are in `../cases.py`; a section added to `ops/mul.md` needs its cases there before the
next run, or the run passes vacuously for that section. `iter-001-impl.py` also still passes — it
was written to a text that differs from the current one only in the two explanatory sentences of
the amendment. A pass that follows an amendment to `ops/mul.md` is not convergence: re-spawn the
blind agent every time (see `../../../.claude/skills/operator-spec/references/verification.md`,
step 4).
