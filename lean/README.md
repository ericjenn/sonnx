# The Lean 4 build root

This directory is the Lake package the SONNX pipeline uses to type-check and execute the Lean
modules it generates. It holds two libraries: `Sonnx`, the one shared Lean library every
generated module imports, and `CGen`, a C code generator that renders operator kernels from a
Lean description. No operator-specific Lean code lives here; neither library depends on the
other.

| file | purpose |
|---|---|
| `lakefile.lean` | the package definition; the `Sonnx` library is its default target, `CGen` its second target |
| `lean-toolchain` | the pinned Lean toolchain |
| `lake-manifest.json` | the dependency manifest (no dependencies) |
| `Sonnx.lean` | the library root: imports every module below |
| `Sonnx/Prelude.lean` | list lemmas; `TypeConsistent` |
| `Sonnx/Tensor.lean` | `Tensor` (shape and flat data), `size`, `validIdxB`, `offset`, `Tensor.read`, `wellFormed` |
| `Sonnx/Broadcast.lean` | `padTo`, `bcastDim`, `bcastShape`, `bcastIdx`, `Broadcasts`, `BroadcastCompatible`, `BroadcastShapeIs` |
| `Sonnx/Index.lean` | `indices` over a shape, and its offset lemma |
| `Sonnx/Int.lean` | `CongM`, `intModulus`/`intMin`/`intMax`, the signed and unsigned ranges, `uintWrap`/`sintWrap` with range, congruence and uniqueness, `UIntN`/`SIntN` as bit patterns |
| `Sonnx/Float.lean` | `FloatKind`, `FloatFormat`, `FVal`, representability and `RoundNearestEven`, `Overflows`, `RoundToFormat` |
| `Sonnx/Elementwise.lean` | `binary` (the functional form) and `PointwiseBinaryRel`/`ElementwiseBinaryRel` (the relational form) |
| `CGen.lean` | the generator root: imports the three modules below (not the drivers, which define `main`) |
| `CGen/Ast.lean` | the C11 datatypes: `CType`, `UnOp`/`BinOp`, `Expr`, `Stmt`, `Param`, `Fun`, `Decl`, `CFile` |
| `CGen/Print.lean` | the precedence-aware printer: `renderExpr`, `renderType`/`declType`, `renderStmt`, `renderFun`, `renderFile` |
| `CGen/Kernel.lean` | `Elementwise`, `rhs`, `kernel`, `funsFor`/`funs`, `operatorsFileFor`/`operatorsFile`, `headerName`, `headerFileFor`, `sourceFileFor`, `binOpName` |
| `CGen/Emit.lean` | the emit driver: writes `operatorsFile` to a path or to stdout |
| `CGen/Probe.lean` | the printer probes; exits nonzero on a mismatch |

Build the libraries first; `lake build Sonnx` and `lake build CGen` write
`.lake/build/lib/lean/{Sonnx,CGen}/*.olean` onto the `LEAN_PATH` that `lake env` exports, so
`lake env lean` can then `import Sonnx`:

```sh
cd lean
lake build Sonnx
printf 'import Sonnx\n#check @Sonnx.Tensor\n#check @Sonnx.binary\n' \
  | lake env lean --stdin
```

The pipeline's `lean.build_library()` runs `lake build Sonnx CGen` together, because the C
stage's derivation trailer imports `CGen.Derive`/`CGen.Print`; an olean that does not exist
fails that compile. The generated formal module itself needs no `CGen` import.

## The import contract

A generated module starts with `import Sonnx` (or the one library module it needs,
`Sonnx.Float`, say) and may otherwise import only Lean core (`Init`, `Std`, `Lean`). `CGen` is
still accepted for older artifacts that carry the retired descriptor, but new modules do not
import it. A generated module never
imports another operator's generated module, Mathlib, Batteries or Lake, and never redeclares a
name the library exports. The pipeline enforces this mechanically for both the formal
specification and the generated conformance runner: `agents/sonnx_agents/lean.py` rejects a
disallowed import and a redefinition of a shared top-level name, and the formalizer prompt
carries the rule and the API table.

The library is what makes the tensor formalization operator-independent: a generated module
defines only its own relations, example tensors and claims. Trace markers for the document's
constraints usually name a shared declaration (`Sonnx.TypeConsistent`, `Sonnx.BroadcastCompatible`,
`Sonnx.BroadcastShapeIs`), while the operator's function and example markers name its own
declarations.

## The library digest

`Sonnx` is part of the artifact's identity. `lean.library_digest()` is a SHA-256 over
`lakefile.lean`, `lean-toolchain`, `Sonnx.lean` and `Sonnx/*.lean` in sorted relative-path order.
It is written into every generated module as `@library-digest`, into the trace report and the
conformance manifest as `library_sha256`, and is re-read at every freshness check. Editing the
library therefore makes every artifact generated against the previous library stale, by design:
regenerate rather than reuse. The pipeline also builds the library before the formalizer is
asked and before the conformance runner is generated, and fails the stage (`lean-failed`, or
`lean-conformance-error` with status `library-unusable`) rather than compiling against a library
that does not build.

## The C generator — `CGen`

`CGen` renders C11 source from a Lean description of an operator. `CGen.Ast` is an abstract
syntax for the fragment a kernel needs, `CGen.Print` is a precedence-aware pretty-printer for
it, and `CGen.Kernel` builds the kernels:

```sh
cd lean
lake build CGen
lake env lean --run CGen/Probe.lean          # the printer probes
lake env lean --run CGen/Emit.lean out.c     # writes the translation unit
```

With no path the driver prints the translation unit to standard output; the parent directory of
a path must exist. `lake build CGen` also builds the two driver modules, which the root module
does not import because both define `main`.

`operatorsFile` generates, for every operator (`add`, `sub`, `mul`) and every element type, a
prototype and a kernel over the generated tensor struct:

```c
typedef struct {
  void *data;
  size_t ndim;
  const size_t *shape;
} sonnx_tensor;

void sonnx_sub_i8(const sonnx_tensor *a, const sonnx_tensor *b, sonnx_tensor *c) {
  const int8_t *A = (const int8_t *)a->data;
  const int8_t *B = (const int8_t *)b->data;
  int8_t *C = (int8_t *)c->data;
  size_t n = 1;
  for (size_t d = 0; d < c->ndim; d++) {
    n = n * c->shape[d];
  }
  for (size_t i = 0; i < n; i++) {
    C[i] = (int8_t)((uint8_t)A[i] - (uint8_t)B[i]);
  }
}
```

The element types are `float32`, `float64` (`f32`, `f64`) and the signed and unsigned fixed-width
integers from 8 to 64 bits (`i8`…`i64`, `u8`…`u64`); `float16` is omitted because C has no
standard type for it. `data` points at the caller's contiguous row-major array; `ndim` and
`shape` are its dimensions, owned by the caller. The kernel name selects the element type, so
the kernel casts `data` back to it. The kernel computes the element count from `c->shape` and
`c->ndim` — a rank-0 tensor has one element, a zero-sized dimension none — and `a`, `b` and `c`
must describe the same, already-broadcast shape: broadcasting is the caller's business.
Integer kernels wrap modulo `2 ^ bits` — a signed result is computed in the unsigned type of
the same width and converted back — matching `Sonnx.uintWrap` and `Sonnx.sintWrap`; the banner
of the generated file states the whole contract.

`operatorsFileFor ops dtypes` takes the operator table and the element types, so a caller can
render a subset; `operatorsFile` is the library's own table. The split forms render the same
kernels as a header and a source file: `headerFileFor` writes `sonnx_kernels.h` (the standard
includes, the banner, the `sonnx_tensor` definition and the 30 prototypes, no bodies) and
`sourceFileFor` writes `sonnx_kernels.c` (the banner, the quoted `#include "sonnx_kernels.h"`
and the 30 definitions, no repeated standard includes or struct). `headerName` is that quoted name, and `renderIncludes` leaves an
entry that is already delimited (`"…"` or `<…>`) unchanged, so the source's include is
reproduced as written. `binOpName` renders an operator as the string a reader sees (`add`,
`sub`, `mul`) rather than as a Lean constructor.

The pipeline **derives** the split pair from the formal specification, not from a descriptor.
`CGen.Derive.deriveElementwise` reads the declarations named by the family function requirements
in `FormalSpec.trace.json` (the `@declaration` of each `E_<OP>_<FAMILY>_FUNC_<nnnn>` entry) and
runs a **frontier search** over the spec-local step definitions: a node that exposes
`HAdd.hAdd`/`HSub.hSub`/`HMul.hMul`/`HDiv.hDiv`/`HMod.hMod` is a witness and is not descended;
otherwise its constants under `Generated.<Op>.` are enqueued. It intersects the per-family
operator sets and records each witness path. The search is deliberately not a transitive
closure: descending past `floatMulStep` would reach the rounding helpers and pick up `HSub`/`HDiv`
as well.

The C stage copies the formal specification, prepends `import CGen.Derive`, appends a `run_cmd`
that prints the derivation and writes `headerFileForDerived`/`sourceFileForDerived`, and compiles
the copy with `lake env lean`. The header and source carry a comment per kernel naming the
declaration, the witness path, the `@trace` key and the informal span; Python writes
`cgen-trace.json` with the hashes (including `CGen/Derive.lean`), the declarations and the
kernel-to-family map.

Only `add`, `sub` and `mul` are rendered: `CGen`'s signed-integer element step computes in the
unsigned type, which is the specified wrap for those three but not signed division. **Div**
derives `.div` correctly and is recorded `unsupported` with its trace, without C files; any
undecided family (missing declaration, a theorem with no value, or a frontier without a single
operation) is `unsupported`, not a failure of the run. The library's own `operatorsFile` table
and the `Elementwise` two-field constructor are unchanged.

`CGen` is not covered by the library digest (which hashes `lakefile.lean` and the `Sonnx` tree
only). A change to a `CGen` source file therefore does not invalidate a generated Lean artifact;
it changes only the C rendered from it.

## Generated artifacts

Generated artifacts live in the operator's own folder under `ops/`, not here:

```
ops/<op>/lean/FormalSpec.lean        the generated formal specification
ops/<op>/lean/FormalSpec.trace.json  its trace coverage and source/library digests
ops/<op>/lean/LeanConformance.lean   the generated executable runner
ops/<op>/lean/conformance.manifest.json
ops/<op>/c/CGenEmit.lean             the specification with the generated render trailer
ops/<op>/c/sonnx_kernels.h           the rendered prototypes
ops/<op>/c/sonnx_kernels.c           the rendered definitions
ops/<op>/c/c_tests.c                 the accepted C test program
ops/<op>/c/c-report.json             its checks, hashes and failing lines
ops/<op>/c/cgen-trace.json           the derivation: declarations, witnesses, spans, hashes
```

Re-deriving an operator model-free rewrites the header and source with the new derivation
comments. A `c-report.json` written before that still names the earlier header/source hashes;
the next C tester run refreshes it.

A run with an explicit `--run-dir` keeps them there until `--publish` validates the Lean
artifact and conformance and copies the set into `ops/<op>/lean/`. The pipeline invokes:

```sh
cd lean
lake env lean /absolute/path/FormalSpec.lean
lake env lean --run /absolute/path/LeanConformance.lean
```

## Lineage

The library is the generic layer of the hand-written Mul formalization (`lean/MulOp/` at git
HEAD), which was deleted on 2026-11-15 along with the hand-written executable runners
`Conformance.lean` and `AbsConformance.lean`; the deleted modules were re-namespaced into
`Sonnx` and the floating-point model was taken from the archived Sub formalization. The
operator-specific parts of `MulOp` — the Mul relation, its examples and its `Mul` instances on
the integer types — were left behind. Every operator, Mul included, is now formalized and
executed only by the generated pipeline, against this library.
