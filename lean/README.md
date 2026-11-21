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

The pipeline's `lean.build_library()` runs `lake build Sonnx CGen` together, because a generated
module may import `CGen` for its descriptor, and an olean that does not exist fails the compile
that carries it.

## The import contract

A generated module starts with `import Sonnx` (or the one library module it needs,
`Sonnx.Float`, say) and may otherwise import only Lean core (`Init`, `Std`, `Lean`) and, for
the C descriptor, `CGen`. It never
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
prototype and a kernel of the form

```c
void sonnx_sub_i8(const int8_t *a, const int8_t *b, int8_t *c, size_t n) {
  for (size_t i = 0; i < n; i++) {
    c[i] = (int8_t)((uint8_t)a[i] - (uint8_t)b[i]);
  }
}
```

The element types are `float32`, `float64` (`f32`, `f64`) and the signed and unsigned fixed-width
integers from 8 to 64 bits (`i8`…`i64`, `u8`…`u64`); `float16` is omitted because C has no
standard type for it. The kernels assume contiguous row-major arrays of one length `n`: the
broadcast shape and the strides are the caller's business. Integer kernels wrap modulo
`2 ^ bits` — a signed result is computed in the unsigned type of the same width and converted
back — matching `Sonnx.uintWrap` and `Sonnx.sintWrap`; the banner of the generated file states
the whole contract.

`operatorsFileFor ops dtypes` takes the operator table and the element types, so a caller can
render a subset; `operatorsFile` is the library's own table. The split forms render the same
kernels as a header and a source file: `headerFileFor` writes `sonnx_kernels.h` (the standard
includes, the banner and the 30 prototypes, no bodies) and `sourceFileFor` writes
`sonnx_kernels.c` (the banner, the quoted `#include "sonnx_kernels.h"` and the 30 definitions,
no repeated standard includes). `headerName` is that quoted name, and `renderIncludes` leaves an
entry that is already delimited (`"…"` or `<…>`) unchanged, so the source's include is
reproduced as written. `binOpName` renders an operator as the string a reader sees (`add`,
`sub`, `mul`) rather than as a Lean constructor.

The pipeline renders the split pair from a **descriptor** the formalizer writes into the
generated module:

```lean
def sonnxCGenOp : CGen.Elementwise := ⟨"sub", .sub⟩
```

It is a top-level `def` outside every namespace, named exactly `sonnxCGenOp`, whose `name` is the
operator's lowercase entry and whose second field is the matching `CGen.BinOp`. The C stage
copies the formal specification, appends an `#eval` trailer that prints the descriptor and writes
`headerFileFor [sonnxCGenOp] dtypes` and `sourceFileFor [sonnxCGenOp] dtypes` where Lean can see
them, and compiles the copy with `lake env lean`. Only `add`, `sub` and `mul` are rendered:
`CGen`'s signed-integer element step computes in the unsigned type, which is the specified wrap
for those three but not signed division, so `Div` and `Mod` descriptors are refused. A module
without the descriptor — any operator that is not elementwise binary — is reported `unsupported`
by the C stage, which is not a failure of the run.

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
```

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
