/-
The SONNX C code generator — `CGen`.

This library turns an operator description into C11 source.  It holds

* `CGen.Ast`   — the C11 fragment operator kernels need: types, expressions
                 and operators, statements, functions, and a translation unit,
* `CGen.Print` — the precedence-aware pretty-printer for that fragment,
* `CGen.Kernel`— the elementwise operator description and the per-element-type
                 kernel builders, plus `operatorsFile`, the generated unit,
* `CGen.Derive`— the metaprogram that derives the element operation from a formal
                 specification's family declarations, and the derivation trailer's
                 entry point.  It is built by `lake build CGen` but not imported here,
                 so a consumer of the library does not pull in the Lean metaprogramming
                 modules.

The generated kernels take `sonnx_tensor` values — a data pointer, a rank and
a pointer to the dimensions — and cover operands of one, already-broadcast
shape; broadcasting is the caller's responsibility.  Integer kernels wrap
modulo `2 ^ bits`, matching `Sonnx.uintWrap` and `Sonnx.sintWrap`; `float16` is
not generated because C has no standard type for it.

`CGen` is independent of `Sonnx`: it renders code, it does not state
properties.  Two drivers use it; both are built by `lake build CGen` but are not
imported here, because each defines `main`:

    cd lean
    lake env lean --run CGen/Emit.lean /tmp/operators.c
    lake env lean --run CGen/Probe.lean
-/

import CGen.Ast
import CGen.Print
import CGen.Kernel
