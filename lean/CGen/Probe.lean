/-
The C code generator — printer probes.

Each case renders an expression, a declarator or a whole kernel and compares
the text with a hand-checked expectation.  The cases pin the parenthesization
table of `CGen.Print`: associativity, the prefix and cast operand contexts,
negative literals, postfix forms, and the element expressions of the kernel
builders.  The exit code is 0 only when every case matches.

    cd lean
    lake env lean --run CGen/Probe.lean
-/

import CGen.Kernel
import CGen.Print

open CGen

/-- Is `needle` a substring of `hay`?  `String.contains` takes a character. -/
def hasSub (hay needle : String) : Bool := (hay.splitOn needle).length > 1

def subOp : Elementwise := ⟨"sub", .sub⟩
def headerText : String := renderFile (headerFileFor [subOp] [.float 32])
def sourceText : String := renderFile (sourceFileFor [subOp] [.float 32])

/-- `(label, rendered, expected)`. -/
def probeCases : List (String × String × String) :=
  [("sub is left-associative",
    renderExpr 0 (.binary .sub (.binary .sub (.var "a") (.var "b")) (.var "c")),
    "a - b - c"),   ("sub nests on the right",
    renderExpr 0 (.binary .sub (.var "a") (.binary .sub (.var "b") (.var "c"))),
    "a - (b - c)"),
   ("assignment is right-associative",
    renderExpr 0 (.binary .assign (.var "a") (.binary .assign (.var "b") (.var "c"))),
    "a = b = c"),
   ("double negation stays parenthesized",
    renderExpr 0 (.unary .neg (.unary .neg (.var "x"))), "-(-x)"),
   ("double address-of stays parenthesized",
    renderExpr 0 (.unary .addr (.unary .addr (.var "x"))), "&(&x)"),
   ("a cast binds its unary operand",
    renderExpr 0 (.cast (.int 32 true) (.unary .neg (.var "x"))), "(int32_t)(-x)"),
   ("a comma argument is parenthesized",
    renderExpr 0 (.call "f" [.binary .comma (.var "a") (.var "b")]), "f((a, b))"),
   ("a negative literal is parenthesized",
    renderExpr 0 (.binary .add (.intLit (-1)) (.var "x")), "(-1) + x"),
   ("an index offset needs no parentheses",
    renderExpr 0 (.index (.var "a") (.binary .add (.var "i") (.intLit 1))), "a[i + 1]"),
   ("an arrow field binds tightly",
    renderExpr 0 (.field (.var "p") "x" true), "p->x"),
   ("a dereference as a field base is parenthesized",
    renderExpr 0 (.field (.unary .deref (.var "p")) "x" false), "(*p).x"),
   ("sizeof takes a parenthesized type",
    renderExpr 0 (.sizeofType .size), "sizeof(size_t)"),
   ("a ternary renders in all three positions",
    renderExpr 0 (.ternary (.var "c") (.binary .add (.var "a") (.var "b")) (.var "d")),
    "c ? a + b : d"),
   ("a pointer declarator",
    declType (.ptr (.const (.float 32))) "a", "const float *a"),
   ("a signed kernel wraps in the unsigned type",
    renderExpr 0 (rhs .sub (.int 8 true)), "(int8_t)((uint8_t)a[i] - (uint8_t)b[i])"),
   ("an unsigned kernel converts the result",
    renderExpr 0 (rhs .mul (.int 16 false)), "(uint16_t)(a[i] * b[i])"),
   ("a float kernel applies the operator directly",
    renderExpr 0 (rhs .add (.float 64)), "a[i] + b[i]"),
   ("the tensor struct",
    renderDecl tensorStruct,
    "typedef struct {\n  void *data;\n  size_t ndim;\n  const size_t *shape;\n} " ++
      "sonnx_tensor;"),
   ("a whole kernel",
    renderFun (kernel ⟨"add", .add⟩ (.float 32)),
    "void sonnx_add_f32(const sonnx_tensor *a, const sonnx_tensor *b, " ++
      "sonnx_tensor *c) {\n" ++
      "  const float *A = (const float *)a->data;\n" ++
      "  const float *B = (const float *)b->data;\n" ++
      "  float *C = (float *)c->data;\n" ++
      "  size_t n = 1;\n" ++
      "  for (size_t d = 0; d < c->ndim; d++) {\n" ++
      "    n = n * c->shape[d];\n" ++
      "  }\n" ++
      "  for (size_t i = 0; i < n; i++) {\n    C[i] = A[i] + B[i];\n  }\n}"),
   ("a quoted include stays local",
    renderIncludes ["\"sonnx_kernels.h\""], "#include \"sonnx_kernels.h\"\n"),
   ("a bare include is a system header",
    renderIncludes ["stddef.h"], "#include <stddef.h>\n"),
   ("binOpName names the element operation", binOpName .sub, "sub"),
   ("the header declares the kernel",
    toString (hasSub headerText
      ("void sonnx_sub_f32(const sonnx_tensor *a, const sonnx_tensor *b, " ++
        "sonnx_tensor *c);")), "true"),
   ("the header defines the tensor struct",
    toString (hasSub headerText "} sonnx_tensor;"), "true"),
   ("the header defines no kernel",
    toString (!hasSub headerText "C[i] ="), "true"),
   ("the source includes its header",
    toString (hasSub sourceText "#include \"sonnx_kernels.h\""), "true"),
   ("the source does not redefine the tensor struct",
    toString (!hasSub sourceText "} sonnx_tensor;"), "true"),
   ("the source defines the kernel",
    toString (hasSub sourceText
      ("void sonnx_sub_f32(const sonnx_tensor *a, const sonnx_tensor *b, " ++
        "sonnx_tensor *c) {")), "true"),
   ("the source computes the element count from the output shape",
    toString (hasSub sourceText "n = n * c->shape[d];"), "true"),
   ("the source carries no prototype",
    toString (!hasSub sourceText
      ("void sonnx_sub_f32(const sonnx_tensor *a, const sonnx_tensor *b, " ++
        "sonnx_tensor *c);")), "true")]

/-- Run every probe; print the failures and return 1 if any case differs. -/
def main : IO UInt32 := do
  let mut failures := 0
  for (label, got, want) in probeCases do
    if got != want then
      failures := failures + 1
      IO.eprintln s!"FAILED {label}: got {got}, want {want}"
  if failures == 0 then
    IO.println s!"probe ok: {probeCases.length} cases"
    return 0
  else
    IO.eprintln s!"{failures} probe failure(s)"
    return 1
