import Lake
open Lake DSL

package «sonnx_lean»

/-- The shared SONNX library: the tensor model, broadcasting, indices, the
integer and floating-point families, and elementwise binary operations.  Every
generated operator module imports it as `Sonnx`. -/
@[default_target]
lean_lib «Sonnx»

/-- The C generator: a C11 AST, a precedence-aware printer, and the elementwise
kernel builders.  Independent of `Sonnx`.  The explicit globs also build the
two drivers (`CGen.Emit`, `CGen.Probe`), which the root module does not import
because both define `main`. -/
lean_lib «CGen» where
  globs := #[Glob.andSubmodules `CGen]
