/-
The C derivation — probes.

Each case names a family declaration whose element step is reached through a
spec-local wrapper, derives the operation, and checks the operation and the
witness path.  The exit status is nonzero when a case fails.

    cd lean
    lake env lean CGen/DeriveProbe.lean
-/

import CGen.Derive

namespace DeriveProbe

def stepSub (a b : Int) : Int := a - b
def relSub (a b c : Int) : Prop := c = stepSub a b

def stepMul (a b : Int) : Int := a * b
def relMul (a b c : Int) : Prop := c = stepMul a b

def stepDiv (a b : Int) : Int := a / b
def relDiv (a b c : Int) : Prop := c = stepDiv a b

end DeriveProbe

open Lean Elab Command

/-- Derive one family and fail the command when the operation or the witness
is not the expected one. -/
def probe (name : String) (declaration : String) (expected : String)
    (witness : String) : CommandElabM Unit := do
  match ← CGen.deriveElementwise name
      [⟨.float, declaration, "", "E_PROBE_FLOAT_FUNC_0010", "ops/probe/probe.md:L1-L2", []⟩] with
  | .error reason => throwError "derivation failed: {reason}"
  | .ok derived =>
      let op := CGen.binOpName derived.op.op
      unless op == expected do
        throwError "expected {expected}, derived {op}"
      let source := derived.derivations.head!
      unless source.witness == witness do
        throwError "expected witness {witness}, got {source.witness}"
      IO.println s!"ok {name}: {op} from {source.witness}"

run_cmd do
  probe "sub" "DeriveProbe.relSub" "sub" "DeriveProbe.relSub -> DeriveProbe.stepSub"
  probe "mul" "DeriveProbe.relMul" "mul" "DeriveProbe.relMul -> DeriveProbe.stepMul"
  probe "div" "DeriveProbe.relDiv" "div" "DeriveProbe.relDiv -> DeriveProbe.stepDiv"
  IO.println "derivation probes hold"
