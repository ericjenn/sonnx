/-
The C code generator — derive the element operation from a formal specification.

The C stage does not ask the formalizer for a descriptor.  It reads the
declarations named by the `@trace` entries of the family function requirements
(`E_<OP>_<FAMILY>_FUNC_<nnnn>`) and derives the element operation from their
bodies.  Most are `Prop`-valued definitions that call a spec-local step
definition, so the search walks the spec-local constants until a node exposes
an arithmetic operator; it stops there and does not descend into the rounding
helpers below it.

The derivation is a frontier search, not a transitive closure: a transitive
closure of `FloatMulRel` would reach `RoundMagNearest`/`RepMag` and pick up
`HSub`/`HDiv` as well as the `HMul` the family means.  The witness path and the
per-family operator sets are reported so the choice is auditable.

Lean only derives and reports.  The pipeline owns the JSON manifest and the
content hashes; this module prints a structured protocol line and, when the
operation is one CGen has kernels for, renders the C files.
-/

import Lean.Elab.Command
import Lean.Util.FoldConsts
import CGen.Kernel

namespace CGen

open Lean Elab Command

/-- The arithmetic constants the derivation recognises. -/
def arithmeticOps : List Name :=
  [``HAdd.hAdd, ``HSub.hSub, ``HMul.hMul, ``HDiv.hDiv, ``HMod.hMod]

/-- The `CGen.BinOp` an arithmetic constant denotes. -/
def binOpOfName : Name → Option BinOp
  | ``HAdd.hAdd => some .add
  | ``HSub.hSub => some .sub
  | ``HMul.hMul => some .mul
  | ``HDiv.hDiv => some .div
  | ``HMod.hMod => some .mod
  | _ => none

/-- The arithmetic operations used in an expression. -/
def opsIn (e : Lean.Expr) : List Name :=
  let used := e.getUsedConstants.toList
  arithmeticOps.filter (fun op => used.elem op)

/-- A declaration's value, when it has one.  The theorems and opaque
definitions the trace may name have none and are reported as undecided. -/
def declarationValue (env : Environment) (name : Name) : Option Lean.Expr :=
  match env.find? name with
  | some info => info.value?
  | none => none

/-- One node of the frontier search: the declaration, its distance from the
root and the path that reached it. -/
structure Node where
  name : Name
  depth : Nat
  path : List Name
  deriving Inhabited

/-- The frontier search.  A node that exposes an arithmetic operation is a
witness and is not descended; otherwise its spec-local constants are
enqueued.  `visited` keeps the search finite. -/
partial def frontierGo (env : Environment) (ns : Name) (maxDepth : Nat)
    (queue : List Node) (visited : List Name)
    (found : List (List Name × List Name)) : List (List Name × List Name) :=
  match queue with
  | [] => found
  | node :: rest =>
      if visited.elem node.name then
        frontierGo env ns maxDepth rest visited found
      else
        let visited := node.name :: visited
        match declarationValue env node.name with
        | none => frontierGo env ns maxDepth rest visited found
        | some value =>
            let ops := opsIn value
            if !ops.isEmpty then
              frontierGo env ns maxDepth rest visited (found ++ [(ops, node.path)])
            else if node.depth >= maxDepth then
              frontierGo env ns maxDepth rest visited found
            else
              let next := value.getUsedConstants.toList.filter
                (fun n => ns.isPrefixOf n && !visited.elem n)
              let queue := rest ++ next.map (fun n =>
                { name := n, depth := node.depth + 1, path := node.path ++ [n] })
              frontierGo env ns maxDepth queue visited found

/-- The witnesses of one family declaration: the arithmetic operations and
the path to the step that exposed them. -/
def frontier (env : Environment) (root : Name) : List (List Name × List Name) :=
  frontierGo env root.getPrefix 6 [{ name := root, depth := 0, path := [root] }] [] []

/-- Derive the element operation of `name` from the family declarations in
`inputs`.  Each input names the family, its trace declaration, the
requirement key and the informal span; `witness` and `ops` are filled here.
The result is the derived operator and the per-family evidence, or the reason
the families do not decide a single operation. -/
def deriveElementwise (name : String) (inputs : List Derivation) :
    CommandElabM (Except String Derived) := do
  let env ← getEnv
  let mut results : List (Derivation × List Name) := []
  for d in inputs do
    let decl := d.declaration.toName
    match env.find? decl with
    | none => return .error s!"{d.declaration}: no such declaration"
    | some info =>
        match info.value? with
        | none => return .error s!"{d.declaration}: no value to inspect"
        | some _ =>
            let witnesses := frontier env decl
            if witnesses.isEmpty then
              return .error s!"{d.declaration}: no arithmetic operation in the frontier"
            let ops := witnesses.foldl (fun acc (found, _) => acc ++ found) []
            let path := match witnesses with
              | (_, path) :: _ => path
              | [] => []
            let witness := " -> ".intercalate (path.map (fun n => n.toString))
            results := results ++
              [({ d with witness := witness, ops := (ops.eraseDups).map (fun n => n.toString) },
                ops.eraseDups)]
  let sets : List (List Name) := results.map (fun (_, ops) => ops)
  let inter : List Name := match sets with
    | [] => []
    | first :: rest =>
        rest.foldl (fun (acc : List Name) (s : List Name) =>
          acc.filter (fun n => s.elem n)) first
  let inter : List Name :=
    if inter.length > 1 then inter.filter (fun n => n != ``HMod.hMod) else inter
  if inter.length != 1 then
    let detail := results.map (fun (d, ops) =>
      familyName d.family ++ ": " ++ ", ".intercalate (ops.map (fun n => n.toString)))
    let joined := "; ".intercalate detail
    return .error s!"no single element operation: {joined}"
  let opName := inter.head!
  let some op := binOpOfName opName
    | return .error s!"unknown element operation {opName}"
  let derivations := results.map (fun (d, _) => d)
  return .ok { op := { name := name, op := op }, derivations := derivations }

end CGen
