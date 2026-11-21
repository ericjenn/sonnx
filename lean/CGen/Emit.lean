/-
The C code generator — the emit driver.

Renders `CGen.operatorsFile` and writes it to the path given as the first
command-line argument, or to standard output when no path is given:

    cd lean
    lake env lean --run CGen/Emit.lean /tmp/operators.c

The parent directory of the path must exist: `IO.FS.writeFile` creates the
file but not directories.
-/

import CGen.Kernel
import CGen.Print

open CGen

/-- Write the generated translation unit to the first argument, or print it. -/
def main (args : List String) : IO UInt32 := do
  let text := renderFile operatorsFile
  match args with
  | [] =>
      IO.print text
      return 0
  | path :: _ =>
      IO.FS.writeFile (System.FilePath.mk path) text
      return 0
