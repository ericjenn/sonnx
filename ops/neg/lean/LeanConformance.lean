import Sonnx

namespace NegRunner

def digitVal (base : Nat) (c : Char) : Option Nat :=
  let n := c.toNat
  if n >= 48 && n <= 57 then
    let d := n - 48
    if d < base then some d else none
  else if n >= 65 && n <= 70 then
    let d := n - 55
    if d < base then some d else none
  else if n >= 97 && n <= 102 then
    let d := n - 87
    if d < base then some d else none
  else none

def parseNatBase (base : Nat) (digits : List Char) (acc : Nat) : Option Nat :=
  match digits with
  | [] => some acc
  | c :: rest =>
      match digitVal base c with
      | some d => parseNatBase base rest (acc * base + d)
      | none => none

def parseNat (s : String) : Option Nat :=
  match s.toList with
  | '0' :: 'x' :: rest =>
      match rest with
      | [] => none
      | _ => parseNatBase 16 rest 0
  | '0' :: 'X' :: rest =>
      match rest with
      | [] => none
      | _ => parseNatBase 16 rest 0
  | [] => none
  | cs => parseNatBase 10 cs 0

def parseInt (s : String) : Option Int :=
  match s.toList with
  | '-' :: rest =>
      match parseNat (String.ofList rest) with
      | some n => some (-(Int.ofNat n))
      | none => none
  | '+' :: rest =>
      match parseNat (String.ofList rest) with
      | some n => some (Int.ofNat n)
      | none => none
  | _ =>
      match parseNat s with
      | some n => some (Int.ofNat n)
      | none => none

def intToString (i : Int) : String :=
  if i < 0 then "-" ++ toString (Int.natAbs i)
  else toString (Int.natAbs i)

def joinWith (sep : String) : List String → String
  | [] => ""
  | x :: xs =>
      match xs with
      | [] => x
      | _ => x ++ sep ++ joinWith sep xs

def prodShape : List Nat → Nat
  | [] => 1
  | d :: ds => d * prodShape ds

def formatShape : List Nat → String
  | [] => "-"
  | xs => joinWith "," (xs.map (fun n => toString n))

def formatData : List String → String
  | [] => "-"
  | xs => joinWith "," xs

def stripCR (s : String) : String :=
  match s.toList.reverse with
  | '\r' :: rest => String.ofList rest.reverse
  | _ => s

def splitLines (s : String) : List String :=
  if s == "" then []
  else
    let parts := s.splitOn "\n"
    let rev := parts.reverse
    let parts :=
      match rev with
      | [] => []
      | last :: rest => if last == "" then rest.reverse else parts
    parts.map stripCR

def parseNatList : List String → Option (List Nat)
  | [] => some []
  | s :: rest =>
      match parseNat s with
      | some n =>
          match parseNatList rest with
          | some ns => some (n :: ns)
          | none => none
      | none => none

def parseShape (s : String) : Option (List Nat) :=
  if s == "-" || s == "" then some []
  else parseNatList (s.splitOn ",")

def hasEmpty : List String → Bool
  | [] => false
  | x :: rest => x == "" || hasEmpty rest

def parseData (s : String) : Option (List String) :=
  if s == "-" || s == "" then some []
  else
    let parts := s.splitOn ","
    if hasEmpty parts then none else some parts

def parseTensorField (s : String) : Option (String × List Nat × List String) :=
  match s.splitOn "|" with
  | kind :: shapeStr :: dataStr :: [] =>
      match parseShape shapeStr with
      | some shape =>
          match parseData dataStr with
          | some data => some (kind, shape, data)
          | none => none
      | none => none
  | _ => none

def parseBitsList : List String → Except String (List Nat)
  | [] => .ok []
  | s :: rest =>
      match parseNat s with
      | some n =>
          match parseBitsList rest with
          | .ok ns => .ok (n :: ns)
          | .error e => .error e
      | none => .error ("invalid bit pattern: " ++ s)

def parseRat (s : String) : Except String (Int × Nat) :=
  match s.splitOn "/" with
  | n :: d :: [] =>
      match (parseInt n, parseNat d) with
      | (some num, some den) =>
          if den = 0 then .error ("zero denominator: " ++ s)
          else .ok (num, den)
      | _ => .error ("invalid rational: " ++ s)
  | _ => .error ("invalid rational: " ++ s)

def parseRatList : List String → Except String (List (Int × Nat))
  | [] => .ok []
  | s :: rest =>
      match parseRat s with
      | .ok r =>
          match parseRatList rest with
          | .ok rs => .ok (r :: rs)
          | .error e => .error e
      | .error e => .error e

def mapBits (f : Nat → Nat) (data : List String) : Except String (List String) :=
  match parseBitsList data with
  | .ok ns => .ok (ns.map (fun n => toString (f n)))
  | .error e => .error e

def mapRats (data : List String) : Except String (List String) :=
  match parseRatList data with
  | .ok rs =>
      .ok (rs.map (fun r =>
        match r with
        | (num, den) =>
            let negNum := -num
            let nStr := intToString negNum
            let dStr := toString den
            nStr ++ "/" ++ dStr))
  | .error e => .error e

def validateShapeData (shape : List Nat) (data : List String) : Except String Unit :=
  if data.length = prodShape shape then .ok () else .error "data length mismatch"

def flipSignBit (width bits : Nat) : Nat :=
  let mask := 2 ^ width
  let b := bits % mask
  let signBit := 2 ^ (width - 1)
  if b >= signBit then b - signBit else b + signBit

def negMod (width bits : Nat) : Nat :=
  let mask := 2 ^ width
  let b := bits % mask
  (mask - b) % mask

def isSupportedKind (kind : String) : Bool :=
  kind == "f64" || kind == "f32" ||
  kind == "i8" || kind == "i16" || kind == "i32" || kind == "i64" ||
  kind == "rat"

def negSupported (kind : String) (shape : List Nat) (data : List String) : Except String (List String) :=
  if kind == "f64" then mapBits (flipSignBit 64) data
  else if kind == "f32" then mapBits (flipSignBit 32) data
  else if kind == "i8" then mapBits (negMod 8) data
  else if kind == "i16" then mapBits (negMod 16) data
  else if kind == "i32" then mapBits (negMod 32) data
  else if kind == "i64" then mapBits (negMod 64) data
  else if kind == "rat" then mapRats data
  else .error ("unsupported kind " ++ kind)

def malformed (id : String) (msg : String) : String :=
  id ++ "\tMALFORMED\t0\t" ++ msg

def err (id : String) (msg : String) : String :=
  id ++ "\tERR\t0\t" ++ msg

def unsupported (id : String) (msg : String) : String :=
  id ++ "\tUNSUPPORTED\t0\t" ++ msg

def okLine (id kind shape data : String) : String :=
  id ++ "\tOK\t1\t" ++ kind ++ "|" ++ shape ++ "|" ++ data

def processLine (line : String) : String :=
  let fields := line.splitOn "\t"
  match fields with
  | id :: attrs :: arityStr :: rest =>
      match parseNat arityStr with
      | none => malformed id "invalid arity"
      | some arity =>
          if arity != 1 then
            err id "unsupported arity"
          else if (attrs != "-") && (attrs != "") then
            err id "unexpected attributes"
          else
            match rest with
            | tensorStr :: [] =>
                match parseTensorField tensorStr with
                | none => malformed id "invalid tensor field"
                | some (kind, shape, data) =>
                    match validateShapeData shape data with
                    | .error e => malformed id e
                    | .ok () =>
                        if isSupportedKind kind then
                          match negSupported kind shape data with
                          | .ok newData =>
                              okLine id kind (formatShape shape) (formatData newData)
                          | .error e => malformed id e
                        else
                          unsupported id ("unsupported kind " ++ kind)
            | _ => malformed id "arity/tensor field mismatch"
  | id :: _ => malformed id "missing fields"
  | [] => malformed "" "empty line"

end NegRunner

open NegRunner

def main : IO Unit := do
  let stdin ← IO.getStdin
  let input ← stdin.readToEnd
  let lines := splitLines input
  for line in lines do
    IO.println (processLine line)
