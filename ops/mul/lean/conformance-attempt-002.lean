import Sonnx

/-!
# Executable conformance runner for ONNX `Mul` (opset 14)

Reads one tab-separated request per line from stdin and writes exactly one
tab-separated response per line on stdout.  Floats and integers travel as raw
bit patterns.  `float16` cannot be represented by Lean core, so any request
involving it is answered with `UNSUPPORTED`.
-/

namespace MulRunner

/-! ## Generic list and string helpers -/

def joinWith (sep : String) : List String → String
  | [] => ""
  | [x] => x
  | x :: xs => x ++ sep ++ joinWith sep xs

def listGet? {α : Type} : List α → Nat → Option α
  | [], _ => none
  | x :: _, 0 => some x
  | _ :: xs, n + 1 => listGet? xs n

def listGetD {α : Type} (l : List α) (i : Nat) (d : α) : α :=
  match listGet? l i with
  | some x => x
  | none => d

def trimCR (s : String) : String :=
  match s.toList.reverse with
  | '\r' :: rest => String.ofList rest.reverse
  | _ => s

def endsWithNewline (s : String) : Bool :=
  match s.toList.reverse with
  | '\n' :: _ => true
  | _ => false

def dropLastList {α : Type} : List α → List α
  | [] => []
  | [_] => []
  | x :: xs => x :: dropLastList xs

def shapeToString (s : List Nat) : String :=
  match s with
  | [] => "-"
  | _ => joinWith "," (s.map (fun n => toString n))

def dataToStringNat (l : List Nat) : String :=
  match l with
  | [] => "-"
  | _ => joinWith "," (l.map (fun n => toString n))

def ratToString (n : Int) (d : Nat) : String :=
  if n = 0 then "0/1"
  else
    let g0 := Nat.gcd n.natAbs d
    let g := if g0 = 0 then 1 else g0
    toString (n / Int.ofNat g) ++ "/" ++ toString (d / g)

def dataToStringRat (l : List (Int × Nat)) : String :=
  match l with
  | [] => "-"
  | _ => joinWith "," (l.map (fun p => ratToString p.1 p.2))

/-! ## Field parsing -/

def parseNatList (s : String) : Option (List Nat) :=
  if s == "" || s == "-" then some []
  else
    let rec go : List String → Option (List Nat)
      | [] => some []
      | p :: ps =>
          match String.toNat? p with
          | none => none
          | some n =>
              match go ps with
              | none => none
              | some rest => some (n :: rest)
    go (s.splitOn ",")

def parseSignedInt (s : String) : Option Int :=
  match s.toList with
  | '-' :: rest =>
      match String.toNat? (String.ofList rest) with
      | some n => some (-(Int.ofNat n))
      | none => none
  | _ =>
      match String.toNat? s with
      | some n => some (Int.ofNat n)
      | none => none

def parseRatVal (s : String) : Option (Int × Nat) :=
  match s.splitOn "/" with
  | [num, den] =>
      match parseSignedInt num, String.toNat? den with
      | some n, some d => if d = 0 then none else some (n, d)
      | _, _ => none
  | _ => none

/-- Read an integer token as a raw bit pattern of width `bits`. -/
def parseBits (s : String) (bits : Nat) : Option Nat :=
  match parseSignedInt s with
  | none => none
  | some n =>
      let m := 2 ^ bits
      if n ≥ 0 then
        let v := n.natAbs
        if v < m then some v else some (v % m)
      else
        let v := n.natAbs % m
        some ((m - v) % m)

structure TensorField where
  kind : String
  shape : List Nat
  dataTokens : List String

def parseTensorField (s : String) : Option TensorField :=
  match s.splitOn "|" with
  | [k, sh, dat] =>
      match parseNatList sh with
      | none => none
      | some shape =>
          let tokens := if dat == "-" || dat == "" then [] else dat.splitOn ","
          some { kind := k, shape := shape, dataTokens := tokens }
  | _ => none

/-! ## Shapes, broadcasting, index mapping -/

def shapeSize (s : List Nat) : Nat :=
  s.foldl (fun acc d => acc * d) 1

def strides : List Nat → List Nat
  | [] => []
  | _ :: ds => shapeSize ds :: strides ds

def getDim (s : List Nat) (rank i : Nat) : Nat :=
  let offset := rank - s.length
  if i < offset then 1
  else
    match listGet? s (i - offset) with
    | some d => d
    | none => 1

def bcastDim (A B : List Nat) (rank i : Nat) : Option Nat :=
  let a := getDim A rank i
  let b := getDim B rank i
  if a = b then some a
  else if a = 1 then some b
  else if b = 1 then some a
  else none

/-- Multidirectional (Numpy-style) broadcast shape. -/
def bcastShape (A B : List Nat) : Option (List Nat) :=
  let rank := max A.length B.length
  (List.range rank).foldl
    (fun acc i =>
      match acc with
      | none => none
      | some dims =>
          match bcastDim A B rank i with
          | none => none
          | some d => some (dims ++ [d]))
    (some [])

/-- Flat index into the operand of shape `S` for the flat index `flat` of the
result of shape `R`.  Dimensions of `S` that are `1` (or missing) map to `0`. -/
def bcastFlatIndex (R S : List Nat) (flat : Nat) : Nat :=
  let rank := R.length
  let st := strides R
  (List.range rank).foldl
    (fun acc i =>
      let dimR := listGetD R i 1
      let stride := listGetD st i 1
      let coordR := (flat / stride) % dimR
      let dimS := getDim S rank i
      let coordS := if dimS = 1 then 0 else coordR
      acc * dimS + coordS)
    0

def buildData {α β γ : Type} (step : α → β → γ) (R S T : List Nat)
    (dataA : List α) (dataB : List β) : Option (List γ) :=
  let rec go : List Nat → Option (List γ)
    | [] => some []
    | j :: js =>
        match listGet? dataA (bcastFlatIndex R S j),
              listGet? dataB (bcastFlatIndex R T j) with
        | some a, some b =>
            match go js with
            | some rest => some (step a b :: rest)
            | none => none
        | _, _ => none
  go (List.range (shapeSize R))

def bcastBinary {α β γ : Type} (step : α → β → γ)
    (A : List Nat) (dataA : List α) (B : List Nat) (dataB : List β) :
    Option (List Nat × List γ) :=
  match bcastShape A B with
  | none => none
  | some R =>
      if dataA.length == shapeSize A && dataB.length == shapeSize B then
        match buildData step R A B dataA dataB with
        | some out => some (R, out)
        | none => none
      else none

/-! ## IEEE 754 values, decoding, rounding and encoding -/

inductive FKind where
  | nan
  | inf (neg : Bool)
  | zero (neg : Bool)
  | finite (neg : Bool) (mant : Nat) (exp : Int)

def fIsNaN : FKind → Bool
  | .nan => true
  | _ => false

def fIsInf : FKind → Bool
  | .inf _ => true
  | _ => false

def fIsZero : FKind → Bool
  | .zero _ => true
  | _ => false

def fNeg : FKind → Bool
  | .nan => false
  | .inf s => s
  | .zero s => s
  | .finite s _ _ => s

def fMant : FKind → Nat
  | .finite _ m _ => m
  | _ => 0

def fExp : FKind → Int
  | .finite _ _ e => e
  | _ => 0

def bxor (a b : Bool) : Bool := if a then !b else b

structure Fmt where
  fbits : Nat
  expBits : Nat
  bias : Nat
  emin : Int
  emax : Int

def fmt32 : Fmt :=
  { fbits := 23, expBits := 8, bias := 127, emin := -126, emax := 127 }

def fmt64 : Fmt :=
  { fbits := 52, expBits := 11, bias := 1023, emin := -1022, emax := 1023 }

def decodeF32 (bits : Nat) : FKind :=
  let sign := bits ≥ 2 ^ 31
  let expField := (bits / 2 ^ 23) % 2 ^ 8
  let frac := bits % 2 ^ 23
  if expField = 255 then
    (if frac = 0 then .inf sign else .nan)
  else if expField = 0 then
    (if frac = 0 then .zero sign else .finite sign frac (-149))
  else .finite sign (2 ^ 23 + frac) (Int.ofNat expField - 150)

def decodeF64 (bits : Nat) : FKind :=
  let sign := bits ≥ 2 ^ 63
  let expField := (bits / 2 ^ 52) % 2 ^ 11
  let frac := bits % 2 ^ 52
  if expField = 2047 then
    (if frac = 0 then .inf sign else .nan)
  else if expField = 0 then
    (if frac = 0 then .zero sign else .finite sign frac (-1074))
  else .finite sign (2 ^ 52 + frac) (Int.ofNat expField - 1075)

def encodeBits (fmt : Fmt) (sign : Bool) (expField : Nat) (frac : Nat) : Nat :=
  let signBit := if sign then 2 ^ (fmt.expBits + fmt.fbits) else 0
  signBit + expField * 2 ^ fmt.fbits + frac

def encodeNaN (fmt : Fmt) : Nat :=
  encodeBits fmt false ((2 ^ fmt.expBits) - 1) (2 ^ (fmt.fbits - 1))

def encodeInf (fmt : Fmt) (sign : Bool) : Nat :=
  encodeBits fmt sign ((2 ^ fmt.expBits) - 1) 0

def encodeZero (fmt : Fmt) (sign : Bool) : Nat :=
  encodeBits fmt sign 0 0

def bitLenAux : Nat → Nat → Nat
  | 0, _ => 0
  | _, 0 => 0
  | fuel + 1, n + 1 => 1 + bitLenAux fuel ((n + 1) / 2)

def bitLen (n : Nat) : Nat := bitLenAux 4096 n

/-- Round `sig * 2^shift` to an integer, ties to even. -/
def roundScaled (sig : Nat) (shift : Int) : Nat :=
  if shift ≥ 0 then
    sig * 2 ^ shift.natAbs
  else
    let d := shift.natAbs
    let den := 2 ^ d
    let q := sig / den
    let r := sig % den
    let half := 2 ^ (d - 1)
    if r > half || (r = half && q % 2 = 1) then q + 1 else q

/-- Round the exact magnitude `m * 2^e` to the format `fmt`, ties to even, with
the subnormal quantum of the format; overflow yields an infinity of the sign of
the exact product, underflow a signed zero. -/
def roundToFormat (fmt : Fmt) (sign : Bool) (m : Nat) (e : Int) : Nat :=
  if m = 0 then encodeZero fmt sign
  else
    let bl := bitLen m
    let E : Int := Int.ofNat (bl - 1) + e
    if E > fmt.emax then encodeInf fmt sign
    else
      let q : Int :=
        if E ≥ fmt.emin then E - Int.ofNat fmt.fbits
        else fmt.emin - Int.ofNat fmt.fbits
      let k0 := roundScaled m (e - q)
      let kk : Nat × Int :=
        if k0 = 2 ^ (fmt.fbits + 1) then (2 ^ fmt.fbits, q + 1) else (k0, q)
      let k := kk.1
      let q' := kk.2
      if k = 0 then encodeZero fmt sign
      else
        let E2 : Int := q' + Int.ofNat (bitLen k - 1)
        if E2 > fmt.emax then encodeInf fmt sign
        else if k < 2 ^ fmt.fbits then
          encodeBits fmt sign 0 k
        else
          encodeBits fmt sign (E2 + Int.ofNat fmt.bias).toNat (k - 2 ^ fmt.fbits)

/-- The IEEE 754 element step of `Mul` (result encoded as a bit pattern). -/
def mulFloat (fmt : Fmt) (a b : FKind) : Nat :=
  if fIsNaN a || fIsNaN b then encodeNaN fmt
  else if (fIsZero a && fIsInf b) || (fIsInf a && fIsZero b) then encodeNaN fmt
  else if fIsInf a || fIsInf b then encodeInf fmt (bxor (fNeg a) (fNeg b))
  else
    let sign := bxor (fNeg a) (fNeg b)
    if fIsZero a || fIsZero b then encodeZero fmt sign
    else roundToFormat fmt sign (fMant a * fMant b) (fExp a + fExp b)

/-! ## Data parsers -/

def parseF32Data (tokens : List String) : Option (List FKind) :=
  let rec go : List String → Option (List FKind)
    | [] => some []
    | t :: ts =>
        match String.toNat? t with
        | some bits =>
            if bits < 2 ^ 32 then
              match go ts with
              | some rest => some (decodeF32 bits :: rest)
              | none => none
            else none
        | none => none
  go tokens

def parseF64Data (tokens : List String) : Option (List FKind) :=
  let rec go : List String → Option (List FKind)
    | [] => some []
    | t :: ts =>
        match String.toNat? t with
        | some bits =>
            if bits < 2 ^ 64 then
              match go ts with
              | some rest => some (decodeF64 bits :: rest)
              | none => none
            else none
        | none => none
  go tokens

def parseIntData (bits : Nat) (tokens : List String) : Option (List Nat) :=
  let rec go : List String → Option (List Nat)
    | [] => some []
    | t :: ts =>
        match parseBits t bits with
        | some v =>
            match go ts with
            | some rest => some (v :: rest)
            | none => none
        | none => none
  go tokens

def parseRatData (tokens : List String) : Option (List (Int × Nat)) :=
  let rec go : List String → Option (List (Int × Nat))
    | [] => some []
    | t :: ts =>
        match parseRatVal t with
        | some v =>
            match go ts with
            | some rest => some (v :: rest)
            | none => none
        | none => none
  go tokens

/-! ## Kind classification and line processing -/

inductive KindTag where
  | f32
  | f64
  | f16
  | i (bits : Nat)
  | u (bits : Nat)
  | rat
  | unsupported

def parseKind (s : String) : KindTag :=
  if s == "f32" || s == "float32" then .f32
  else if s == "f64" || s == "float64" || s == "double" || s == "float" then .f64
  else if s == "f16" || s == "float16" || s == "half" then .f16
  else if s == "i8" || s == "int8" then .i 8
  else if s == "i16" || s == "int16" then .i 16
  else if s == "i32" || s == "int32" then .i 32
  else if s == "i64" || s == "int64" then .i 64
  else if s == "u8" || s == "uint8" then .u 8
  else if s == "u16" || s == "uint16" then .u 16
  else if s == "u32" || s == "uint32" then .u 32
  else if s == "u64" || s == "uint64" then .u 64
  else if s == "rat" then .rat
  else .unsupported

def mkOk (id kind : String) (shape : List Nat) (data : String) : String :=
  id ++ "\tOK\t1\t" ++ kind ++ "|" ++ shapeToString shape ++ "|" ++ data

def mkUnsupported (id reason : String) : String :=
  id ++ "\tUNSUPPORTED\t0\t" ++ reason

def mkErr (id reason : String) : String :=
  id ++ "\tERR\t0\t" ++ reason

def mkMalformed (id msg : String) : String :=
  id ++ "\tMALFORMED\t0\t" ++ msg

def processTensor (id : String) (ka : KindTag) (ta tb : TensorField) : String :=
  match ka with
  | .f32 =>
      match parseF32Data ta.dataTokens, parseF32Data tb.dataTokens with
      | some da, some db =>
          match bcastBinary (mulFloat fmt32) ta.shape da tb.shape db with
          | some (shape, out) => mkOk id ta.kind shape (dataToStringNat out)
          | none => mkErr id "broadcast failure"
      | _, _ => mkMalformed id "f32 data"
  | .f64 =>
      match parseF64Data ta.dataTokens, parseF64Data tb.dataTokens with
      | some da, some db =>
          match bcastBinary (mulFloat fmt64) ta.shape da tb.shape db with
          | some (shape, out) => mkOk id ta.kind shape (dataToStringNat out)
          | none => mkErr id "broadcast failure"
      | _, _ => mkMalformed id "f64 data"
  | .i bits =>
      match parseIntData bits ta.dataTokens, parseIntData bits tb.dataTokens with
      | some da, some db =>
          match bcastBinary (fun a b => (a * b) % 2 ^ bits) ta.shape da tb.shape db with
          | some (shape, out) => mkOk id ta.kind shape (dataToStringNat out)
          | none => mkErr id "broadcast failure"
      | _, _ => mkMalformed id "integer data"
  | .u bits =>
      match parseIntData bits ta.dataTokens, parseIntData bits tb.dataTokens with
      | some da, some db =>
          match bcastBinary (fun a b => (a * b) % 2 ^ bits) ta.shape da tb.shape db with
          | some (shape, out) => mkOk id ta.kind shape (dataToStringNat out)
          | none => mkErr id "broadcast failure"
      | _, _ => mkMalformed id "integer data"
  | .rat =>
      match parseRatData ta.dataTokens, parseRatData tb.dataTokens with
      | some da, some db =>
          let step : (Int × Nat) → (Int × Nat) → (Int × Nat) :=
            fun a b => (a.1 * b.1, a.2 * b.2)
          match bcastBinary step ta.shape da tb.shape db with
          | some (shape, out) => mkOk id ta.kind shape (dataToStringRat out)
          | none => mkErr id "broadcast failure"
      | _, _ => mkMalformed id "rat data"
  | .f16 => mkUnsupported id "float16"
  | .unsupported => mkUnsupported id "unsupported kind"

def runBinary (id : String) (ta tb : TensorField) : String :=
  let ka := parseKind ta.kind
  let kb := parseKind tb.kind
  match ka, kb with
  | .unsupported, _ => mkUnsupported id ("unsupported kind " ++ ta.kind)
  | _, .unsupported => mkUnsupported id ("unsupported kind " ++ tb.kind)
  | .f16, _ => mkUnsupported id "float16"
  | _, .f16 => mkUnsupported id "float16"
  | _, _ =>
      if ta.kind ≠ tb.kind then mkErr id "type mismatch"
      else processTensor id ka ta tb

def processLine (line : String) : String :=
  let line := trimCR line
  match line.splitOn "\t" with
  | id :: _attrs :: arityStr :: rest =>
      match String.toNat? arityStr with
      | none => mkMalformed id "arity"
      | some arity =>
          if arity ≠ 2 then mkErr id "unsupported arity"
          else
            match rest with
            | [sa, sb] =>
                match parseTensorField sa, parseTensorField sb with
                | some ta, some tb => runBinary id ta tb
                | _, _ => mkMalformed id "tensor field"
            | _ => mkMalformed id "tensor count"
  | id :: _ => mkMalformed id "missing fields"
  | [] => mkMalformed "" "empty"

/-! ## Main loop -/

def mainLoop : IO Unit := do
  let input ← IO.FS.readFile "/dev/stdin"
  let rawLines := input.splitOn "\n"
  let lines :=
    if input == "" then []
    else if endsWithNewline input then dropLastList rawLines
    else rawLines
  for line in lines do
    IO.println (processLine line)

end MulRunner

def main : IO Unit := MulRunner.mainLoop
