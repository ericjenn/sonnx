import Sonnx

namespace DivRunner

inductive Kind where
  | f16 | f32 | f64
  | i8 | i16 | i32 | i64
  | u8 | u16 | u32 | u64
  | rat
  deriving BEq, DecidableEq

def kindName : Kind → String
  | .f16 => "f16"
  | .f32 => "f32"
  | .f64 => "f64"
  | .i8 => "i8"
  | .i16 => "i16"
  | .i32 => "i32"
  | .i64 => "i64"
  | .u8 => "u8"
  | .u16 => "u16"
  | .u32 => "u32"
  | .u64 => "u64"
  | .rat => "rat"

def parseKind (s : String) : Option Kind :=
  if s = "f16" then some .f16
  else if s = "float16" then some .f16
  else if s = "f32" then some .f32
  else if s = "float32" then some .f32
  else if s = "f64" then some .f64
  else if s = "float64" then some .f64
  else if s = "i8" then some .i8
  else if s = "int8" then some .i8
  else if s = "i16" then some .i16
  else if s = "int16" then some .i16
  else if s = "i32" then some .i32
  else if s = "int32" then some .i32
  else if s = "i64" then some .i64
  else if s = "int64" then some .i64
  else if s = "u8" then some .u8
  else if s = "uint8" then some .u8
  else if s = "u16" then some .u16
  else if s = "uint16" then some .u16
  else if s = "u32" then some .u32
  else if s = "uint32" then some .u32
  else if s = "u64" then some .u64
  else if s = "uint64" then some .u64
  else if s = "rat" then some .rat
  else none

def pow2 (n : Nat) : Nat := 2 ^ n

def joinWith (sep : String) : List String → String
  | [] => ""
  | x :: [] => x
  | x :: rest => x ++ sep ++ joinWith sep rest

def parseNatChars (cs : List Char) : Option Nat :=
  match cs with
  | [] => none
  | _ =>
    let rec go (cs : List Char) (acc : Nat) : Option Nat :=
      match cs with
      | [] => some acc
      | c :: rest =>
          let n := c.toNat
          if 48 ≤ n then
            if n ≤ 57 then go rest (acc * 10 + (n - 48)) else none
          else none
    go cs 0

def parseNat? (s : String) : Option Nat :=
  parseNatChars s.toList

def parseInt? (s : String) : Option Int :=
  match s.toList with
  | [] => none
  | '-' :: rest =>
      match parseNatChars rest with
      | some n => some (-(Int.ofNat n))
      | none => none
  | _ =>
      match parseNat? s with
      | some n => some (Int.ofNat n)
      | none => none

def normRat (n d : Int) : Option (Int × Int) :=
  if d = 0 then none
  else
    let den := if d < 0 then -d else d
    let num := if d < 0 then -n else n
    let g := Nat.gcd (Int.natAbs num) (Int.natAbs den)
    if g = 0 then some (0, 1)
    else
      let gi : Int := Int.ofNat g
      some (num / gi, den / gi)

def parseRatToken (s : String) : Option (Int × Int) :=
  match s.splitOn "/" with
  | [n, d] =>
      match parseInt? n, parseInt? d with
      | some a, some b => if b = 0 then none else normRat a b
      | _, _ => none
  | _ => none

structure TensorRaw where
  kind : Kind
  shape : List Nat
  data : List String

def parseNatList (xs : List String) : Option (List Nat) :=
  let rec go (xs : List String) (acc : List Nat) : Option (List Nat) :=
    match xs with
    | [] => some (List.reverse acc)
    | x :: rest =>
        match parseNat? x with
        | some n => go rest (n :: acc)
        | none => none
  go xs []

def parseShape (s : String) : Option (List Nat) :=
  if s = "-" then some []
  else if s = "" then none
  else parseNatList (s.splitOn ",")

def parseDataTokens (s : String) : Option (List String) :=
  if s = "-" then some []
  else if s = "" then none
  else some (s.splitOn ",")

def parseTensor (s : String) : Option TensorRaw :=
  match s.splitOn "|" with
  | [k, sh, da] =>
      match parseKind k with
      | some kind =>
          match parseShape sh with
          | some shape =>
              match parseDataTokens da with
              | some data => some { kind := kind, shape := shape, data := data }
              | none => none
          | none => none
      | none => none
  | _ => none

def prodShape (s : List Nat) : Nat :=
  List.foldl (fun acc d => acc * d) 1 s

def bcastDim (x y : Nat) : Option Nat :=
  if x = y then some x
  else if x = 1 then some y
  else if y = 1 then some x
  else none

def bcastShapeRev : List Nat → List Nat → Option (List Nat)
  | [], [] => some []
  | [], y :: yr =>
      match bcastShapeRev [] yr with
      | some rest => some (y :: rest)
      | none => none
  | x :: xr, [] =>
      match bcastShapeRev xr [] with
      | some rest => some (x :: rest)
      | none => none
  | x :: xr, y :: yr =>
      match bcastDim x y, bcastShapeRev xr yr with
      | some d, some rest => some (d :: rest)
      | _, _ => none

def bcastShape (a b : List Nat) : Option (List Nat) :=
  match bcastShapeRev (List.reverse a) (List.reverse b) with
  | some r => some (List.reverse r)
  | none => none

def coordsOf (shape : List Nat) (idx : Nat) : List Nat :=
  let rec go (shape : List Nat) (idx : Nat) : List Nat :=
    match shape with
    | [] => []
    | d :: ds =>
        let rest := prodShape ds
        if rest = 0 then
          0 :: go ds 0
        else
          let c := if d = 0 then 0 else (idx / rest) % d
          c :: go ds (idx % rest)
  go shape idx

def dropN {α : Type} : Nat → List α → List α
  | 0, xs => xs
  | Nat.succ _, [] => []
  | Nat.succ k, _ :: rest => dropN k rest

def bcastOffset (opShape resCoords : List Nat) : Nat :=
  let extra := resCoords.length - opShape.length
  let aligned := dropN extra resCoords
  let rec go (shape coords : List Nat) (acc : Nat) : Nat :=
    match shape, coords with
    | [], _ => acc
    | d :: ds, c :: cs =>
        let cc := if d = 1 then 0 else c
        go ds cs (acc * d + cc)
    | _ :: _, [] => acc
  go opShape aligned 0

def indexPairs (rshape ashape bshape : List Nat) : List (Nat × Nat) :=
  (List.range (prodShape rshape)).map (fun idx =>
    let coords := coordsOf rshape idx
    (bcastOffset ashape coords, bcastOffset bshape coords))

def nth? {α : Type} : List α → Nat → Option α
  | [], _ => none
  | x :: _, 0 => some x
  | _ :: rest, Nat.succ k => nth? rest k

def truncDivInt (a b : Int) : Int :=
  if b = 0 then 0
  else
    let q : Nat := Int.natAbs a / Int.natAbs b
    if a < 0 then
      if b < 0 then Int.ofNat q else -(Int.ofNat q)
    else
      if b < 0 then -(Int.ofNat q) else Int.ofNat q

def wrapNat (n : Nat) (q : Int) : Nat :=
  let m := pow2 n
  if q >= 0 then
    (Int.toNat q) % m
  else
    let a := (Int.natAbs q) % m
    (m - a) % m

def decodeSInt (w : Nat) (raw : Nat) : Int :=
  let r := raw % pow2 w
  let half := pow2 (w - 1)
  if r < half then Int.ofNat r
  else Int.ofNat r - Int.ofNat (pow2 w)

inductive FClass where
  | nan
  | inf (neg : Bool)
  | zero (neg : Bool)
  | finite (neg : Bool) (num : Nat) (den : Nat)

def decodeFloat (w : Nat) (raw : Nat) : FClass :=
  let p := if w = 32 then 23 else 52
  let expBits := if w = 32 then 8 else 11
  let bias := if w = 32 then 127 else 1023
  let r := raw % pow2 w
  let signBit := (r / pow2 (p + expBits)) % 2
  let neg : Bool := if signBit = 1 then true else false
  let e := (r / pow2 p) % pow2 expBits
  let f := r % pow2 p
  let maxExp := pow2 expBits - 1
  if e = maxExp then
    if f = 0 then FClass.inf neg else FClass.nan
  else if e = 0 then
    if f = 0 then FClass.zero neg
    else FClass.finite neg f (pow2 (bias + p - 1))
  else
    let sig := pow2 p + f
    let sh : Int := (Int.ofNat e) - (Int.ofNat bias) - (Int.ofNat p)
    if sh >= 0 then
      FClass.finite neg (sig * pow2 (Int.toNat sh)) 1
    else
      FClass.finite neg sig (pow2 (Int.toNat (-sh)))

def xorBool (a b : Bool) : Bool :=
  if a = b then false else true

def gePow (N D : Nat) (e : Int) : Bool :=
  if e >= 0 then
    N >= D * pow2 (Int.toNat e)
  else
    N * pow2 (Int.toNat (-e)) >= D

def floorLog2 (N D : Nat) : Int :=
  if N = 0 then 0
  else if D = 0 then 0
  else
    let a := Nat.log2 N
    let b := Nat.log2 D
    let base : Int := (Int.ofNat a) - (Int.ofNat b)
    if gePow N D base then base else base - 1

def roundDivNat (N D : Nat) : Nat :=
  if D = 0 then 0
  else
    let q := N / D
    let r := N % D
    let tw := 2 * r
    if tw < D then q
    else if tw > D then q + 1
    else if q % 2 = 0 then q else q + 1

def roundScaled (N D : Nat) (shift : Int) : Nat :=
  if shift >= 0 then
    roundDivNat (N * pow2 (Int.toNat shift)) D
  else
    roundDivNat N (D * pow2 (Int.toNat (-shift)))

def encode (w : Nat) (neg : Bool) (expField frac : Nat) : Nat :=
  let p := if w = 32 then 23 else 52
  let signBit := if neg then pow2 (w - 1) else 0
  signBit + expField * pow2 p + frac

def infBits (w : Nat) (neg : Bool) : Nat :=
  let expBits := if w = 32 then 8 else 11
  encode w neg (pow2 expBits - 1) 0

def zeroBits (w : Nat) (neg : Bool) : Nat :=
  encode w neg 0 0

def nanBits (w : Nat) : Nat :=
  let p := if w = 32 then 23 else 52
  let expBits := if w = 32 then 8 else 11
  encode w false (pow2 expBits - 1) (pow2 (p - 1))

def roundFloatMag (w : Nat) (neg : Bool) (N D : Nat) : Nat :=
  if N = 0 then zeroBits w neg
  else if D = 0 then infBits w neg
  else
    let p := if w = 32 then 23 else 52
    let bias := if w = 32 then 127 else 1023
    let e := floorLog2 N D
    let emin : Int := 1 - (Int.ofNat bias)
    let emax : Int := Int.ofNat bias
    if e >= emin then
      let shift : Int := (Int.ofNat p) - e
      let m := roundScaled N D shift
      let twoP := pow2 p
      let twoP1 := twoP * 2
      if e > emax then
        infBits w neg
      else if m >= twoP1 then
        let m2 := m / 2
        let e2 := e + 1
        if e2 > emax then infBits w neg
        else
          let expField := Int.toNat (e2 + Int.ofNat bias)
          let frac := m2 - twoP
          encode w neg expField frac
      else if m < twoP then
        encode w neg 0 m
      else
        let expField := Int.toNat (e + Int.ofNat bias)
        let frac := m - twoP
        encode w neg expField frac
    else
      let shift : Int := (Int.ofNat p) - emin
      let m := roundScaled N D shift
      let twoP := pow2 p
      if m = 0 then zeroBits w neg
      else if m >= twoP then encode w neg 1 0
      else encode w neg 0 m

def floatDivOp (w : Nat) (aBits bBits : Nat) : Nat :=
  let a := decodeFloat w aBits
  let b := decodeFloat w bBits
  match a, b with
  | .nan, _ => nanBits w
  | _, .nan => nanBits w
  | .inf sa, .inf sb => nanBits w
  | .zero sa, .zero sb => nanBits w
  | .inf sa, .zero sb => infBits w (xorBool sa sb)
  | .inf sa, .finite sb _ _ => infBits w (xorBool sa sb)
  | .zero sa, .inf sb => zeroBits w (xorBool sa sb)
  | .finite sa _ _, .inf sb => zeroBits w (xorBool sa sb)
  | .zero sa, .finite sb _ _ => zeroBits w (xorBool sa sb)
  | .finite sa _ _, .zero sb => infBits w (xorBool sa sb)
  | .finite sa na da, .finite sb nb db =>
      let neg := xorBool sa sb
      roundFloatMag w neg (na * db) (da * nb)

def parseBitData (xs : List String) : Option (List Nat) :=
  let rec go (xs : List String) (acc : List Nat) : Option (List Nat) :=
    match xs with
    | [] => some (List.reverse acc)
    | x :: rest =>
        match parseNat? x with
        | some n => go rest (n :: acc)
        | none => none
  go xs []

def parseRatData (xs : List String) : Option (List (Int × Int)) :=
  let rec go (xs : List String) (acc : List (Int × Int)) : Option (List (Int × Int)) :=
    match xs with
    | [] => some (List.reverse acc)
    | x :: rest =>
        match parseRatToken x with
        | some r => go rest (r :: acc)
        | none => none
  go xs []

def ratDiv (a b : Int × Int) : Option (Int × Int) :=
  let (an, ad) := a
  let (bn, bd) := b
  if bn = 0 then none
  else normRat (an * bd) (ad * bn)

inductive CaseResult where
  | ok (kind : Kind) (shape : List Nat) (data : List String)
  | unsupported (reason : String)
  | err (reason : String)
  | malformed (message : String)

def computeFloat (w : Nat) (t1 t2 : TensorRaw) : CaseResult :=
  match parseBitData t1.data, parseBitData t2.data with
  | some da, some db =>
      if da.length ≠ prodShape t1.shape then
        .malformed "data length does not match shape"
      else if db.length ≠ prodShape t2.shape then
        .malformed "data length does not match shape"
      else
        match bcastShape t1.shape t2.shape with
        | none => .err "incompatible shapes"
        | some rshape =>
            let pairs := indexPairs rshape t1.shape t2.shape
            let rec loop (ps : List (Nat × Nat)) (acc : List String) : Except String (List String) :=
              match ps with
              | [] => .ok (List.reverse acc)
              | (ai, bi) :: rest =>
                  match nth? da ai, nth? db bi with
                  | some a, some b => loop rest (toString (floatDivOp w a b) :: acc)
                  | _, _ => .error "broadcast index out of range"
            match loop pairs [] with
            | .ok out => .ok t1.kind rshape out
            | .error e => .err e
  | _, _ => .malformed "bad float data"

def computeSigned (w : Nat) (t1 t2 : TensorRaw) : CaseResult :=
  match parseBitData t1.data, parseBitData t2.data with
  | some da, some db =>
      if da.length ≠ prodShape t1.shape then
        .malformed "data length does not match shape"
      else if db.length ≠ prodShape t2.shape then
        .malformed "data length does not match shape"
      else
        match bcastShape t1.shape t2.shape with
        | none => .err "incompatible shapes"
        | some rshape =>
            let pairs := indexPairs rshape t1.shape t2.shape
            let rec loop (ps : List (Nat × Nat)) (acc : List String) : Except String (List String) :=
              match ps with
              | [] => .ok (List.reverse acc)
              | (ai, bi) :: rest =>
                  match nth? da ai, nth? db bi with
                  | some a, some b =>
                      let av := decodeSInt w a
                      let bv := decodeSInt w b
                      if bv = 0 then .error "division by zero"
                      else
                        let q := truncDivInt av bv
                        loop rest (toString (wrapNat w q) :: acc)
                  | _, _ => .error "broadcast index out of range"
            match loop pairs [] with
            | .ok out => .ok t1.kind rshape out
            | .error e => .err e
  | _, _ => .malformed "bad integer data"

def computeUnsigned (w : Nat) (t1 t2 : TensorRaw) : CaseResult :=
  match parseBitData t1.data, parseBitData t2.data with
  | some da, some db =>
      if da.length ≠ prodShape t1.shape then
        .malformed "data length does not match shape"
      else if db.length ≠ prodShape t2.shape then
        .malformed "data length does not match shape"
      else
        match bcastShape t1.shape t2.shape with
        | none => .err "incompatible shapes"
        | some rshape =>
            let pairs := indexPairs rshape t1.shape t2.shape
            let rec loop (ps : List (Nat × Nat)) (acc : List String) : Except String (List String) :=
              match ps with
              | [] => .ok (List.reverse acc)
              | (ai, bi) :: rest =>
                  match nth? da ai, nth? db bi with
                  | some a, some b =>
                      let an := a % pow2 w
                      let bn := b % pow2 w
                      if bn = 0 then .error "division by zero"
                      else loop rest (toString (an / bn) :: acc)
                  | _, _ => .error "broadcast index out of range"
            match loop pairs [] with
            | .ok out => .ok t1.kind rshape out
            | .error e => .err e
  | _, _ => .malformed "bad integer data"

def computeRat (t1 t2 : TensorRaw) : CaseResult :=
  match parseRatData t1.data, parseRatData t2.data with
  | some da, some db =>
      if da.length ≠ prodShape t1.shape then
        .malformed "data length does not match shape"
      else if db.length ≠ prodShape t2.shape then
        .malformed "data length does not match shape"
      else
        match bcastShape t1.shape t2.shape with
        | none => .err "incompatible shapes"
        | some rshape =>
            let pairs := indexPairs rshape t1.shape t2.shape
            let rec loop (ps : List (Nat × Nat)) (acc : List String) : Except String (List String) :=
              match ps with
              | [] => .ok (List.reverse acc)
              | (ai, bi) :: rest =>
                  match nth? da ai, nth? db bi with
                  | some a, some b =>
                      match ratDiv a b with
                      | some r =>
                          let (n, d) := r
                          loop rest ((toString n ++ "/" ++ toString d) :: acc)
                      | none => .error "division by zero"
                  | _, _ => .error "broadcast index out of range"
            match loop pairs [] with
            | .ok out => .ok Kind.rat rshape out
            | .error e => .err e
  | _, _ => .malformed "bad rational data"

def computeDiv (t1 t2 : TensorRaw) : CaseResult :=
  if t1.kind == Kind.f16 || t2.kind == Kind.f16 then
    .unsupported "float16 is not representable"
  else if t1.kind == t2.kind then
    match t1.kind with
    | .f32 => computeFloat 32 t1 t2
    | .f64 => computeFloat 64 t1 t2
    | .i8 => computeSigned 8 t1 t2
    | .i16 => computeSigned 16 t1 t2
    | .i32 => computeSigned 32 t1 t2
    | .i64 => computeSigned 64 t1 t2
    | .u8 => computeUnsigned 8 t1 t2
    | .u16 => computeUnsigned 16 t1 t2
    | .u32 => computeUnsigned 32 t1 t2
    | .u64 => computeUnsigned 64 t1 t2
    | .rat => computeRat t1 t2
    | .f16 => .unsupported "float16 is not representable"
  else
    .err "operand types differ"

def formatShape (s : List Nat) : String :=
  match s with
  | [] => "-"
  | _ => joinWith "," (s.map toString)

def formatData (xs : List String) : String :=
  match xs with
  | [] => "-"
  | _ => joinWith "," xs

def formatOk (id : String) (kind : Kind) (shape : List Nat) (data : List String) : String :=
  id ++ "\tOK\t1\t" ++ kindName kind ++ "|" ++ formatShape shape ++ "|" ++ formatData data

def processLine (line : String) : String :=
  let line := String.trim line
  match line.splitOn "\t" with
  | id :: _attrs :: arityStr :: rest =>
      match parseNat? arityStr with
      | some arity =>
          if arity ≠ 2 then
            id ++ "\tMALFORMED\t0\tarity must be 2"
          else
            match rest with
            | [t1, t2] =>
                match parseTensor t1, parseTensor t2 with
                | some a, some b =>
                    match computeDiv a b with
                    | .ok kind shape data => formatOk id kind shape data
                    | .unsupported reason => id ++ "\tUNSUPPORTED\t0\t" ++ reason
                    | .err reason => id ++ "\tERR\t0\t" ++ reason
                    | .malformed msg => id ++ "\tMALFORMED\t0\t" ++ msg
                | _, _ => id ++ "\tMALFORMED\t0\tbad tensor field"
            | _ => id ++ "\tMALFORMED\t0\tarity field mismatch"
      | none => id ++ "\tMALFORMED\t0\tbad arity"
  | _ => "\tMALFORMED\t0\tbad field count"

def splitLines (s : String) : List String :=
  if s = "" then []
  else
    let parts := s.splitOn "\n"
    match List.reverse parts with
    | [] => parts
    | last :: rest =>
        if last = "" then List.reverse rest else parts

end DivRunner

open DivRunner

def main : IO Unit := do
  let stdin ← IO.getStdin
  let content ← stdin.readToEnd
  for line in splitLines content do
    IO.println (processLine line)
