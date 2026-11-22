import Sonnx

namespace DivRunner

/-! ## Generic helpers -/

def pow2 (k : Nat) : Nat := 2 ^ k

def shl (n k : Nat) : Nat := n * pow2 k

def shr (n k : Nat) : Nat := n / pow2 k

def joinWith (sep : String) : List String → String
  | [] => ""
  | [x] => x
  | x :: xs => x ++ sep ++ joinWith sep xs

def listGet? {α : Type} (l : List α) (n : Nat) : Option α :=
  match l with
  | [] => none
  | x :: xs =>
      match n with
      | 0 => some x
      | k + 1 => listGet? xs k

def traverseOpt {α β : Type} (f : α → Option β) : List α → Option (List β)
  | [] => some []
  | x :: xs =>
      match f x with
      | none => none
      | some y =>
          match traverseOpt f xs with
          | none => none
          | some ys => some (y :: ys)

def mapMExcept {α β : Type} (f : α → Except String β) : List α → Except String (List β)
  | [] => .ok []
  | x :: xs =>
      match f x with
      | .error e => .error e
      | .ok y =>
          match mapMExcept f xs with
          | .error e => .error e
          | .ok ys => .ok (y :: ys)

def rangeNat (n : Nat) : List Nat :=
  let rec go (k : Nat) (acc : List Nat) : List Nat :=
    match k with
    | 0 => acc
    | m + 1 => go m (m :: acc)
  go n []

/-! ## Parsing primitives -/

def digitVal (c : Char) : Option Nat :=
  let v := c.toNat
  if v >= 48 ∧ v <= 57 then some (v - 48) else none

def parseNatAcc : List Char → Nat → Option Nat
  | [], acc => some acc
  | c :: rest, acc =>
      match digitVal c with
      | some d => parseNatAcc rest (acc * 10 + d)
      | none => none

def parseNat (s : String) : Option Nat :=
  match s.toList with
  | [] => none
  | cs => parseNatAcc cs 0

def parseSignedInt (s : String) : Option Int :=
  match s.toList with
  | '-' :: rest =>
      match parseNatAcc rest 0 with
      | some n => some (-(Int.ofNat n))
      | none => none
  | '+' :: rest =>
      match parseNatAcc rest 0 with
      | some n => some (Int.ofNat n)
      | none => none
  | cs =>
      match parseNatAcc cs 0 with
      | some n => some (Int.ofNat n)
      | none => none

/-! ## Shapes and broadcasting -/

def shapeSize (s : List Nat) : Nat :=
  s.foldl (fun acc d => acc * d) 1

def padLeft (n : Nat) (l : List Nat) : List Nat :=
  List.replicate (n - l.length) 1 ++ l

def bcastDim (x y : Nat) : Option Nat :=
  if x = y then some x
  else if x = 1 then some y
  else if y = 1 then some x
  else none

def broadcastShape (a b : List Nat) : Option (List Nat) :=
  let n := if a.length < b.length then b.length else a.length
  let a' := padLeft n a
  let b' := padLeft n b
  let rec go : List Nat → List Nat → Option (List Nat)
    | [], [] => some []
    | x :: xs, y :: ys =>
        match bcastDim x y with
        | none => none
        | some z =>
            match go xs ys with
            | none => none
            | some zs => some (z :: zs)
    | _, _ => none
  go a' b'

def unravel (shape : List Nat) (k : Nat) : List Nat :=
  match shape with
  | [] => []
  | d :: rest =>
      let stride := shapeSize rest
      let idx := if stride = 0 then 0 else if d = 0 then 0 else (k / stride) % d
      let k' := if stride = 0 then 0 else k % stride
      idx :: unravel rest k'

def bcastFlat (aShape cIdx : List Nat) : Nat :=
  let n := cIdx.length
  let a' := padLeft n aShape
  let rec go (as_ : List Nat) (is : List Nat) (acc : Nat) : Nat :=
    match as_, is with
    | [], [] => acc
    | ad :: ats, id :: ids =>
        let i := if ad = 1 then 0 else id
        go ats ids (acc * ad + i)
    | _, _ => acc
  go a' cIdx 0

/-! ## Signed / unsigned integer arithmetic -/

def signedValue (width : Nat) (bits : Nat) : Int :=
  let half := pow2 (width - 1)
  if bits < half then Int.ofNat bits
  else Int.ofNat bits - Int.ofNat (pow2 width)

def truncQuotInt (a b : Int) : Int :=
  let m : Int := Int.ofNat (a.natAbs / b.natAbs)
  if decide (a < 0) = decide (b < 0) then m else -m

def intToBits (width : Nat) (q : Int) : Nat :=
  let modn := pow2 width
  if q >= 0 then q.toNat % modn
  else
    let a := (-q).toNat % modn
    if a = 0 then 0 else modn - a

/-! ## Floating-point bit level division -/

def xorBool (a b : Bool) : Bool := if a = b then false else true

def pow2Nat (k : Nat) : Nat := 2 ^ k

def bitLenAux : Nat → Nat → Nat
  | 0, _ => 0
  | k + 1, n => if n < pow2Nat k then bitLenAux k n else k + 1

def bitLen (n : Nat) : Nat := bitLenAux 64 n

def makeZero (eBits mBits : Nat) (sign : Bool) : Nat :=
  if sign then pow2Nat (eBits + mBits) else 0

def makeInf (eBits mBits : Nat) (sign : Bool) : Nat :=
  (if sign then pow2Nat (eBits + mBits) else 0) + shl (pow2Nat eBits - 1) mBits

def makeNaN (eBits mBits : Nat) : Nat :=
  shl (pow2Nat eBits - 1) mBits + shl 1 (mBits - 1)

def makeFinite (eBits mBits : Nat) (sign : Bool) (expField mant : Nat) : Nat :=
  (if sign then pow2Nat (eBits + mBits) else 0) + shl expField mBits + mant

inductive FClass where
  | nan
  | inf (sign : Bool)
  | zero (sign : Bool)
  | fin (sign : Bool) (sig : Nat) (e : Int)

def classifyFloat (eBits mBits bits : Nat) : FClass :=
  let width := 1 + eBits + mBits
  let sign : Bool := decide ((shr bits (width - 1)) % 2 = 1)
  let exp := (shr bits mBits) % pow2Nat eBits
  let mant := bits % pow2Nat mBits
  let maxExp := pow2Nat eBits - 1
  let biasNat := pow2Nat (eBits - 1) - 1
  if exp = maxExp then
    if mant = 0 then FClass.inf sign else FClass.nan
  else if exp = 0 then
    if mant = 0 then FClass.zero sign
    else FClass.fin sign mant (1 - Int.ofNat biasNat - Int.ofNat mBits)
  else
    FClass.fin sign (pow2Nat mBits + mant)
      (Int.ofNat exp - Int.ofNat biasNat - Int.ofNat mBits)

def floorLog2Ratio (N D : Nat) : Int :=
  if N = 0 ∨ D = 0 then 0
  else
    let bN := bitLen N
    let bD := bitLen D
    let k := Int.ofNat bN - Int.ofNat bD
    if N * pow2Nat bD >= D * pow2Nat bN then k else k - 1

def roundQuotient (num den : Nat) : Nat :=
  if den = 0 then 0
  else
    let q := num / den
    let r := num % den
    if 2 * r > den then q + 1
    else if 2 * r = den then (if q % 2 = 1 then q + 1 else q)
    else q

def divFloatMag (eBits mBits : Nat) (sign : Bool) (N D : Nat) (E : Int) : Nat :=
  let biasNat := pow2Nat (eBits - 1) - 1
  let bias : Int := Int.ofNat biasNat
  let emin : Int := 1 - bias
  let emax : Int := bias
  let k := floorLog2Ratio N D
  let e := E + k
  if e > emax then makeInf eBits mBits sign
  else
    let base : Int := if e >= emin then e else emin
    let qexp : Int := base - Int.ofNat mBits
    let shift : Int := E - qexp
    let num := if shift >= 0 then N * pow2Nat shift.toNat else N
    let den := if shift >= 0 then D else D * pow2Nat (-shift).toNat
    let r := roundQuotient num den
    let top := pow2Nat mBits
    if r = 0 then makeZero eBits mBits sign
    else if r > top then
      let e2 : Int := base + 1
      if e2 > emax then makeInf eBits mBits sign
      else makeFinite eBits mBits sign (Int.toNat (e2 + bias)) 0
    else if r < top then
      makeFinite eBits mBits sign 0 r
    else
      makeFinite eBits mBits sign (Int.toNat (base + bias)) (r - top)

def divFloatBits (eBits mBits : Nat) (aBits bBits : Nat) : Nat :=
  match classifyFloat eBits mBits aBits, classifyFloat eBits mBits bBits with
  | FClass.nan, _ => makeNaN eBits mBits
  | _, FClass.nan => makeNaN eBits mBits
  | FClass.inf _, FClass.inf _ => makeNaN eBits mBits
  | FClass.zero _, FClass.zero _ => makeNaN eBits mBits
  | FClass.inf sa, FClass.zero sb => makeInf eBits mBits (xorBool sa sb)
  | FClass.zero sa, FClass.inf sb => makeZero eBits mBits (xorBool sa sb)
  | FClass.inf sa, FClass.fin sb _ _ => makeInf eBits mBits (xorBool sa sb)
  | FClass.fin sa _ _, FClass.inf sb => makeZero eBits mBits (xorBool sa sb)
  | FClass.fin sa _ _, FClass.zero sb => makeInf eBits mBits (xorBool sa sb)
  | FClass.zero sa, FClass.fin sb _ _ => makeZero eBits mBits (xorBool sa sb)
  | FClass.fin sa sigA eA, FClass.fin sb sigB eB =>
      divFloatMag eBits mBits (xorBool sa sb) sigA sigB (eA - eB)

/-! ## Rational helpers -/

def normRat (n d : Int) : Int × Int :=
  if d = 0 then (0, 1)
  else
    let s : Int := if d < 0 then -n else n
    let dd : Int := Int.ofNat d.natAbs
    let g : Nat := Nat.gcd s.natAbs dd.natAbs
    let gI : Int := Int.ofNat (if g = 0 then 1 else g)
    (s / gI, dd / gI)

def parseRatStr (s : String) : Option (Int × Int) :=
  match s.splitOn "/" with
  | [nStr, dStr] =>
      match parseSignedInt nStr, parseSignedInt dStr with
      | some n, some d => if d = 0 then none else some (normRat n d)
      | _, _ => none
  | _ => none

def parseRatList (data : List String) : Option (List (Int × Int)) :=
  traverseOpt parseRatStr data

/-! ## Kind classification -/

inductive KindInfo where
  | f16
  | f32
  | f64
  | i (width : Nat)
  | u (width : Nat)
  | rat

def parseKind (s : String) : Option KindInfo :=
  if s = "f16" ∨ s = "float16" then some KindInfo.f16
  else if s = "f32" ∨ s = "float32" then some KindInfo.f32
  else if s = "f64" ∨ s = "float64" then some KindInfo.f64
  else if s = "i8" ∨ s = "int8" then some (KindInfo.i 8)
  else if s = "i16" ∨ s = "int16" then some (KindInfo.i 16)
  else if s = "i32" ∨ s = "int32" then some (KindInfo.i 32)
  else if s = "i64" ∨ s = "int64" then some (KindInfo.i 64)
  else if s = "u8" ∨ s = "uint8" then some (KindInfo.u 8)
  else if s = "u16" ∨ s = "uint16" then some (KindInfo.u 16)
  else if s = "u32" ∨ s = "uint32" then some (KindInfo.u 32)
  else if s = "u64" ∨ s = "uint64" then some (KindInfo.u 64)
  else if s = "rat" then some KindInfo.rat
  else none

/-! ## Wire parsing -/

structure TensorField where
  kind : String
  shape : List Nat
  data : List String

structure Case where
  id : String
  attrs : String
  arity : Nat
  tensors : List TensorField

def parseShape (s : String) : Option (List Nat) :=
  if s = "-" then some []
  else if s.length = 0 then none
  else traverseOpt parseNat (s.splitOn ",")

def parseTensor (s : String) : Option TensorField :=
  match s.splitOn "|" with
  | [k, sh, da] =>
      match parseShape sh with
      | some shape =>
          let data := if da = "-" then [] else da.splitOn ","
          some { kind := k, shape := shape, data := data }
      | none => none
  | _ => none

def parseCase (line : String) : Option Case :=
  match line.splitOn "\t" with
  | id :: attrs :: arityStr :: rest =>
      match parseNat arityStr with
      | some ar =>
          if rest.length = ar then
            match traverseOpt parseTensor rest with
            | some ts => some { id := id, attrs := attrs, arity := ar, tensors := ts }
            | none => none
          else none
      | none => none
  | _ => none

def parseBitsList (width : Nat) (data : List String) : Option (List Nat) :=
  traverseOpt
    (fun s =>
      match parseNat s with
      | some n => if n < pow2Nat width then some n else none
      | none => none)
    data

/-! ## Tensor execution -/

def runInt (signed : Bool) (width : Nat)
    (shapeC shapeA shapeB : List Nat)
    (dataA dataB : List Nat) : Except String (List Nat) :=
  mapMExcept
    (fun k =>
      let idx := unravel shapeC k
      let ia := bcastFlat shapeA idx
      let ib := bcastFlat shapeB idx
      match listGet? dataA ia, listGet? dataB ib with
      | some ba, some bb =>
          if signed then
            let a := signedValue width ba
            let b := signedValue width bb
            if b = 0 then Except.error "division by zero"
            else Except.ok (intToBits width (truncQuotInt a b))
          else
            if bb = 0 then Except.error "division by zero"
            else Except.ok (ba / bb)
      | _, _ => Except.error "data index out of range")
    (rangeNat (shapeSize shapeC))

def runFloat (eBits mBits : Nat)
    (shapeC shapeA shapeB : List Nat)
    (dataA dataB : List Nat) : Except String (List Nat) :=
  mapMExcept
    (fun k =>
      let idx := unravel shapeC k
      let ia := bcastFlat shapeA idx
      let ib := bcastFlat shapeB idx
      match listGet? dataA ia, listGet? dataB ib with
      | some ba, some bb => Except.ok (divFloatBits eBits mBits ba bb)
      | _, _ => Except.error "data index out of range")
    (rangeNat (shapeSize shapeC))

def runRat (shapeC shapeA shapeB : List Nat)
    (dataA dataB : List (Int × Int)) : Except String (List (Int × Int)) :=
  mapMExcept
    (fun k =>
      let idx := unravel shapeC k
      let ia := bcastFlat shapeA idx
      let ib := bcastFlat shapeB idx
      match listGet? dataA ia, listGet? dataB ib with
      | some (na, da), some (nb, db) =>
          if nb = 0 then Except.error "division by zero"
          else Except.ok (normRat (na * db) (da * nb))
      | _, _ => Except.error "data index out of range")
    (rangeNat (shapeSize shapeC))

/-! ## Output formatting -/

def formatShape (s : List Nat) : String :=
  if s.length = 0 then "-" else joinWith "," (s.map toString)

def formatData (xs : List String) : String :=
  if xs.length = 0 then "-" else joinWith "," xs

def okLine (id kind : String) (shape : List Nat) (data : List String) : String :=
  id ++ "\tOK\t1\t" ++ kind ++ "|" ++ formatShape shape ++ "|" ++ formatData data

def errLine (id msg : String) : String :=
  id ++ "\tERR\t0\t" ++ msg

def malformedLine (id msg : String) : String :=
  id ++ "\tMALFORMED\t0\t" ++ msg

def unsupportedLine (id reason : String) : String :=
  id ++ "\tUNSUPPORTED\t0\t" ++ reason

/-! ## Dispatch -/

def dispatch (id kindStr : String) (ta tb : TensorField) (shapeC : List Nat) : String :=
  match parseKind kindStr with
  | none => unsupportedLine id ("unsupported kind: " ++ kindStr)
  | some k =>
      match k with
      | KindInfo.f16 => unsupportedLine id "float16 is not representable"
      | KindInfo.f32 =>
          match parseBitsList 32 ta.data, parseBitsList 32 tb.data with
          | some da, some db =>
              match runFloat 8 23 shapeC ta.shape tb.shape da db with
              | Except.ok out => okLine id kindStr shapeC (out.map toString)
              | Except.error e => errLine id e
          | _, _ => malformedLine id "invalid float32 bit pattern"
      | KindInfo.f64 =>
          match parseBitsList 64 ta.data, parseBitsList 64 tb.data with
          | some da, some db =>
              match runFloat 11 52 shapeC ta.shape tb.shape da db with
              | Except.ok out => okLine id kindStr shapeC (out.map toString)
              | Except.error e => errLine id e
          | _, _ => malformedLine id "invalid float64 bit pattern"
      | KindInfo.i width =>
          match parseBitsList width ta.data, parseBitsList width tb.data with
          | some da, some db =>
              match runInt true width shapeC ta.shape tb.shape da db with
              | Except.ok out => okLine id kindStr shapeC (out.map toString)
              | Except.error e => errLine id e
          | _, _ => malformedLine id "invalid signed integer bit pattern"
      | KindInfo.u width =>
          match parseBitsList width ta.data, parseBitsList width tb.data with
          | some da, some db =>
              match runInt false width shapeC ta.shape tb.shape da db with
              | Except.ok out => okLine id kindStr shapeC (out.map toString)
              | Except.error e => errLine id e
          | _, _ => malformedLine id "invalid unsigned integer bit pattern"
      | KindInfo.rat =>
          match parseRatList ta.data, parseRatList tb.data with
          | some da, some db =>
              match runRat shapeC ta.shape tb.shape da db with
              | Except.ok out =>
                  okLine id kindStr shapeC
                    (out.map (fun p => toString p.1 ++ "/" ++ toString p.2))
              | Except.error e => errLine id e
          | _, _ => malformedLine id "invalid rational data"

def runCase (c : Case) : String :=
  if c.arity ≠ 2 then malformedLine c.id "arity must be 2"
  else
    match c.tensors with
    | ta :: tb :: [] =>
        if ta.kind ≠ tb.kind then errLine c.id "type mismatch between operands"
        else if ta.data.length ≠ shapeSize ta.shape then
          malformedLine c.id "data length does not match shape of A"
        else if tb.data.length ≠ shapeSize tb.shape then
          malformedLine c.id "data length does not match shape of B"
        else
          match broadcastShape ta.shape tb.shape with
          | none => errLine c.id "incompatible shapes for broadcasting"
          | some shapeC => dispatch c.id ta.kind ta tb shapeC
    | _ => malformedLine c.id "tensor count does not match arity"

/-! ## Line handling -/

def stripCR (s : String) : String :=
  match s.toList.reverse with
  | '\r' :: rest => String.mk rest.reverse
  | _ => s

def processLine (line : String) : String :=
  match parseCase line with
  | some c => runCase c
  | none =>
      let id := match line.splitOn "\t" with
        | x :: _ => x
        | [] => ""
      malformedLine id "malformed line"

def splitLines (input : String) : List String :=
  let raw := input.splitOn "\n"
  match raw.reverse with
  | "" :: rest => rest.reverse
  | _ => raw

def processInput (input : String) : List String :=
  (splitLines input).map (fun l => processLine (stripCR l))

end DivRunner

open DivRunner

def main : IO Unit := do
  let stdin ← IO.getStdin
  let input ← stdin.readToEnd
  for line in processInput input do
    IO.println line
