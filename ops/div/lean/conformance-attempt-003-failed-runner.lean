import Sonnx

/-!
Executable conformance runner for ONNX `Div` (opset 14).

Input: one tab-separated request per line.  Output: exactly one tab-separated
response per request line, in the same order and with the same id.  Integers and
floats travel as raw bit patterns; rationals as `numerator/denominator`.  A kind
Lean core cannot represent (float16) is answered with `UNSUPPORTED`.

Float division is computed from the bit patterns: the exact quotient of the two
dyadic rationals is formed as an integer fraction and rounded to the format with
roundTiesToEven, the exponent range being unbounded for the overflow decision.
-/

namespace DivRunner

/-! ## Small string and list helpers -/

def joinWith (sep : String) : List String → String
  | [] => ""
  | [x] => x
  | x :: xs => x ++ sep ++ joinWith sep xs

/-- Drop a trailing carriage return, tolerating CRLF input lines. -/
def stripCR (s : String) : String :=
  match s.splitOn "\r" with
  | [] => s
  | x :: _ => x

def at? {α : Type} : List α → Nat → Option α
  | [], _ => none
  | x :: _, 0 => some x
  | _ :: xs, n + 1 => at? xs n

def splitLines (s : String) : List String :=
  let ps := (s.splitOn "\n").reverse
  match ps with
  | [] => []
  | p :: rest => if p.isEmpty then rest.reverse else ps.reverse

def charsToString (cs : List Char) : String :=
  cs.foldl (fun acc c => acc ++ c.toString) ""

/-! ## Field parsers -/

def parseNat? (s : String) : Option Nat :=
  if s.isEmpty then none else s.toNat?

def parseInt? (s : String) : Option Int :=
  match s.toList with
  | '-' :: ds =>
      match String.toNat? (charsToString ds) with
      | some n => some (Int.negOfNat n)
      | none => none
  | _ =>
      match String.toNat? s with
      | some n => some (Int.ofNat n)
      | none => none

def parseShape (s : String) : Option (List Nat) :=
  if s = "-" then some []
  else if s.isEmpty then none
  else (s.splitOn ",").mapM parseNat?

def parseData (s : String) : List String :=
  if s = "-" then []
  else if s.isEmpty then [] else s.splitOn ","

structure TRaw where
  kind : String
  shape : List Nat
  data : List String

def parseTensor (s : String) : Option TRaw :=
  match s.splitOn "|" with
  | [k, sh, da] =>
      if k.isEmpty then none
      else
        match parseShape sh with
        | some shp => some ⟨k, shp, parseData da⟩
        | none => none
  | _ => none

/-! ## Shapes, broadcasting, index translation -/

def shapeSize (s : List Nat) : Nat := s.foldl (fun acc d => acc * d) 1

def padLeft (n : Nat) (s : List Nat) : List Nat :=
  List.replicate (n - s.length) 1 ++ s

def bcastDim (a b : Nat) : Option Nat :=
  if a = b then some a
  else if a = 1 then some b
  else if b = 1 then some a
  else none

def bcastShape (sa sb : List Nat) : Option (List Nat) :=
  let n := max sa.length sb.length
  List.mapM (fun p => bcastDim p.1 p.2) ((padLeft n sa).zip (padLeft n sb))

/-- Translate a flat index of the result to the flat index of an operand,
    walking the dimensions from the last one backwards. -/
def idxAux : List Nat → List Nat → Nat → Nat → Nat → Option Nat
  | [], _, _, _, acc => some acc
  | od :: ods, opd, rem, stride, acc =>
      if od = 0 then none
      else
        let coord := rem % od
        let rem' := rem / od
        match opd with
        | [] => idxAux ods [] rem' stride acc
        | pd :: pds =>
            if pd = 0 then none
            else
              let c := if pd = 1 then 0 else coord
              if c ≥ pd then none
              else idxAux ods pds rem' (stride * pd) (acc + c * stride)

def flatIndex (outShape opShape : List Nat) (i : Nat) : Option Nat :=
  idxAux outShape.reverse opShape.reverse i 1 0

def elemPairs (outShape aShape bShape : List Nat) : Option (List (Nat × Nat)) :=
  List.mapM (fun i =>
    match flatIndex outShape aShape i, flatIndex outShape bShape i with
    | some x, some y => some (x, y)
    | _, _ => none) (List.range (shapeSize outShape))

/-! ## Rendering -/

def fmtShape (s : List Nat) : String :=
  if s.isEmpty then "-" else joinWith "," (s.map toString)

def fmtData (xs : List String) : String :=
  if xs.isEmpty then "-" else joinWith "," xs

def renderOk (id : String) (shape : List Nat) (kind : String) (data : List String) : String :=
  id ++ "\tOK\t1\t" ++ kind ++ "|" ++ fmtShape shape ++ "|" ++ fmtData data

def renderUnsupported (id : String) (reason : String) : String :=
  id ++ "\tUNSUPPORTED\t0\t" ++ reason

def renderErr (id : String) (reason : String) : String :=
  id ++ "\tERR\t0\t" ++ reason

def renderMalformed (id : String) (msg : String) : String :=
  id ++ "\tMALFORMED\t0\t" ++ msg

/-! ## Bit length -/

def bitLenAux : Nat → Nat → Nat → Nat
  | 0, _, acc => acc
  | fuel + 1, n, acc => if n = 0 then acc else bitLenAux fuel (n / 2) (acc + 1)

/-- Number of significant bits of `n` (`0` for `n = 0`). -/
def bitLen (n : Nat) : Nat := bitLenAux 4200 n 0

/-! ## Floating-point formats -/

structure Fmt where
  ebits : Nat
  mbits : Nat

def fmt32 : Fmt := ⟨8, 23⟩
def fmt64 : Fmt := ⟨11, 52⟩

def biasOf (fmt : Fmt) : Int := Int.ofNat (2 ^ (fmt.ebits - 1) - 1)

def emaxOf (fmt : Fmt) : Int := biasOf fmt

def eminOf (fmt : Fmt) : Int := 1 - biasOf fmt

inductive FCls where
  | nan
  | inf (sign : Bool)
  | zero (sign : Bool)
  | fin (sign : Bool) (mant : Nat) (expo : Int)

def signOfBits (fmt : Fmt) (bits : Nat) : Bool :=
  decide (bits / 2 ^ (fmt.ebits + fmt.mbits) % 2 = 1)

def classify (fmt : Fmt) (bits : Nat) : FCls :=
  let sign := signOfBits fmt bits
  let expField := bits / 2 ^ fmt.mbits % 2 ^ fmt.ebits
  let frac := bits % 2 ^ fmt.mbits
  let maxExp := 2 ^ fmt.ebits - 1
  if expField = maxExp then
    (if frac = 0 then FCls.inf sign else FCls.nan)
  else if expField = 0 then
    (if frac = 0 then FCls.zero sign
     else FCls.fin sign frac (1 - biasOf fmt - Int.ofNat fmt.mbits))
  else
    FCls.fin sign (2 ^ fmt.mbits + frac)
      (Int.ofNat expField - biasOf fmt - Int.ofNat fmt.mbits)

/-- The quiet NaN carrying the sign and the low payload of `bits`. -/
def nanFrom (fmt : Fmt) (bits : Nat) : Nat :=
  let s := if signOfBits fmt bits then 2 ^ (fmt.ebits + fmt.mbits) else 0
  let low := bits % 2 ^ (fmt.mbits - 1)
  s + (2 ^ fmt.ebits - 1) * 2 ^ fmt.mbits + 2 ^ (fmt.mbits - 1) + low

/-- The default quiet NaN of the invalid operations (sign bit set). -/
def defaultNaN (fmt : Fmt) : Nat :=
  2 ^ (fmt.ebits + fmt.mbits) + (2 ^ fmt.ebits - 1) * 2 ^ fmt.mbits
    + 2 ^ (fmt.mbits - 1)

def infBits (fmt : Fmt) (sign : Bool) : Nat :=
  (if sign then 2 ^ (fmt.ebits + fmt.mbits) else 0)
    + (2 ^ fmt.ebits - 1) * 2 ^ fmt.mbits

def zeroBits (fmt : Fmt) (sign : Bool) : Nat :=
  if sign then 2 ^ (fmt.ebits + fmt.mbits) else 0

/-- Round the positive rational `N / D` to the format, with roundTiesToEven and an
    unbounded exponent range.  `none` means the rounded magnitude exceeds the
    largest finite value (overflow to infinity). -/
def roundRat (fmt : Fmt) (N D : Nat) : Option (Nat × Int) :=
  let p := fmt.mbits + 1
  let k0 : Int := Int.ofNat (bitLen N) - Int.ofNat (bitLen D)
  let ge : Bool :=
    if k0 ≥ 0 then decide (N ≥ D * 2 ^ k0.toNat)
    else decide (N * 2 ^ (-k0).toNat ≥ D)
  let k : Int := if ge then k0 else k0 - 1
  let sh : Int := Int.ofNat p - 1
  let hi := k - sh
  let lo := eminOf fmt - sh
  let eUlp : Int := if hi ≥ lo then hi else lo
  let PQ : Nat × Nat :=
    if eUlp ≤ 0 then (N * 2 ^ (-eUlp).toNat, D) else (N, D * 2 ^ eUlp.toNat)
  let m0 := PQ.1 / PQ.2
  let r := PQ.1 % PQ.2
  let m1 :=
    if 2 * r > PQ.2 then m0 + 1
    else if 2 * r < PQ.2 then m0
    else if m0 % 2 = 0 then m0 else m0 + 1
  let carry := m1 ≥ 2 ^ p
  let e1 : Int := if carry then eUlp + 1 else eUlp
  let m2 : Nat := if carry then m1 / 2 else m1
  if m2 = 0 then some (0, e1)
  else if m2 ≥ 2 ^ (p - 1) then
    (if e1 + Int.ofNat p - 1 > emaxOf fmt then none else some (m2, e1))
  else some (m2, e1)

/-- Encode a rounded value `m * 2^e` (with `0 ≤ m < 2^p`) as a bit pattern. -/
def encodeFloat (fmt : Fmt) (sign : Bool) (m : Nat) (e : Int) : Nat :=
  let p := fmt.mbits + 1
  let sbit := if sign then 2 ^ (fmt.ebits + fmt.mbits) else 0
  if m = 0 then sbit
  else if m < 2 ^ (p - 1) then sbit + m
  else
    let ex : Int := e + Int.ofNat p - 1
    let ef : Nat := (ex + biasOf fmt).toNat
    sbit + ef * 2 ^ fmt.mbits + (m - 2 ^ (p - 1))

/-- IEEE 754 division on the bit patterns of one format. -/
def divFloatBits (fmt : Fmt) (ab bb : Nat) : Nat :=
  let sa := signOfBits fmt ab
  let sb := signOfBits fmt bb
  let sr := sa != sb
  match classify fmt ab, classify fmt bb with
  | FCls.nan, _ => nanFrom fmt ab
  | _, FCls.nan => nanFrom fmt bb
  | FCls.inf _, FCls.inf _ => defaultNaN fmt
  | FCls.inf _, _ => infBits fmt sr
  | _, FCls.inf _ => zeroBits fmt sr
  | FCls.zero _, FCls.zero _ => defaultNaN fmt
  | FCls.zero _, _ => zeroBits fmt sr
  | _, FCls.zero _ => infBits fmt sr
  | FCls.fin _ ma ea, FCls.fin _ mb eb =>
      let d : Int := ea - eb
      let nd : Nat × Nat :=
        if d ≥ 0 then (ma * 2 ^ d.toNat, mb) else (ma, mb * 2 ^ (-d).toNat)
      match roundRat fmt nd.1 nd.2 with
      | none => infBits fmt sr
      | some me => encodeFloat fmt sr me.1 me.2
  | _, _ => defaultNaN fmt

/-! ## Integer element steps -/

def truncDiv (a b : Int) : Int :=
  if b = 0 then 0
  else
    let q := a.natAbs / b.natAbs
    if (a < 0) = (b < 0) then Int.ofNat q else Int.negOfNat q

/-- Raw bit pattern of the value `q` read modulo `2^width`. -/
def wrapBits (width : Nat) (q : Int) : Nat :=
  let m := 2 ^ width
  if q ≥ 0 then q.toNat % m
  else
    let a := q.natAbs % m
    if a = 0 then 0 else m - a

def parseSInt (width : Nat) (s : String) : Option Int :=
  match parseNat? s with
  | none => none
  | some bits =>
      if bits ≥ 2 ^ width then none
      else if bits < 2 ^ (width - 1) then some (Int.ofNat bits)
      else some (Int.ofNat bits - Int.ofNat (2 ^ width))

def parseUInt (width : Nat) (s : String) : Option Nat :=
  match parseNat? s with
  | none => none
  | some bits => if bits ≥ 2 ^ width then none else some bits

def sdivOut (width : Nat) (ad bd : List Int) (p : Nat × Nat) : Except String Nat :=
  match at? ad p.1, at? bd p.2 with
  | some x, some y =>
      if y = 0 then Except.error "division by zero"
      else Except.ok (wrapBits width (truncDiv x y))
  | _, _ => Except.error "data index error"

def udivOut (ad bd : List Nat) (p : Nat × Nat) : Except String Nat :=
  match at? ad p.1, at? bd p.2 with
  | some x, some y =>
      if y = 0 then Except.error "division by zero"
      else Except.ok (x / y)
  | _, _ => Except.error "data index error"

/-! ## Rational element step -/

structure Q where
  num : Int
  den : Nat

def gcdAux : Nat → Nat → Nat → Nat
  | 0, a, _ => a
  | _, a, 0 => a
  | f + 1, a, b + 1 => gcdAux f (b + 1) (a % (b + 1))

def gcd (a b : Nat) : Nat := gcdAux (a + b + 1) a b

def mkQ (n : Int) (d : Nat) : Q :=
  if d = 0 then ⟨0, 1⟩
  else
    let g := gcd n.natAbs d
    if g = 0 then ⟨0, 1⟩ else ⟨truncDiv n (Int.ofNat g), d / g⟩

def qdiv (x y : Q) : Option Q :=
  if y.num = 0 then none
  else
    let n := x.num * Int.ofNat y.den
    let d := Int.ofNat x.den * y.num
    if d > 0 then some (mkQ n d.toNat) else some (mkQ (-n) (-d).toNat)

def qStr (q : Q) : String := toString q.num ++ "/" ++ toString q.den

def parseRat (s : String) : Option Q :=
  match s.splitOn "/" with
  | [ns, ds] =>
      match parseInt? ns, parseNat? ds with
      | some n, some d => if d = 0 then none else some (mkQ n d)
      | _, _ => none
  | _ => none

def rdivOut (ad bd : List Q) (p : Nat × Nat) : Except String String :=
  match at? ad p.1, at? bd p.2 with
  | some x, some y =>
      match qdiv x y with
      | some c => Except.ok (qStr c)
      | none => Except.error "division by zero"
  | _, _ => Except.error "data index error"

/-! ## Per-kind runners -/

def runFloat (fmt : Fmt) (kname : String) (id : String) (outShape : List Nat)
    (a b : TRaw) : String :=
  let w := fmt.ebits + fmt.mbits + 1
  match List.mapM parseNat? a.data, List.mapM parseNat? b.data with
  | some ad, some bd =>
      if ad.any (fun x => decide (x ≥ 2 ^ w)) || bd.any (fun x => decide (x ≥ 2 ^ w)) then
        renderMalformed id "float bit pattern out of range"
      else
        match elemPairs outShape a.shape b.shape with
        | none => renderErr id "broadcast index error"
        | some ps =>
            match List.mapM (fun p =>
                match at? ad p.1, at? bd p.2 with
                | some x, some y => some (divFloatBits fmt x y)
                | _, _ => none) ps with
            | some outs => renderOk id outShape kname (outs.map toString)
            | none => renderErr id "data index error"
  | _, _ => renderMalformed id "bad floating-point data"

def runSInt (width : Nat) (kname : String) (id : String) (outShape : List Nat)
    (a b : TRaw) : String :=
  match List.mapM (parseSInt width) a.data, List.mapM (parseSInt width) b.data with
  | some ad, some bd =>
      match elemPairs outShape a.shape b.shape with
      | none => renderErr id "broadcast index error"
      | some ps =>
          match List.mapM (fun p => sdivOut width ad bd p) ps with
          | Except.ok qs => renderOk id outShape kname (qs.map toString)
          | Except.error msg => renderErr id msg
  | _, _ => renderMalformed id "bad signed integer data"

def runUInt (width : Nat) (kname : String) (id : String) (outShape : List Nat)
    (a b : TRaw) : String :=
  match List.mapM (parseUInt width) a.data, List.mapM (parseUInt width) b.data with
  | some ad, some bd =>
      match elemPairs outShape a.shape b.shape with
      | none => renderErr id "broadcast index error"
      | some ps =>
          match List.mapM (fun p => udivOut ad bd p) ps with
          | Except.ok qs => renderOk id outShape kname (qs.map toString)
          | Except.error msg => renderErr id msg
  | _, _ => renderMalformed id "bad unsigned integer data"

def runRat (id : String) (outShape : List Nat) (a b : TRaw) : String :=
  match List.mapM parseRat a.data, List.mapM parseRat b.data with
  | some ad, some bd =>
      match elemPairs outShape a.shape b.shape with
      | none => renderErr id "broadcast index error"
      | some ps =>
          match List.mapM (fun p => rdivOut ad bd p) ps with
          | Except.ok ds => renderOk id outShape "rat" ds
          | Except.error msg => renderErr id msg
  | _, _ => renderMalformed id "bad rational data"

/-! ## Dispatch -/

def canonKind (k : String) : String :=
  if k = "float16" then "f16"
  else if k = "half" then "f16"
  else if k = "float32" then "f32"
  else if k = "float" then "f32"
  else if k = "float64" then "f64"
  else if k = "double" then "f64"
  else if k = "int8" then "i8"
  else if k = "int16" then "i16"
  else if k = "int32" then "i32"
  else if k = "int64" then "i64"
  else if k = "uint8" then "u8"
  else if k = "uint16" then "u16"
  else if k = "uint32" then "u32"
  else if k = "uint64" then "u64"
  else k

def runDiv (id : String) (a b : TRaw) : String :=
  let ka := canonKind a.kind
  let kb := canonKind b.kind
  if ka = "f16" then renderUnsupported id "float16 is not representable"
  else if kb = "f16" then renderUnsupported id "float16 is not representable"
  else if ka ≠ kb then renderErr id "the two inputs have different kinds"
  else if shapeSize a.shape ≠ a.data.length then
    renderMalformed id "A data length does not match its shape"
  else if shapeSize b.shape ≠ b.data.length then
    renderMalformed id "B data length does not match its shape"
  else
    match bcastShape a.shape b.shape with
    | none => renderErr id "shapes are not broadcast-compatible"
    | some outShape =>
        if shapeSize outShape = 0 then renderOk id outShape ka []
        else if ka = "f64" then runFloat fmt64 "f64" id outShape a b
        else if ka = "f32" then runFloat fmt32 "f32" id outShape a b
        else if ka = "i8" then runSInt 8 "i8" id outShape a b
        else if ka = "i16" then runSInt 16 "i16" id outShape a b
        else if ka = "i32" then runSInt 32 "i32" id outShape a b
        else if ka = "i64" then runSInt 64 "i64" id outShape a b
        else if ka = "u8" then runUInt 8 "u8" id outShape a b
        else if ka = "u16" then runUInt 16 "u16" id outShape a b
        else if ka = "u32" then runUInt 32 "u32" id outShape a b
        else if ka = "u64" then runUInt 64 "u64" id outShape a b
        else if ka = "rat" then runRat id outShape a b
        else renderUnsupported id ("unsupported kind " ++ ka)

def processLine (line : String) : String :=
  let l := stripCR line
  match l.splitOn "\t" with
  | [] => renderMalformed "" "empty line"
  | id :: rest =>
      match rest with
      | _attrs :: arityStr :: tensors =>
          match parseNat? arityStr with
          | none => renderMalformed id "bad arity field"
          | some arity =>
              if arity ≠ tensors.length then
                renderMalformed id "arity does not match the number of tensor fields"
              else if arity ≠ 2 then
                renderMalformed id "Div expects exactly 2 inputs"
              else
                match List.mapM parseTensor tensors with
                | some [ta, tb] => runDiv id ta tb
                | some _ => renderMalformed id "expected exactly 2 tensors"
                | none => renderMalformed id "bad tensor field"
      | _ => renderMalformed id "expected: id, attrs, arity, tensors"

end DivRunner

/-- The runner's entry point, at the root namespace. -/
def main : IO Unit := do
  let stdin ← IO.getStdin
  let input ← stdin.readToEnd
  for line in DivRunner.splitLines input do
    IO.println (DivRunner.processLine line)
