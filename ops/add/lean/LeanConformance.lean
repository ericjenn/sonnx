import Sonnx

namespace AddRunner

/-! ## String utilities -/

def splitChar (sep : Char) (s : String) : List String :=
  let step (st : List String × String) (c : Char) : List String × String :=
    if c = sep then (st.2 :: st.1, "") else (st.1, st.2.push c)
  let r := s.toList.foldl step ([], "")
  (r.2 :: r.1).reverse

def joinWith (sep : String) (xs : List String) : String :=
  match xs with
  | [] => ""
  | [x] => x
  | x :: rest => x ++ sep ++ joinWith sep rest

def dropTrailingEmpty (xs : List String) : List String :=
  match xs.reverse with
  | "" :: rest => rest.reverse
  | _ => xs

def splitLines (s : String) : List String :=
  if s = "" then [] else dropTrailingEmpty (splitChar '\n' s)

def trimCR (s : String) : String :=
  match s.toList.reverse with
  | '\r' :: rest => String.mk rest.reverse
  | _ => s

/-! ## Parsing elementary values -/

def hexDigit (c : Char) : Option Nat :=
  if c = '0' then some 0
  else if c = '1' then some 1
  else if c = '2' then some 2
  else if c = '3' then some 3
  else if c = '4' then some 4
  else if c = '5' then some 5
  else if c = '6' then some 6
  else if c = '7' then some 7
  else if c = '8' then some 8
  else if c = '9' then some 9
  else if c = 'a' || c = 'A' then some 10
  else if c = 'b' || c = 'B' then some 11
  else if c = 'c' || c = 'C' then some 12
  else if c = 'd' || c = 'D' then some 13
  else if c = 'e' || c = 'E' then some 14
  else if c = 'f' || c = 'F' then some 15
  else none

def parseHex (cs : List Char) : Option Nat :=
  cs.foldl
    (fun (acc : Option Nat) (c : Char) =>
      match acc, hexDigit c with
      | some n, some v => some (n * 16 + v)
      | _, _ => none)
    (some 0)

def parseNat (s : String) : Option Nat :=
  match s.toList with
  | '0' :: 'x' :: rest => parseHex rest
  | '0' :: 'X' :: rest => parseHex rest
  | _ => String.toNat? s

def parseBits (width : Nat) (s : String) : Option Nat :=
  match parseNat s with
  | some n => some (n % (2 ^ width))
  | none =>
      match String.toInt? s with
      | some z =>
          if z >= 0 then
            some (Int.toNat z % (2 ^ width))
          else
            let m := Int.natAbs z % (2 ^ width)
            some (if m = 0 then 0 else (2 ^ width) - m)
      | none => none

def parseNatList (parts : List String) : Option (List Nat) :=
  parts.foldr
    (fun p acc =>
      match parseNat p, acc with
      | some n, some xs => some (n :: xs)
      | _, _ => none)
    (some [])

def parseShape (s : String) : Option (List Nat) :=
  if s = "-" then some [] else parseNatList (splitChar ',' s)

def dataTokens (s : String) : List String :=
  if s = "-" then [] else splitChar ',' s

def normalizeRat (n d : Int) : Option (Int × Int) :=
  if d = 0 then none
  else
    let n1 := if d < 0 then -n else n
    let d1 := if d < 0 then -d else d
    let g := Nat.gcd (Int.natAbs n1) (Int.natAbs d1)
    if g = 0 then some (0, 1)
    else
      let gi := Int.ofNat g
      some (n1 / gi, d1 / gi)

def parseRat (s : String) : Option (Int × Int) :=
  match splitChar '/' s with
  | [n, d] =>
      match String.toInt? n, String.toInt? d with
      | some nn, some dd => normalizeRat nn dd
      | _, _ => none
  | [n] =>
      match String.toInt? n with
      | some nn => normalizeRat nn 1
      | none => none
  | _ => none

def parseNatData (width : Nat) (tokens : List String) : Option (List Nat) :=
  tokens.foldr
    (fun t acc =>
      match parseBits width t, acc with
      | some n, some xs => some (n :: xs)
      | _, _ => none)
    (some [])

def parseRatData (tokens : List String) : Option (List (Int × Int)) :=
  tokens.foldr
    (fun t acc =>
      match parseRat t, acc with
      | some r, some xs => some (r :: xs)
      | _, _ => none)
    (some [])

/-! ## Kind recognition -/

inductive TK where
  | f32
  | f64
  | uint (bits : Nat)
  | sint (bits : Nat)
  | rat
  | f16
  | unknown
  deriving DecidableEq

def parseKind (s : String) : TK :=
  if s = "f32" || s = "float32" then .f32
  else if s = "f64" || s = "float64" || s = "double" then .f64
  else if s = "f16" || s = "float16" || s = "half" then .f16
  else if s = "u8" || s = "uint8" then .uint 8
  else if s = "u16" || s = "uint16" then .uint 16
  else if s = "u32" || s = "uint32" then .uint 32
  else if s = "u64" || s = "uint64" then .uint 64
  else if s = "i8" || s = "int8" then .sint 8
  else if s = "i16" || s = "int16" then .sint 16
  else if s = "i32" || s = "int32" then .sint 32
  else if s = "i64" || s = "int64" then .sint 64
  else if s = "rat" then .rat
  else .unknown

def isF16 : TK → Bool
  | .f16 => true
  | _ => false

def isUnknown : TK → Bool
  | .unknown => true
  | _ => false

def sameKind : TK → TK → Bool
  | .f32, .f32 => true
  | .f64, .f64 => true
  | .rat, .rat => true
  | .uint a, .uint b => a = b
  | .sint a, .sint b => a = b
  | _, _ => false

/-! ## Shapes, broadcasting and indexing -/

def prod (l : List Nat) : Nat :=
  l.foldl (fun a b => a * b) 1

def strides (shape : List Nat) : List Nat :=
  match shape with
  | [] => []
  | [_] => [1]
  | _ :: rest => prod rest :: strides rest

def coords (shape : List Nat) (linear : Nat) : List Nat :=
  let st := strides shape
  List.zipWith (fun d s => (linear / s) % d) shape st

def flatIndexBroadcast (outShape opShape : List Nat) (linear : Nat) : Nat :=
  let outCoords := coords outShape linear
  let aligned := List.drop (outShape.length - opShape.length) outCoords
  let strs := strides opShape
  let rec go (shape coords strs : List Nat) : Nat :=
    match shape, coords, strs with
    | d :: ds, c :: cs, s :: ss => (if d = 1 then 0 else c) * s + go ds cs ss
    | _, _, _ => 0
  go opShape aligned strs

def combineDim (a b : Nat) : Option Nat :=
  if a = b then some a
  else if a = 1 then some b
  else if b = 1 then some a
  else none

def broadcastShape (sa sb : List Nat) : Option (List Nat) :=
  let r := if sa.length >= sb.length then sa.length else sb.length
  let pad (s : List Nat) (n : Nat) : List Nat := List.replicate (n - s.length) 1 ++ s
  let a := pad sa r
  let b := pad sb r
  let rec go (a b : List Nat) : Option (List Nat) :=
    match a, b with
    | [], [] => some []
    | da :: as, db :: bs =>
        match combineDim da db, go as bs with
        | some d, some rest => some (d :: rest)
        | _, _ => none
    | _, _ => none
  go a b

def nthOpt : List α → Nat → Option α
  | [], _ => none
  | x :: _, 0 => some x
  | _ :: xs, n + 1 => nthOpt xs n

def getNat (l : List Nat) (i : Nat) : Nat :=
  match nthOpt l i with
  | some x => x
  | none => 0

def getRat (l : List (Int × Int)) (i : Nat) : Int × Int :=
  match nthOpt l i with
  | some x => x
  | none => (0, 1)

/-! ## Elementwise computation -/

def computeIntData
    (width : Nat) (shapeA shapeB outShape : List Nat)
    (dataA dataB : List Nat) : List Nat :=
  (List.range (prod outShape)).map (fun idx =>
    let ia := flatIndexBroadcast outShape shapeA idx
    let ib := flatIndexBroadcast outShape shapeB idx
    (getNat dataA ia + getNat dataB ib) % (2 ^ width))

def addRat (r1 r2 : Int × Int) : Int × Int :=
  let n1 := r1.1
  let d1 := r1.2
  let n2 := r2.1
  let d2 := r2.2
  match normalizeRat (n1 * d2 + n2 * d1) (d1 * d2) with
  | some r => r
  | none => (0, 1)

def computeRatData
    (shapeA shapeB outShape : List Nat)
    (dataA dataB : List (Int × Int)) : List (Int × Int) :=
  (List.range (prod outShape)).map (fun idx =>
    let ia := flatIndexBroadcast outShape shapeA idx
    let ib := flatIndexBroadcast outShape shapeB idx
    addRat (getRat dataA ia) (getRat dataB ib))

/-! ## IEEE 754 addition over exact Nat significands -/

structure Fmt where
  width : Nat
  expBits : Nat
  fracBits : Nat
  bias : Nat

def fmt32 : Fmt := { width := 32, expBits := 8, fracBits := 23, bias := 127 }
def fmt64 : Fmt := { width := 64, expBits := 11, fracBits := 52, bias := 1023 }

inductive MyFVal where
  | zero (neg : Bool)
  | inf (neg : Bool)
  | nan
  | finite (neg : Bool) (sig : Nat) (e : Int)

def mkBits (fmt : Fmt) (sign : Bool) (expField frac : Nat) : Nat :=
  (if sign then 1 else 0) * (2 ^ (fmt.width - 1)) +
    expField * (2 ^ fmt.fracBits) + frac

def zeroBits (fmt : Fmt) (sign : Bool) : Nat := mkBits fmt sign 0 0

def infBits (fmt : Fmt) (sign : Bool) : Nat :=
  mkBits fmt sign (2 ^ fmt.expBits - 1) 0

def nanBits (fmt : Fmt) : Nat :=
  mkBits fmt false (2 ^ fmt.expBits - 1) (2 ^ (fmt.fracBits - 1))

def decodeFloat (fmt : Fmt) (bits : Nat) : MyFVal :=
  let sign := (bits / (2 ^ (fmt.width - 1))) % 2 = 1
  let ef := (bits / (2 ^ fmt.fracBits)) % (2 ^ fmt.expBits)
  let fr := bits % (2 ^ fmt.fracBits)
  if ef = 2 ^ fmt.expBits - 1 then
    (if fr = 0 then MyFVal.inf sign else MyFVal.nan)
  else if ef = 0 then
    (if fr = 0 then MyFVal.zero sign
     else MyFVal.finite sign fr
       ((1 : Int) - Int.ofNat fmt.bias - Int.ofNat fmt.fracBits))
  else
    MyFVal.finite sign (2 ^ fmt.fracBits + fr)
      (Int.ofNat ef - Int.ofNat fmt.bias - Int.ofNat fmt.fracBits)

def bitLen (n : Nat) : Nat :=
  if n = 0 then 0 else Nat.log2 n + 1

def roundShiftRight (m shift : Nat) : Nat :=
  if shift = 0 then m
  else
    let d := 2 ^ shift
    let q := m / d
    let r := m % d
    let half := 2 ^ (shift - 1)
    if r < half then q
    else if r > half then q + 1
    else if q % 2 = 1 then q + 1 else q

def roundFinite (fmt : Fmt) (sign : Bool) (m : Nat) (e : Int) : Nat :=
  let p := fmt.fracBits + 1
  let b := bitLen m
  let k := Int.ofNat (b - 1) + e
  let emin := (1 : Int) - Int.ofNat fmt.bias
  let maxK := Int.ofNat ((2 ^ fmt.expBits - 2) - fmt.bias)
  if k < emin then
    let qexp := emin - Int.ofNat fmt.fracBits
    let t := e - qexp
    let s :=
      if t >= 0 then m * (2 ^ t.toNat) else roundShiftRight m (-t).toNat
    let minNormalFrac := 2 ^ fmt.fracBits
    if s = 0 then zeroBits fmt sign
    else if s >= minNormalFrac then mkBits fmt sign 1 0
    else mkBits fmt sign 0 s
  else if k > maxK then
    infBits fmt sign
  else
    let s0 := if b > p then roundShiftRight m (b - p) else m * (2 ^ (p - b))
    let k2 := if s0 >= 2 ^ p then k + 1 else k
    let s := if s0 >= 2 ^ p then s0 / 2 else s0
    if k2 > maxK then
      infBits fmt sign
    else
      let ef := (k2 + Int.ofNat fmt.bias).toNat
      let frac := s - 2 ^ fmt.fracBits
      mkBits fmt sign ef frac

def addFloat (fmt : Fmt) (bitsA bitsB : Nat) : Nat :=
  match decodeFloat fmt bitsA, decodeFloat fmt bitsB with
  | MyFVal.nan, _ => nanBits fmt
  | _, MyFVal.nan => nanBits fmt
  | MyFVal.inf s1, MyFVal.inf s2 =>
      if s1 = s2 then infBits fmt s1 else nanBits fmt
  | MyFVal.inf s, _ => infBits fmt s
  | _, MyFVal.inf s => infBits fmt s
  | MyFVal.zero s1, MyFVal.zero s2 => zeroBits fmt (s1 && s2)
  | MyFVal.zero _, _ => bitsB
  | MyFVal.finite _ _ _, MyFVal.zero _ => bitsA
  | MyFVal.finite s1 m1 e1, MyFVal.finite s2 m2 e2 =>
      let eMin := if e1 <= e2 then e1 else e2
      let sh1 := (e1 - eMin).toNat
      let sh2 := (e2 - eMin).toNat
      let a1 := m1 * (2 ^ sh1)
      let a2 := m2 * (2 ^ sh2)
      let sgn :=
        if s1 = s2 then s1
        else if a1 > a2 then s1
        else if a2 > a1 then s2
        else false
      let mag :=
        if s1 = s2 then a1 + a2
        else if a1 > a2 then a1 - a2
        else a2 - a1
      if mag = 0 then zeroBits fmt false else roundFinite fmt sgn mag eMin

def computeFloatData
    (fmt : Fmt) (shapeA shapeB outShape : List Nat)
    (dataA dataB : List Nat) : List Nat :=
  (List.range (prod outShape)).map (fun idx =>
    let ia := flatIndexBroadcast outShape shapeA idx
    let ib := flatIndexBroadcast outShape shapeB idx
    addFloat fmt (getNat dataA ia) (getNat dataB ib))

/-! ## Formatting and responses -/

def formatShape (shape : List Nat) : String :=
  match shape with
  | [] => "-"
  | _ => joinWith "," (shape.map (fun n => toString n))

def formatBits (data : List Nat) : String :=
  match data with
  | [] => "-"
  | _ => joinWith "," (data.map (fun n => toString n))

def formatRat (r : Int × Int) : String :=
  toString r.1 ++ "/" ++ toString r.2

def formatRats (data : List (Int × Int)) : String :=
  match data with
  | [] => "-"
  | _ => joinWith "," (data.map formatRat)

def respond (id status msg : String) : String :=
  id ++ "\t" ++ status ++ "\t0\t" ++ msg

def okLine (id kind : String) (shape : List Nat) (dataStr : String) : String :=
  id ++ "\tOK\t1\t" ++ kind ++ "|" ++ formatShape shape ++ "|" ++ dataStr

structure TensorHeader where
  kind : String
  shape : String
  data : String

def parseTensorHeader (s : String) : Option TensorHeader :=
  match splitChar '|' s with
  | [k, sh, d] => some { kind := k, shape := sh, data := d }
  | _ => none

/-! ## Per-kind finishing -/

def finishFloat
    (id kindName : String) (fmt : Fmt) (shapeA shapeB : List Nat)
    (tokensA tokensB : List String) : String :=
  match parseNatData fmt.width tokensA, parseNatData fmt.width tokensB with
  | some dataA, some dataB =>
      if dataA.length = prod shapeA && dataB.length = prod shapeB then
        match broadcastShape shapeA shapeB with
        | some outShape =>
            okLine id kindName outShape
              (formatBits (computeFloatData fmt shapeA shapeB outShape dataA dataB))
        | none => respond id "ERR" "incompatible shapes"
      else respond id "ERR" "data length"
  | _, _ => respond id "MALFORMED" "bad float data"

def finishInt
    (id kindName : String) (width : Nat) (shapeA shapeB : List Nat)
    (tokensA tokensB : List String) : String :=
  match parseNatData width tokensA, parseNatData width tokensB with
  | some dataA, some dataB =>
      if dataA.length = prod shapeA && dataB.length = prod shapeB then
        match broadcastShape shapeA shapeB with
        | some outShape =>
            okLine id kindName outShape
              (formatBits (computeIntData width shapeA shapeB outShape dataA dataB))
        | none => respond id "ERR" "incompatible shapes"
      else respond id "ERR" "data length"
  | _, _ => respond id "MALFORMED" "bad integer data"

def finishRat
    (id kindName : String) (shapeA shapeB : List Nat)
    (tokensA tokensB : List String) : String :=
  match parseRatData tokensA, parseRatData tokensB with
  | some dataA, some dataB =>
      if dataA.length = prod shapeA && dataB.length = prod shapeB then
        match broadcastShape shapeA shapeB with
        | some outShape =>
            okLine id kindName outShape
              (formatRats (computeRatData shapeA shapeB outShape dataA dataB))
        | none => respond id "ERR" "incompatible shapes"
      else respond id "ERR" "data length"
  | _, _ => respond id "MALFORMED" "bad rational data"

def dispatch
    (id : String) (k : TK) (sA sB : List Nat)
    (tA tB : List String) : String :=
  match k with
  | .f64 => finishFloat id "f64" fmt64 sA sB tA tB
  | .f32 => finishFloat id "f32" fmt32 sA sB tA tB
  | .uint w => finishInt id ("u" ++ toString w) w sA sB tA tB
  | .sint w => finishInt id ("i" ++ toString w) w sA sB tA tB
  | .rat => finishRat id "rat" sA sB tA tB
  | _ => respond id "MALFORMED" "unsupported kind token"

def process2 (id ta tb : String) : String :=
  match parseTensorHeader ta, parseTensorHeader tb with
  | some hA, some hB =>
      let kA := parseKind hA.kind
      let kB := parseKind hB.kind
      if isF16 kA || isF16 kB then
        respond id "UNSUPPORTED" "float16 is not representable"
      else if isUnknown kA || isUnknown kB then
        respond id "MALFORMED" "unknown kind"
      else if sameKind kA kB then
        match parseShape hA.shape, parseShape hB.shape with
        | some sA, some sB =>
            dispatch id kA sA sB (dataTokens hA.data) (dataTokens hB.data)
        | _, _ => respond id "MALFORMED" "bad shape"
      else respond id "ERR" "type mismatch"
  | _, _ => respond id "MALFORMED" "bad tensor field"

def processLine (rawLine : String) : String :=
  let line := trimCR rawLine
  match splitChar '\t' line with
  | id :: _attrs :: arity :: ta :: tb :: [] =>
      if arity = "2" then process2 id ta tb
      else respond id "ERR" "arity must be 2"
  | id :: _ => respond id "MALFORMED" "expected 5 fields"
  | [] => respond "" "MALFORMED" "empty line"

/-! ## I/O -/

partial def readAllBytes (s : IO.FS.Stream) (acc : ByteArray) : IO ByteArray := do
  let chunk ← s.read 65536
  if chunk.size == 0 then pure acc else readAllBytes s (acc ++ chunk)

def run : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  let bytes ← readAllBytes stdin ByteArray.empty
  let input :=
    match String.fromUTF8? bytes with
    | some s => s
    | none => ""
  for line in splitLines input do
    stdout.putStrLn (processLine line)
  stdout.flush

end AddRunner

def main : IO Unit := AddRunner.run
