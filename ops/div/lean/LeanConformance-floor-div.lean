import Sonnx

namespace DivRunner

inductive Kind where
  | f16 | f32 | f64
  | u8 | u16 | u32 | u64
  | i8 | i16 | i32 | i64
  | rat
  deriving DecidableEq, Repr

structure FloatFormat where
  mantissaBits : Nat
  exponentBits : Nat
  bias : Nat

def f32Format : FloatFormat := { mantissaBits := 23, exponentBits := 8, bias := 127 }
def f64Format : FloatFormat := { mantissaBits := 52, exponentBits := 11, bias := 1023 }

inductive FVal where
  | nan
  | inf (neg : Bool)
  | zero (neg : Bool)
  | finite (neg : Bool) (mant : Nat) (exp : Int)

def decodeFloat (fmt : FloatFormat) (bits : Nat) : FVal :=
  let sign := (bits / 2^(fmt.mantissaBits + fmt.exponentBits)) % 2 == 1
  let expField := (bits / 2^fmt.mantissaBits) % 2^fmt.exponentBits
  let mantField := bits % 2^fmt.mantissaBits
  if expField == 2^fmt.exponentBits - 1 then
    if mantField == 0 then FVal.inf sign else FVal.nan
  else if expField == 0 then
    if mantField == 0 then FVal.zero sign
    else FVal.finite sign mantField (1 - (fmt.bias : Int) - (fmt.mantissaBits : Int))
  else
    let e := (expField : Int) - (fmt.bias : Int)
    let m := 2^fmt.mantissaBits + mantField
    FVal.finite sign m (e - (fmt.mantissaBits : Int))

def encodeSignExpMant (fmt : FloatFormat) (neg : Bool) (expField : Nat) (mant : Nat) : Nat :=
  let signBit := if neg then 1 else 0
  signBit * 2^(fmt.mantissaBits + fmt.exponentBits) + expField * 2^fmt.mantissaBits + mant

def nanBits (fmt : FloatFormat) : Nat :=
  encodeSignExpMant fmt false (2^fmt.exponentBits - 1) (2^(fmt.mantissaBits - 1))

def infBits (fmt : FloatFormat) (neg : Bool) : Nat :=
  encodeSignExpMant fmt neg (2^fmt.exponentBits - 1) 0

def zeroBits (fmt : FloatFormat) (neg : Bool) : Nat :=
  encodeSignExpMant fmt neg 0 0

def log2Floor (n : Nat) : Nat := Nat.log2 n

def roundDiv (num den : Nat) : Nat :=
  let q := num / den
  let r := num % den
  let r2 := r * 2
  if r2 < den then q
  else if r2 > den then q + 1
  else if q % 2 == 0 then q else q + 1

def ltPow2 (p q : Nat) (e : Int) : Bool :=
  if e >= 0 then
    p < q * 2^(e.natAbs)
  else
    p * 2^(e.natAbs) < q

def roundToBits (fmt : FloatFormat) (neg : Bool) (p q : Nat) : Nat :=
  let mb := fmt.mantissaBits
  let e_min : Int := 1 - (fmt.bias : Int)
  let e_max : Int := (fmt.bias : Int)
  let a := log2Floor p
  let b := log2Floor q
  let e0 : Int := (a : Int) - (b : Int)
  let e : Int := if ltPow2 p q e0 then e0 - 1 else e0
  if e < e_min then
    let shiftInt : Int := (mb : Int) - e_min
    let shift : Nat := shiftInt.toNat
    let num := p * 2^shift
    let k := roundDiv num q
    if k == 0 then zeroBits fmt neg
    else if k < 2^mb then encodeSignExpMant fmt neg 0 k
    else encodeSignExpMant fmt neg 1 0
  else
    let shiftInt : Int := (mb : Int) - e
    let (num, den) := if shiftInt >= 0 then
        (p * 2^(shiftInt.toNat), q)
      else
        (p, q * 2^((-shiftInt).toNat))
    let m := roundDiv num den
    let (m', e') := if m == 2^(mb + 1) then (2^mb, e + 1) else (m, e)
    if e' > e_max then
      infBits fmt neg
    else
      let expField : Nat := (e' + (fmt.bias : Int)).toNat
      let mantField : Nat := m' - 2^mb
      encodeSignExpMant fmt neg expField mantField

def divFVal (fmt : FloatFormat) (a b : FVal) : Nat :=
  match a, b with
  | FVal.nan, _ | _, FVal.nan => nanBits fmt
  | FVal.inf _, FVal.inf _ => nanBits fmt
  | FVal.zero _, FVal.zero _ => nanBits fmt
  | FVal.finite sa _ _, FVal.zero sb => infBits fmt (sa != sb)
  | FVal.zero sa, FVal.finite sb _ _ => zeroBits fmt (sa != sb)
  | FVal.inf sa, FVal.finite sb _ _ => infBits fmt (sa != sb)
  | FVal.finite sa _ _, FVal.inf sb => zeroBits fmt (sa != sb)
  | FVal.inf sa, FVal.zero sb => infBits fmt (sa != sb)
  | FVal.zero sa, FVal.inf sb => zeroBits fmt (sa != sb)
  | FVal.finite sa ma ea, FVal.finite sb mb eb =>
      let neg := sa != sb
      let d : Int := ea - eb
      let p := if d >= 0 then ma * 2^(d.toNat) else ma
      let q := if d >= 0 then mb else mb * 2^((-d).toNat)
      roundToBits fmt neg p q

def divIntBits (width : Nat) (signed : Bool) (aBits bBits : Nat) : Option Nat :=
  if bBits == 0 then none
  else
    if signed then
      let half := 2^(width - 1)
      let a : Int := if aBits >= half then (aBits : Int) - (2^width : Int) else (aBits : Int)
      let b : Int := if bBits >= half then (bBits : Int) - (2^width : Int) else (bBits : Int)
      let quot : Int := a / b
      let res : Int := if quot == (half : Int) then - (half : Int) else quot
      let bits : Nat := if res < 0 then (res + (2^width : Int)).toNat else res.toNat
      some bits
    else
      some (aBits / bBits)

def broadcastDim (a b : Nat) : Option Nat :=
  if a == b then some a
  else if a == 1 then some b
  else if b == 1 then some a
  else none

def broadcastShape (shapeA shapeB : List Nat) : Option (List Nat) :=
  let rank := max shapeA.length shapeB.length
  let pad (s : List Nat) : List Nat := List.replicate (rank - s.length) 1 ++ s
  let a := pad shapeA
  let b := pad shapeB
  let rec go (a b : List Nat) (acc : List Nat) : Option (List Nat) :=
    match a, b with
    | [], [] => some acc.reverse
    | da :: as, db :: bs =>
      match broadcastDim da db with
      | some d => go as bs (d :: acc)
      | none => none
    | _, _ => none
  go a b []

def prod (l : List Nat) : Nat := l.foldl (· * ·) 1

def flatToIndices (shape : List Nat) (i : Nat) : List Nat :=
  let rec go (shape : List Nat) (i : Nat) : List Nat :=
    match shape with
    | [] => []
    | d :: ds =>
      let sizeRest := prod ds
      let idx := if sizeRest == 0 then 0 else i / sizeRest
      let rem := if sizeRest == 0 then 0 else i % sizeRest
      idx :: go ds rem
  go shape i

def operandFlatIndex (opShape : List Nat) (outIndices : List Nat) : Nat :=
  let rec go (outIdx : List Nat) (opDims : List Nat) (stride : Nat) : Nat :=
    match outIdx, opDims with
    | [], _ => 0
    | _, [] => 0
    | idx :: outRest, d :: opRest =>
      let effectiveIdx := if d == 1 then 0 else idx
      effectiveIdx * stride + go outRest opRest (stride * d)
  go outIndices.reverse opShape.reverse 1

def hexDigit (c : Char) : Option Nat :=
  if c >= '0' && c <= '9' then some (c.toNat - '0'.toNat)
  else if c >= 'a' && c <= 'f' then some (c.toNat - 'a'.toNat + 10)
  else if c >= 'A' && c <= 'F' then some (c.toNat - 'A'.toNat + 10)
  else none

def parseHex (s : String) : Option Nat :=
  s.toList.foldl (fun acc c =>
    match acc, hexDigit c with
    | some n, some d => some (n * 16 + d)
    | _, _ => none) (some 0)

def parseBitPattern (s : String) : Option Nat :=
  if s.startsWith "0x" || s.startsWith "0X" then
    parseHex ((s.drop 2).toString)
  else
    s.toNat?

def parseRat (s : String) : Option Rat :=
  match s.splitOn "/" with
  | [numStr, denStr] =>
    match numStr.toInt?, denStr.toNat? with
    | some n, some d => if d == 0 then none else some ((n : Rat) / (d : Rat))
    | _, _ => none
  | _ => none

def parseShape (s : String) : Option (List Nat) :=
  if s == "-" then some []
  else
    let parts := s.splitOn ","
    let rec go (parts : List String) : Option (List Nat) :=
      match parts with
      | [] => some []
      | p :: ps =>
        match p.toNat? with
        | some n =>
          match go ps with
          | some ns => some (n :: ns)
          | none => none
        | none => none
    go parts

def parseTensorField (field : String) : Option (String × List Nat × List String) :=
  match field.splitOn "|" with
  | [kindStr, shapeStr, dataStr] =>
    let shapeOpt := parseShape shapeStr
    let dataTokens := if dataStr == "-" then [] else dataStr.splitOn ","
    match shapeOpt with
    | some shape => some (kindStr, shape, dataTokens)
    | none => none
  | _ => none

def parseKind (s : String) : Option Kind :=
  match s with
  | "f64" => some Kind.f64
  | "f32" => some Kind.f32
  | "f16" => some Kind.f16
  | "u8" => some Kind.u8
  | "u16" => some Kind.u16
  | "u32" => some Kind.u32
  | "u64" => some Kind.u64
  | "i8" => some Kind.i8
  | "i16" => some Kind.i16
  | "i32" => some Kind.i32
  | "i64" => some Kind.i64
  | "rat" => some Kind.rat
  | _ => none

def joinWith (sep : String) (l : List String) : String :=
  match l with
  | [] => ""
  | [x] => x
  | x :: xs => x ++ sep ++ joinWith sep xs

def formatShape (shape : List Nat) : String :=
  if shape.isEmpty then "-" else joinWith "," (shape.map toString)

def formatOk (id kindStr : String) (outShape : List Nat) (outData : String) : String :=
  s!"{id}\tOK\t1\t{kindStr}|{formatShape outShape}|{outData}"

def processInt (id kindStr : String) (width : Nat) (signed : Bool)
    (shapeA shapeB : List Nat) (dataA dataB : List String) : String :=
  let parseBits (tokens : List String) : Option (List Nat) :=
    let rec go (tokens : List String) (acc : List Nat) : Option (List Nat) :=
      match tokens with
      | [] => some acc.reverse
      | t :: ts =>
        match parseBitPattern t with
        | some n => go ts (n :: acc)
        | none => none
    go tokens []
  match parseBits dataA, parseBits dataB with
  | some aBitsList, some bBitsList =>
    let prodA := prod shapeA
    let prodB := prod shapeB
    if aBitsList.length != prodA || bBitsList.length != prodB then
      s!"{id}\tMALFORMED\t0\tdata length mismatch"
    else
      match broadcastShape shapeA shapeB with
      | none => s!"{id}\tERR\t0\tincompatible shapes"
      | some outShape =>
        let outSize := prod outShape
        if outSize == 0 then
          formatOk id kindStr outShape "-"
        else
          if bBitsList.any (· == 0) then
            s!"{id}\tERR\t0\tdivision by zero"
          else
            let arrA := aBitsList.toArray
            let arrB := bBitsList.toArray
            let rec loop (i : Nat) (acc : List String) : List String :=
              if i >= outSize then acc.reverse
              else
                let outIndices := flatToIndices outShape i
                let flatA := operandFlatIndex shapeA outIndices
                let flatB := operandFlatIndex shapeB outIndices
                let aBits := arrA[flatA]?.getD 0
                let bBits := arrB[flatB]?.getD 0
                match divIntBits width signed aBits bBits with
                | some resBits => loop (i+1) (toString resBits :: acc)
                | none => loop (i+1) ("0" :: acc)
            let outTokens := loop 0 []
            let outData := joinWith "," outTokens
            formatOk id kindStr outShape outData
  | _, _ => s!"{id}\tMALFORMED\t0\tinvalid integer data"

def processFloat (id kindStr : String) (fmt : FloatFormat)
    (shapeA shapeB : List Nat) (dataA dataB : List String) : String :=
  let parseBits (tokens : List String) : Option (List Nat) :=
    let rec go (tokens : List String) (acc : List Nat) : Option (List Nat) :=
      match tokens with
      | [] => some acc.reverse
      | t :: ts =>
        match parseBitPattern t with
        | some n => go ts (n :: acc)
        | none => none
    go tokens []
  match parseBits dataA, parseBits dataB with
  | some aBitsList, some bBitsList =>
    let prodA := prod shapeA
    let prodB := prod shapeB
    if aBitsList.length != prodA || bBitsList.length != prodB then
      s!"{id}\tMALFORMED\t0\tdata length mismatch"
    else
      match broadcastShape shapeA shapeB with
      | none => s!"{id}\tERR\t0\tincompatible shapes"
      | some outShape =>
        let outSize := prod outShape
        if outSize == 0 then
          formatOk id kindStr outShape "-"
        else
          let arrA := aBitsList.toArray
          let arrB := bBitsList.toArray
          let rec loop (i : Nat) (acc : List String) : List String :=
            if i >= outSize then acc.reverse
            else
              let outIndices := flatToIndices outShape i
              let flatA := operandFlatIndex shapeA outIndices
              let flatB := operandFlatIndex shapeB outIndices
              let aBits := arrA[flatA]?.getD 0
              let bBits := arrB[flatB]?.getD 0
              let aVal := decodeFloat fmt aBits
              let bVal := decodeFloat fmt bBits
              let resBits := divFVal fmt aVal bVal
              loop (i+1) (toString resBits :: acc)
          let outTokens := loop 0 []
          let outData := joinWith "," outTokens
          formatOk id kindStr outShape outData
  | _, _ => s!"{id}\tMALFORMED\t0\tinvalid float data"

def processRat (id kindStr : String)
    (shapeA shapeB : List Nat) (dataA dataB : List String) : String :=
  let parseRats (tokens : List String) : Option (List Rat) :=
    let rec go (tokens : List String) (acc : List Rat) : Option (List Rat) :=
      match tokens with
      | [] => some acc.reverse
      | t :: ts =>
        match parseRat t with
        | some r => go ts (r :: acc)
        | none => none
    go tokens []
  match parseRats dataA, parseRats dataB with
  | some aRats, some bRats =>
    let prodA := prod shapeA
    let prodB := prod shapeB
    if aRats.length != prodA || bRats.length != prodB then
      s!"{id}\tMALFORMED\t0\tdata length mismatch"
    else
      match broadcastShape shapeA shapeB with
      | none => s!"{id}\tERR\t0\tincompatible shapes"
      | some outShape =>
        let outSize := prod outShape
        if outSize == 0 then
          formatOk id kindStr outShape "-"
        else
          if bRats.any (fun r => r == 0) then
            s!"{id}\tERR\t0\tdivision by zero"
          else
            let arrA := aRats.toArray
            let arrB := bRats.toArray
            let rec loop (i : Nat) (acc : List String) : List String :=
              if i >= outSize then acc.reverse
              else
                let outIndices := flatToIndices outShape i
                let flatA := operandFlatIndex shapeA outIndices
                let flatB := operandFlatIndex shapeB outIndices
                let a := arrA[flatA]?.getD 0
                let b := arrB[flatB]?.getD 0
                let res := a / b
                let token := s!"{res.num}/{res.den}"
                loop (i+1) (token :: acc)
            let outTokens := loop 0 []
            let outData := joinWith "," outTokens
            formatOk id kindStr outShape outData
  | _, _ => s!"{id}\tMALFORMED\t0\tinvalid rational data"

def handleLine (line : String) : String :=
  let line := if line.endsWith "\r" then (line.dropEnd 1).toString else line
  let fields := line.splitOn "\t"
  match fields with
  | [] => "\tMALFORMED\t0\tempty line"
  | id :: _attrs :: arityStr :: rest =>
    match arityStr.toNat? with
    | some arity =>
      if arity != 2 then
        s!"{id}\tMALFORMED\t0\tarity must be 2"
      else
        if rest.length != 2 then
          s!"{id}\tMALFORMED\t0\texpected 2 tensors"
        else
          match rest with
          | t1 :: t2 :: _ =>
            match parseTensorField t1, parseTensorField t2 with
            | some (kindStr1, shape1, data1), some (kindStr2, shape2, data2) =>
              if kindStr1 != kindStr2 then
                s!"{id}\tMALFORMED\t0\ttype mismatch"
              else
                match parseKind kindStr1 with
                | none => s!"{id}\tMALFORMED\t0\tunknown kind {kindStr1}"
                | some kind =>
                  match kind with
                  | Kind.f16 => s!"{id}\tUNSUPPORTED\t0\tfloat16 not supported"
                  | Kind.f32 => processFloat id kindStr1 f32Format shape1 shape2 data1 data2
                  | Kind.f64 => processFloat id kindStr1 f64Format shape1 shape2 data1 data2
                  | Kind.i8 => processInt id kindStr1 8 true shape1 shape2 data1 data2
                  | Kind.i16 => processInt id kindStr1 16 true shape1 shape2 data1 data2
                  | Kind.i32 => processInt id kindStr1 32 true shape1 shape2 data1 data2
                  | Kind.i64 => processInt id kindStr1 64 true shape1 shape2 data1 data2
                  | Kind.u8 => processInt id kindStr1 8 false shape1 shape2 data1 data2
                  | Kind.u16 => processInt id kindStr1 16 false shape1 shape2 data1 data2
                  | Kind.u32 => processInt id kindStr1 32 false shape1 shape2 data1 data2
                  | Kind.u64 => processInt id kindStr1 64 false shape1 shape2 data1 data2
                  | Kind.rat => processRat id kindStr1 shape1 shape2 data1 data2
            | _, _ => s!"{id}\tMALFORMED\t0\tinvalid tensor field"
          | _ => s!"{id}\tMALFORMED\t0\texpected 2 tensors"
    | none => s!"{id}\tMALFORMED\t0\tinvalid arity"
  | id :: _ => s!"{id}\tMALFORMED\t0\tmissing fields"

end DivRunner

open DivRunner

def main : IO Unit := do
  let stdin ← IO.getStdin
  let input ← stdin.readToEnd
  let lines := (input.splitOn "\n").map (fun s => if s.endsWith "\r" then (s.dropEnd 1).toString else s)
  let lines := match lines.reverse with
    | "" :: rest => rest.reverse
    | _ => lines
  for line in lines do
    IO.println (handleLine line)
