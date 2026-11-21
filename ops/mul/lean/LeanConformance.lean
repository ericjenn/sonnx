import Sonnx

namespace MulConformance

structure TensorField where
  kind : String
  shape : List Nat
  data : List String

structure Case where
  id : String
  attrs : String
  arity : Nat
  tensors : List TensorField

structure QRat where
  num : Int
  den : Int

inductive ElemKind where
  | f64 | f32 | u8 | u16 | u32 | u64 | i8 | i16 | i32 | i64 | rat
  deriving BEq

def parseKind (s : String) : Option ElemKind :=
  match s with
  | "f64" => some ElemKind.f64
  | "f32" => some ElemKind.f32
  | "u8"  => some ElemKind.u8
  | "u16" => some ElemKind.u16
  | "u32" => some ElemKind.u32
  | "u64" => some ElemKind.u64
  | "i8"  => some ElemKind.i8
  | "i16" => some ElemKind.i16
  | "i32" => some ElemKind.i32
  | "i64" => some ElemKind.i64
  | "rat" => some ElemKind.rat
  | _     => none

def kindWidth : ElemKind -> Nat
  | ElemKind.f64 => 64
  | ElemKind.f32 => 32
  | ElemKind.u8  => 8
  | ElemKind.u16 => 16
  | ElemKind.u32 => 32
  | ElemKind.u64 => 64
  | ElemKind.i8  => 8
  | ElemKind.i16 => 16
  | ElemKind.i32 => 32
  | ElemKind.i64 => 64
  | ElemKind.rat => 0

def digitVal? (c : Char) : Option Nat :=
  if c >= '0' && c <= '9' then
    some (c.toNat - '0'.toNat)
  else if c >= 'a' && c <= 'f' then
    some (c.toNat - 'a'.toNat + 10)
  else if c >= 'A' && c <= 'F' then
    some (c.toNat - 'A'.toNat + 10)
  else
    none

def parseHexList (cs : List Char) (acc : Nat) : Option Nat :=
  match cs with
  | [] => some acc
  | c :: rest =>
      match digitVal? c with
      | some d => parseHexList rest (acc * 16 + d)
      | none => none

def parseDecList (cs : List Char) (acc : Nat) : Option Nat :=
  match cs with
  | [] => some acc
  | c :: rest =>
      if c >= '0' && c <= '9' then
        parseDecList rest (acc * 10 + (c.toNat - '0'.toNat))
      else
        none

def isSpaceChar (c : Char) : Bool :=
  c == ' ' || c == '\t' || c == '\n' || c == '\r'

def dropWhileSpace : List Char -> List Char
  | [] => []
  | c :: cs => if isSpaceChar c then dropWhileSpace cs else c :: cs

def trimStr (s : String) : String :=
  let cs := dropWhileSpace s.toList
  let cs := (dropWhileSpace cs.reverse).reverse
  String.ofList cs

def parseNatToken (s : String) : Option Nat :=
  let s := trimStr s
  if s.isEmpty then
    none
  else if s.startsWith "0x" || s.startsWith "0X" then
    let cs := s.toList.drop 2
    if cs.isEmpty then none else parseHexList cs 0
  else if s.toList.any (fun c => (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) then
    parseHexList s.toList 0
  else
    parseDecList s.toList 0

def parseIntToken (s : String) : Option Int :=
  let s := trimStr s
  if s.isEmpty then
    none
  else if s.startsWith "-" then
    let rest := String.ofList (s.toList.drop 1)
    match parseNatToken rest with
    | some n => some (-(Int.ofNat n))
    | none => none
  else
    match parseNatToken s with
    | some n => some (Int.ofNat n)
    | none => none

def parseShape (s : String) : Option (List Nat) :=
  if s == "-" then
    some []
  else
    (s.splitOn ",").mapM parseNatToken

def parseTensor (s : String) : Option TensorField :=
  match s.splitOn "|" with
  | [k, sh, da] =>
      match parseShape sh with
      | some shape =>
          let data := if da == "-" then [] else da.splitOn ","
          some { kind := k, shape := shape, data := data }
      | none => none
  | _ => none

def parseCase (line : String) : Option Case :=
  match line.splitOn "\t" with
  | id :: attrs :: arityStr :: rest =>
      match parseNatToken arityStr with
      | some arity =>
          if rest.length == arity then
            match rest.mapM parseTensor with
            | some tensors =>
                some { id := id, attrs := attrs, arity := arity, tensors := tensors }
            | none => none
          else
            none
      | none => none
  | _ => none

def lineId (line : String) : String :=
  match line.splitOn "\t" with
  | id :: _ => id
  | [] => ""

def gcdNat (a b : Nat) : Nat :=
  let rec go : Nat -> Nat -> Nat -> Nat
    | 0, x, _ => x
    | Nat.succ fuel, x, y =>
        if y == 0 then x
        else go fuel y (x % y)
  go (a + b + 1) a b

def natAbsInt (n : Int) : Nat :=
  if n < 0 then (-n).toNat else n.toNat

def normalizeRat (n d : Int) : Option QRat :=
  if d == 0 then
    none
  else
    let n' := if d < 0 then -n else n
    let d' := if d < 0 then -d else d
    let g := gcdNat (natAbsInt n') d'.toNat
    if g == 0 then
      some { num := 0, den := 1 }
    else
      let gi := Int.ofNat g
      some { num := n' / gi, den := d' / gi }

def parseRat (s : String) : Option QRat :=
  match s.splitOn "/" with
  | [n] =>
      match parseIntToken n with
      | some ni => normalizeRat ni 1
      | none => none
  | [n, d] =>
      match parseIntToken n, parseIntToken d with
      | some ni, some di => normalizeRat ni di
      | _, _ => none
  | _ => none

def ratMul (a b : QRat) : QRat :=
  match normalizeRat (a.num * b.num) (a.den * b.den) with
  | some r => r
  | none => { num := 0, den := 1 }

def highestBit (n : Nat) : Nat :=
  let rec go (n k : Nat) : Nat :=
    if n <= 1 then k
    else go (n / 2) (k + 1)
  go n 0

def roundShift (m shift : Nat) : Nat :=
  if shift == 0 then
    m
  else
    let q := m / (2 ^ shift)
    let r := m % (2 ^ shift)
    let half := 2 ^ (shift - 1)
    if r < half then q
    else if r > half then q + 1
    else if q % 2 == 0 then q else q + 1

def f32ToF64Bits (b : Nat) : Nat :=
  let s := (b / 0x80000000) % 2
  let e := (b / 0x800000) % 256
  let m := b % 0x800000
  let sign64 := if s == 1 then 0x8000000000000000 else 0
  if e == 255 then
    if m == 0 then sign64 + 0x7ff0000000000000
    else 0x7ff8000000000000
  else if e == 0 then
    if m == 0 then sign64
    else
      let k := highestBit m
      let exp64 := k + 874
      let mant64 := (m - 2 ^ k) * (2 ^ (52 - k))
      sign64 + exp64 * 0x0010000000000000 + mant64
  else
    let exp64 := e + 896
    sign64 + exp64 * 0x0010000000000000 + m * (2 ^ 29)

def f64BitsToF32 (b : Nat) : Nat :=
  let s := (b / 0x8000000000000000) % 2
  let e64 := (b / 0x0010000000000000) % 2048
  let m64 := b % 0x0010000000000000
  let sign32 := if s == 1 then 0x80000000 else 0
  if e64 == 2047 then
    if m64 == 0 then sign32 + 0x7f800000
    else 0x7fc00000
  else if e64 == 0 then
    sign32
  else
    let E : Int := Int.ofNat e64 - 1023
    let M := 0x0010000000000000 + m64
    if E >= 128 then
      sign32 + 0x7f800000
    else if E >= -126 then
      let q := roundShift M 29
      if q >= 0x01000000 then
        let E2 := E + 1
        if E2 > 127 then
          sign32 + 0x7f800000
        else
          let e32 := (E2 + 127).toNat
          sign32 + e32 * 0x00800000
      else
        let e32 := (E + 127).toNat
        let m32 := q - 0x00800000
        sign32 + e32 * 0x00800000 + m32
    else
      let shift := (-E - 97).toNat
      let q := roundShift M shift
      if q >= 0x00800000 then
        sign32 + 0x00800000
      else
        sign32 + q

def isNaN64 (b : Nat) : Bool :=
  let exp := (b / 0x0010000000000000) % 2048
  let mant := b % 0x0010000000000000
  exp == 2047 && mant != 0

def isNaN32 (b : Nat) : Bool :=
  let exp := (b / 0x800000) % 256
  let mant := b % 0x800000
  exp == 255 && mant != 0

def f64BitsMul (a b : Nat) : Nat :=
  let fa := Float.ofBits (UInt64.ofNat a)
  let fb := Float.ofBits (UInt64.ofNat b)
  let r := fa * fb
  let bits := (Float.toBits r).toNat
  if isNaN64 bits then 0x7ff8000000000000 else bits

def f32BitsMul (a b : Nat) : Nat :=
  let fa := Float.ofBits (UInt64.ofNat (f32ToF64Bits a))
  let fb := Float.ofBits (UInt64.ofNat (f32ToF64Bits b))
  let r := fa * fb
  let bits64 := (Float.toBits r).toNat
  let out := f64BitsToF32 bits64
  if isNaN32 out then 0x7fc00000 else out

def elemMulNat (k : ElemKind) (a b : Nat) : Nat :=
  match k with
  | ElemKind.f64 => f64BitsMul a b
  | ElemKind.f32 => f32BitsMul a b
  | ElemKind.rat => 0
  | _ => (a * b) % (2 ^ kindWidth k)

def listProd (xs : List Nat) : Nat :=
  xs.foldl (fun acc x => acc * x) 1

def mergeDim (a b : Nat) : Option Nat :=
  if a == b then
    some a
  else if a == 1 then
    some b
  else if b == 1 then
    some a
  else
    none

def broadcastShape (sa sb : List Nat) : Option (List Nat) :=
  let rec go (ra rb : List Nat) (acc : List Nat) : Option (List Nat) :=
    match ra, rb with
    | [], [] => some acc
    | a :: as, [] =>
        let d := if a == 1 then 1 else a
        go as [] (d :: acc)
    | [], b :: bs =>
        let d := if b == 1 then 1 else b
        go [] bs (d :: acc)
    | a :: as, b :: bs =>
        match mergeDim a b with
        | some d => go as bs (d :: acc)
        | none => none
  go sa.reverse sb.reverse []

def positions (shape : List Nat) (flat : Nat) : List Nat :=
  match shape with
  | [] => []
  | d :: ds =>
      let rest := listProd ds
      let p := if rest == 0 || d == 0 then 0 else (flat / rest) % d
      let rem := if rest == 0 then 0 else flat % rest
      p :: positions ds rem

def inputFlat (inShape outShape : List Nat) (outFlat : Nat) : Nat :=
  let outPosRev := (positions outShape outFlat).reverse
  let inShapeRev := inShape.reverse
  let rec go (inDims outPos : List Nat) (stride acc : Nat) : Nat :=
    match inDims with
    | [] => acc
    | d :: ds =>
        let p :=
          match outPos with
          | [] => 0
          | q :: _ => if d == 1 then 0 else q
        let outPos' :=
          match outPos with
          | [] => []
          | _ :: qs => qs
        go ds outPos' (stride * d) (acc + p * stride)
  go inShapeRev outPosRev 1 0

def getNat (xs : List Nat) (i : Nat) : Nat :=
  match xs[i]? with
  | some v => v
  | none => 0

def getQRat (xs : List QRat) (i : Nat) : QRat :=
  match xs[i]? with
  | some v => v
  | none => { num := 0, den := 1 }

def joinWith (sep : String) (xs : List String) : String :=
  match xs with
  | [] => ""
  | [x] => x
  | x :: rest => x ++ sep ++ joinWith sep rest

def shapeToStr (s : List Nat) : String :=
  if s.isEmpty then "-" else joinWith "," (s.map toString)

def natListToStr (xs : List Nat) : String :=
  if xs.isEmpty then "-" else joinWith "," (xs.map toString)

def ratToStr (r : QRat) : String :=
  toString r.num ++ "/" ++ toString r.den

def ratListToStr (xs : List QRat) : String :=
  if xs.isEmpty then "-" else joinWith "," (xs.map ratToStr)

def parseNatList (width : Nat) (xs : List String) : Option (List Nat) :=
  xs.mapM (fun s =>
    match parseNatToken s with
    | some v => if v < 2 ^ width then some v else none
    | none => none)

def parseRatList (xs : List String) : Option (List QRat) :=
  xs.mapM parseRat

def malformedResp (id msg : String) : String :=
  id ++ "\tMALFORMED\t0\t" ++ msg

def errResp (id msg : String) : String :=
  id ++ "\tERR\t0\t" ++ msg

def unsupportedResp (id msg : String) : String :=
  id ++ "\tUNSUPPORTED\t0\t" ++ msg

def okNat (id kind shape data : String) : String :=
  id ++ "\tOK\t1\t" ++ kind ++ "|" ++ shape ++ "|" ++ data

def processNat (id : String) (k : ElemKind) (a b : TensorField) : String :=
  let width := kindWidth k
  match parseNatList width a.data, parseNatList width b.data with
  | some da, some db =>
      if da.length != listProd a.shape || db.length != listProd b.shape then
        malformedResp id "data length mismatch"
      else
        match broadcastShape a.shape b.shape with
        | none => errResp id "incompatible shapes"
        | some outShape =>
            let outSize := listProd outShape
            let outData := (List.range outSize).map (fun i =>
              let ia := inputFlat a.shape outShape i
              let ib := inputFlat b.shape outShape i
              elemMulNat k (getNat da ia) (getNat db ib))
            okNat id a.kind (shapeToStr outShape) (natListToStr outData)
  | _, _ => malformedResp id "bad data token"

def processRat (id : String) (a b : TensorField) : String :=
  match parseRatList a.data, parseRatList b.data with
  | some da, some db =>
      if da.length != listProd a.shape || db.length != listProd b.shape then
        malformedResp id "data length mismatch"
      else
        match broadcastShape a.shape b.shape with
        | none => errResp id "incompatible shapes"
        | some outShape =>
            let outSize := listProd outShape
            let outData := (List.range outSize).map (fun i =>
              let ia := inputFlat a.shape outShape i
              let ib := inputFlat b.shape outShape i
              ratMul (getQRat da ia) (getQRat db ib))
            okNat id a.kind (shapeToStr outShape) (ratListToStr outData)
  | _, _ => malformedResp id "bad rational token"

def processCase (c : Case) : String :=
  match c.tensors with
  | [a, b] =>
      match parseKind a.kind, parseKind b.kind with
      | some ka, some kb =>
          if ka != kb then
            errResp c.id "type mismatch"
          else if ka == ElemKind.rat then
            processRat c.id a b
          else
            processNat c.id ka a b
      | none, _ =>
          unsupportedResp c.id ("unsupported kind " ++ a.kind)
      | _, none =>
          unsupportedResp c.id ("unsupported kind " ++ b.kind)
  | _ => errResp c.id "unsupported arity"

def processLine (line : String) : String :=
  let id := lineId line
  match parseCase line with
  | none => malformedResp id "malformed line"
  | some c => processCase c

def stripCR (s : String) : String :=
  if s.endsWith "\r" then String.ofList (s.toList.dropLast) else s

def run : IO Unit := do
  let stdin <- IO.getStdin
  let input <- stdin.readToEnd
  let rawLines := input.splitOn "\n"
  let lines :=
    if input.isEmpty then []
    else if input.endsWith "\n" then rawLines.dropLast
    else rawLines
  for line in lines do
    IO.println (processLine (stripCR line))

end MulConformance

def main : IO Unit := MulConformance.run
