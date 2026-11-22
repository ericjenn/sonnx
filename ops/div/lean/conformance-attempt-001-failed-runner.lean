import Sonnx

namespace DivRunner

def joinSep (sep : String) : List String → String
  | [] => ""
  | [x] => x
  | x :: xs => x ++ sep ++ joinSep sep xs

def dropLast {α : Type} : List α → List α
  | [] => []
  | [_] => []
  | x :: xs => x :: dropLast xs

def endsWithNewline (s : String) : Bool :=
  match s.toList.reverse with
  | c :: _ => c == '\n'
  | [] => false

def splitLines (s : String) : List String :=
  if s.isEmpty then []
  else
    let parts := s.splitOn "\n"
    if endsWithNewline s then dropLast parts else parts

def parseNat? (s : String) : Option Nat :=
  if s.isEmpty then none else s.toNat?

def parseInt? (s : String) : Option Int :=
  if s.isEmpty then none else s.toInt?

def parseShape (s : String) : Option (List Nat) :=
  if s = "-" then some []
  else if s.isEmpty then none
  else (s.splitOn ",").mapM parseNat?

def parseData (s : String) : List String :=
  if s = "-" then []
  else if s.isEmpty then []
  else s.splitOn ","

structure TensorRaw where
  kind : String
  shape : List Nat
  data : List String
deriving Repr

def parseTensor (s : String) : Option TensorRaw :=
  match s.splitOn "|" with
  | [k, sh, da] =>
      if k.isEmpty then none
      else
        match parseShape sh with
        | none => none
        | some shape => some { kind := k, shape := shape, data := parseData da }
  | _ => none

def shapeSize (s : List Nat) : Nat :=
  s.foldl (fun acc d => acc * d) 1

def padTo (n : Nat) (s : List Nat) : List Nat :=
  List.replicate (n - s.length) 1 ++ s

def broadcastDim (a b : Nat) : Option Nat :=
  if a = b then some a
  else if a = 1 then some b
  else if b = 1 then some a
  else none

def broadcastShape (sa sb : List Nat) : Option (List Nat) :=
  let n := max sa.length sb.length
  let a := padTo n sa
  let b := padTo n sb
  (a.zip b).mapM (fun p => broadcastDim p.1 p.2)

def opFlatIndex (outShape opShape : List Nat) (idx : Nat) : Option Nat :=
  let rec go (outRev opRev : List Nat) (rem stride acc : Nat) : Option Nat :=
    match outRev with
    | [] => some acc
    | od :: ods =>
        if od = 0 then none
        else
          let coord := rem % od
          let rem' := rem / od
          match opRev with
          | [] => go ods [] rem' stride acc
          | pd :: pds =>
              if pd = 0 then none
              else
                let c := if pd = 1 then 0 else coord
                if c >= pd then none
                else go ods pds rem' (stride * pd) (acc + c * stride)
  go outShape.reverse opShape.reverse idx 1 0

def canonicalKind (k : String) : String :=
  if k = "float16" then "f16"
  else if k = "float32" then "f32"
  else if k = "float64" then "f64"
  else if k = "double" then "f64"
  else if k = "half" then "f16"
  else k

def parseF64 (s : String) : Option Float := do
  let n ← parseNat? s
  if n < 2 ^ 64 then
    pure (Float.ofBits (UInt64.ofNat n))
  else
    none

def parseF32 (s : String) : Option Float32 := do
  let n ← parseNat? s
  if n < 2 ^ 32 then
    pure (Float32.ofBits (UInt32.ofNat n))
  else
    none

def parseSInt (width : Nat) (s : String) : Option Int := do
  let bits ← parseNat? s
  let m := 2 ^ width
  if bits < m then
    let half := 2 ^ (width - 1)
    if bits < half then pure (Int.ofNat bits)
    else pure (Int.ofNat bits - Int.ofNat m)
  else
    none

def parseUInt (width : Nat) (s : String) : Option Nat := do
  let bits ← parseNat? s
  if bits < 2 ^ width then pure bits else none

def parseRat (s : String) : Option Rat := do
  match s.splitOn "/" with
  | [ns, ds] =>
      let n ← parseInt? ns
      let d ← parseNat? ds
      if d = 0 then none else pure (Rat.mk n d)
  | _ => none

def signedBits (width : Nat) (q : Int) : Nat :=
  let m : Int := Int.ofNat (2 ^ width)
  if q >= 0 then q.toNat
  else (q + m).toNat

def formatShape (s : List Nat) : String :=
  if s.isEmpty then "-" else joinSep "," (s.map toString)

def formatData (xs : List String) : String :=
  if xs.isEmpty then "-" else joinSep "," xs

def renderOk (id : String) (shape : List Nat) (kind : String) (data : List String) : String :=
  id ++ "\tOK\t1\t" ++ kind ++ "|" ++ formatShape shape ++ "|" ++ formatData data

def renderUnsupported (id : String) (reason : String) : String :=
  id ++ "\tUNSUPPORTED\t0\t" ++ reason

def renderErr (id : String) (reason : String) : String :=
  id ++ "\tERR\t0\t" ++ reason

def renderMalformed (id : String) (msg : String) : String :=
  id ++ "\tMALFORMED\t0\t" ++ msg

def runF64 (id : String) (outShape : List Nat) (a b : TensorRaw) : String :=
  match a.data.mapM parseF64, b.data.mapM parseF64 with
  | some ad, some bd =>
      match (List.range (shapeSize outShape)).mapM (fun idx => do
        let ia ← opFlatIndex outShape a.shape idx
        let ib ← opFlatIndex outShape b.shape idx
        let va ← List.get? ad ia
        let vb ← List.get? bd ib
        let c := va / vb
        pure (Float.toBits c).toNat) with
      | some bits => renderOk id outShape "f64" (bits.map toString)
      | none => renderErr id "broadcast index error"
  | _, _ => renderMalformed id "bad f64 data"

def runF32 (id : String) (outShape : List Nat) (a b : TensorRaw) : String :=
  match a.data.mapM parseF32, b.data.mapM parseF32 with
  | some ad, some bd =>
      match (List.range (shapeSize outShape)).mapM (fun idx => do
        let ia ← opFlatIndex outShape a.shape idx
        let ib ← opFlatIndex outShape b.shape idx
        let va ← List.get? ad ia
        let vb ← List.get? bd ib
        let c := va / vb
        pure (Float32.toBits c).toNat) with
      | some bits => renderOk id outShape "f32" (bits.map toString)
      | none => renderErr id "broadcast index error"
  | _, _ => renderMalformed id "bad f32 data"

def runSInt (id : String) (width : Nat) (kind : String)
    (outShape : List Nat) (a b : TensorRaw) : String :=
  match a.data.mapM (parseSInt width), b.data.mapM (parseSInt width) with
  | some ad, some bd =>
      let result : Except String (List Nat) :=
        (List.range (shapeSize outShape)).mapM (fun idx => do
          let ia ← match opFlatIndex outShape a.shape idx with
                   | some i => pure i
                   | none => throw "broadcast index error"
          let ib ← match opFlatIndex outShape b.shape idx with
                   | some i => pure i
                   | none => throw "broadcast index error"
          let va ← match List.get? ad ia with
                   | some v => pure v
                   | none => throw "data index error"
          let vb ← match List.get? bd ib with
                   | some v => pure v
                   | none => throw "data index error"
          if vb = 0 then
            throw "division by zero"
          else
            pure (signedBits width (Int.tdiv va vb)))
      match result with
      | Except.ok bits => renderOk id outShape kind (bits.map toString)
      | Except.error msg => renderErr id msg
  | _, _ => renderMalformed id "bad signed integer data"

def runUInt (id : String) (width : Nat) (kind : String)
    (outShape : List Nat) (a b : TensorRaw) : String :=
  match a.data.mapM (parseUInt width), b.data.mapM (parseUInt width) with
  | some ad, some bd =>
      let result : Except String (List Nat) :=
        (List.range (shapeSize outShape)).mapM (fun idx => do
          let ia ← match opFlatIndex outShape a.shape idx with
                   | some i => pure i
                   | none => throw "broadcast index error"
          let ib ← match opFlatIndex outShape b.shape idx with
                   | some i => pure i
                   | none => throw "broadcast index error"
          let va ← match List.get? ad ia with
                   | some v => pure v
                   | none => throw "data index error"
          let vb ← match List.get? bd ib with
                   | some v => pure v
                   | none => throw "data index error"
          if vb = 0 then
            throw "division by zero"
          else
            pure (va / vb))
      match result with
      | Except.ok bits => renderOk id outShape kind (bits.map toString)
      | Except.error msg => renderErr id msg
  | _, _ => renderMalformed id "bad unsigned integer data"

def runRat (id : String) (outShape : List Nat) (a b : TensorRaw) : String :=
  match a.data.mapM parseRat, b.data.mapM parseRat with
  | some ad, some bd =>
      let result : Except String (List String) :=
        (List.range (shapeSize outShape)).mapM (fun idx => do
          let ia ← match opFlatIndex outShape a.shape idx with
                   | some i => pure i
                   | none => throw "broadcast index error"
          let ib ← match opFlatIndex outShape b.shape idx with
                   | some i => pure i
                   | none => throw "broadcast index error"
          let va ← match List.get? ad ia with
                   | some v => pure v
                   | none => throw "data index error"
          let vb ← match List.get? bd ib with
                   | some v => pure v
                   | none => throw "data index error"
          if vb = 0 then
            throw "division by zero"
          else
            let c := va / vb
            pure (toString c.num ++ "/" ++ toString c.den))
      match result with
      | Except.ok data => renderOk id outShape "rat" data
      | Except.error msg => renderErr id msg
  | _, _ => renderMalformed id "bad rational data"

def runDiv (id : String) (a b : TensorRaw) : String :=
  let ka := canonicalKind a.kind
  let kb := canonicalKind b.kind
  if ka = "f16" then
    renderUnsupported id "float16 unsupported"
  else if kb = "f16" then
    renderUnsupported id "float16 unsupported"
  else if ka = kb then
    match broadcastShape a.shape b.shape with
    | none => renderErr id "incompatible shapes"
    | some outShape =>
        if shapeSize a.shape = a.data.length then
          if shapeSize b.shape = b.data.length then
            if shapeSize outShape = 0 then
              renderOk id outShape ka []
            else if ka = "f64" then
              runF64 id outShape a b
            else if ka = "f32" then
              runF32 id outShape a b
            else if ka = "i8" then
              runSInt id 8 ka outShape a b
            else if ka = "i16" then
              runSInt id 16 ka outShape a b
            else if ka = "i32" then
              runSInt id 32 ka outShape a b
            else if ka = "i64" then
              runSInt id 64 ka outShape a b
            else if ka = "u8" then
              runUInt id 8 ka outShape a b
            else if ka = "u16" then
              runUInt id 16 ka outShape a b
            else if ka = "u32" then
              runUInt id 32 ka outShape a b
            else if ka = "u64" then
              runUInt id 64 ka outShape a b
            else if ka = "rat" then
              runRat id outShape a b
            else
              renderUnsupported id ("unsupported kind " ++ ka)
          else
            renderMalformed id "B data length does not match shape"
        else
          renderMalformed id "A data length does not match shape"
  else
    renderErr id "type mismatch"

def processLine (line : String) : String :=
  let fields := line.splitOn "\t"
  let id := match fields with
    | [] => ""
    | x :: _ => x
  match fields with
  | _id :: _attrs :: arityStr :: tensors =>
      match parseNat? arityStr with
      | none => renderMalformed id "bad arity"
      | some arity =>
          if arity = tensors.length then
            if arity = 2 then
              match tensors.mapM parseTensor with
              | some [a, b] => runDiv id a b
              | some _ => renderMalformed id "expected 2 tensors"
              | none => renderMalformed id "bad tensor field"
            else
              renderMalformed id "arity must be 2"
          else
            renderMalformed id "arity does not match tensor count"
  | _ => renderMalformed id "expected id, attrs, arity and tensors"

def main : IO Unit := do
  let input ← IO.FS.readFile "/dev/stdin"
  let lines := splitLines input
  for line in lines do
    IO.println (processLine line)

end DivRunner
