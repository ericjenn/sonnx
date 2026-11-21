import Sonnx

namespace SubRunner

def pow2 (k : Nat) : Nat := 2 ^ k

def floorLog2 (n : Nat) : Nat :=
  let rec go (fuel : Nat) (n : Nat) (k : Nat) : Nat :=
    match fuel with
    | 0 => k
    | fuel + 1 =>
        if n ≤ 1 then k else go fuel (n / 2) (k + 1)
  go n n 0

def roundDivPow2 (n k : Nat) : Nat :=
  if k = 0 then n
  else
    let d := pow2 k
    let q := n / d
    let r := n % d
    let half := pow2 (k - 1)
    if r < half then q
    else if r > half then q + 1
    else if q % 2 = 0 then q else q + 1

inductive FVal
  | nan
  | inf (neg : Bool)
  | zero (neg : Bool)
  | finite (neg : Bool) (mant : Nat) (exp : Int)

def decodeFloat (expBits fracBits bias bits : Nat) : FVal :=
  let maxEF := pow2 expBits - 1
  let sign : Bool := decide ((bits / pow2 (expBits + fracBits)) % 2 = 1)
  let ef := (bits / pow2 fracBits) % pow2 expBits
  let fr := bits % pow2 fracBits
  if ef = maxEF then
    if fr = 0 then FVal.inf sign else FVal.nan
  else if ef = 0 then
    if fr = 0 then FVal.zero sign
    else FVal.finite sign fr ((1 : Int) - Int.ofNat bias - Int.ofNat fracBits)
  else
    FVal.finite sign (pow2 fracBits + fr)
      (Int.ofNat ef - Int.ofNat bias - Int.ofNat fracBits)

def signedMantExp : FVal -> Int × Int
  | FVal.zero _ => (0, 0)
  | FVal.finite neg m e => ((if neg then - (Int.ofNat m) else Int.ofNat m), e)
  | _ => (0, 0)

def intAbsNat (z : Int) : Nat :=
  if z < 0 then Int.toNat (-z) else Int.toNat z

def roundToBits (expBits fracBits bias : Nat) (neg : Bool) (m : Nat) (e : Int) : Nat :=
  let p := fracBits
  let signPos := expBits + fracBits
  let signMask := if neg then pow2 signPos else 0
  let maxExpField := pow2 expBits - 1
  let maxNormalExp : Int := Int.ofNat (pow2 expBits - 2) - Int.ofNat bias
  let minNormalExp : Int := (1 : Int) - Int.ofNat bias
  let minSubExp : Int := minNormalExp - Int.ofNat p
  if m = 0 then signMask
  else
    let k := floorLog2 m
    let E : Int := e + Int.ofNat k
    if E < minNormalExp then
      let shift : Int := e - minSubExp
      let q := if shift ≥ 0 then m * pow2 (Int.toNat shift) else roundDivPow2 m (Int.toNat (- shift))
      if q = 0 then signMask else signMask + q
    else
      let shift : Int := Int.ofNat p - Int.ofNat k
      let sig := if shift ≥ 0 then m * pow2 (Int.toNat shift) else roundDivPow2 m (Int.toNat (- shift))
      let overflowSig := sig = pow2 (p + 1)
      let sig' := if overflowSig then pow2 p else sig
      let E' := if overflowSig then E + 1 else E
      if E' > maxNormalExp then
        signMask + maxExpField * pow2 p
      else
        let expField : Int := E' + Int.ofNat bias
        let expFieldNat := Int.toNat expField
        signMask + expFieldNat * pow2 p + (sig' - pow2 p)

def floatSubBits (expBits fracBits bias : Nat) (a b : FVal) : Nat :=
  let p := fracBits
  let maxExpField := pow2 expBits - 1
  let signPos := expBits + fracBits
  let signMask : Bool -> Nat := fun neg => if neg then pow2 signPos else 0
  let infBits : Bool -> Nat := fun neg => signMask neg + maxExpField * pow2 p
  let nanBits := maxExpField * pow2 p + pow2 (p - 1)
  match a, b with
  | FVal.nan, _ => nanBits
  | _, FVal.nan => nanBits
  | FVal.inf sa, FVal.inf sb =>
      if sa = sb then nanBits else infBits sa
  | FVal.inf sa, _ => infBits sa
  | _, FVal.inf sb => infBits (Bool.not sb)
  | _, _ =>
      let (ma, ea) := signedMantExp a
      let (mb, eb) := signedMantExp b
      let e := if ea ≤ eb then ea else eb
      let ka : Nat := Int.toNat (ea - e)
      let kb : Nat := Int.toNat (eb - e)
      let d : Int := ma * Int.ofNat (pow2 ka) - mb * Int.ofNat (pow2 kb)
      if d = 0 then
        let negZero := match a, b with
          | FVal.zero true, FVal.zero false => true
          | _, _ => false
        signMask negZero
      else
        let neg : Bool := decide (d < 0)
        let m := intAbsNat d
        roundToBits expBits fracBits bias neg m e

def shapeProd : List Nat -> Nat
  | [] => 1
  | d :: ds => d * shapeProd ds

def padLeft (shape : List Nat) (r : Nat) : List Nat :=
  List.replicate (r - shape.length) 1 ++ shape

def broadcastDim (a b : Nat) : Option Nat :=
  if a = b then some a
  else if a = 1 then some b
  else if b = 1 then some a
  else none

def broadcastDims : List (Nat × Nat) -> Option (List Nat)
  | [] => some []
  | (a, b) :: rest =>
      match broadcastDim a b with
      | none => none
      | some d =>
          match broadcastDims rest with
          | none => none
          | some ds => some (d :: ds)

def broadcastShape (sa sb : List Nat) : Option (List Nat) :=
  let r := max sa.length sb.length
  broadcastDims (List.zip (padLeft sa r) (padLeft sb r))

def strides (shape : List Nat) : List Nat :=
  let rec go (dims : List Nat) (acc : Nat) : List Nat :=
    match dims with
    | [] => []
    | d :: ds => acc :: go ds (acc * d)
  (go shape.reverse 1).reverse

def coordsOf (shape strides : List Nat) (idx : Nat) : List Nat :=
  match shape, strides with
  | [], _ => []
  | d :: ds, s :: ss =>
      let c := if d = 0 then 0 else (idx / s) % d
      c :: coordsOf ds ss idx
  | _ :: ds, [] => 0 :: coordsOf ds [] idx

def flatIndex (paddedShape strides coords : List Nat) : Nat :=
  let rec go (sh : List Nat) (st : List Nat) (cs : List Nat) (acc : Nat) : Nat :=
    match sh, st, cs with
    | d :: ds, s :: ss, c :: cc =>
        let ci := if d = 1 then 0 else c
        go ds ss cc (acc + ci * s)
    | _, _, _ => acc
  go paddedShape strides coords 0

def listGet? {α : Type} (xs : List α) (i : Nat) : Option α :=
  match i, xs with
  | 0, x :: _ => some x
  | i + 1, _ :: rest => listGet? rest i
  | _, _ => none

def broadcastMap (sc sa sb : List Nat) (da db : List String)
    (f : String -> String -> Option String) : Option (List String) :=
  let total := shapeProd sc
  let outStrides := strides sc
  let padA := padLeft sa sc.length
  let padB := padLeft sb sc.length
  let stA := strides padA
  let stB := strides padB
  let rec loop (idxs : List Nat) (acc : List String) : Option (List String) :=
    match idxs with
    | [] => some acc.reverse
    | i :: rest =>
        let coords := coordsOf sc outStrides i
        let ai := flatIndex padA stA coords
        let bi := flatIndex padB stB coords
        match listGet? da ai, listGet? db bi with
        | some x, some y =>
            match f x y with
            | some r => loop rest (r :: acc)
            | none => none
        | _, _ => none
  loop (List.range total) []

def parseNatStr (s : String) : Option Nat :=
  let cs := s.toList
  if cs.isEmpty then none
  else
    let rec go (acc : Nat) (cs : List Char) : Option Nat :=
      match cs with
      | [] => some acc
      | c :: rest =>
          if ('0' ≤ c) ∧ (c ≤ '9') then
            go (acc * 10 + (c.toNat - '0'.toNat)) rest
          else none
    go 0 cs

def parseIntStr (s : String) : Option Int :=
  match s.toList with
  | '-' :: rest =>
      match parseNatStr (String.ofList rest) with
      | some n => some (- (Int.ofNat n))
      | none => none
  | _ =>
      match parseNatStr s with
      | some n => some (Int.ofNat n)
      | none => none

def parseShape (s : String) : Option (List Nat) :=
  if s = "-" then some []
  else
    let rec go (ps : List String) : Option (List Nat) :=
      match ps with
      | [] => some []
      | p :: rest =>
          match parseNatStr p with
          | none => none
          | some n =>
              match go rest with
              | none => none
              | some ns => some (n :: ns)
    go (s.splitOn ",")

def parseDataList (s : String) : Option (List String) :=
  if s = "-" then some []
  else if s = "" then some []
  else some (s.splitOn ",")

structure GenericTensor where
  kind : String
  shape : List Nat
  data : List String

def parseTensorField (s : String) : Option GenericTensor :=
  match s.splitOn "|" with
  | k :: sh :: dt :: [] =>
      match parseShape sh, parseDataList dt with
      | some shape, some data => some { kind := k, shape := shape, data := data }
      | _, _ => none
  | _ => none

def parseRat (s : String) : Option (Int × Nat) :=
  match s.splitOn "/" with
  | n :: d :: [] =>
      match parseIntStr n, parseNatStr d with
      | some ni, some di => if di = 0 then none else some (ni, di)
      | _, _ => none
  | _ => none

inductive CaseResult
  | ok (kind : String) (shape : List Nat) (data : List String)
  | unsupported (reason : String)
  | err (reason : String)
  | malformed (msg : String)

def isSupportedKind (s : String) : Bool :=
  s == "f32" || s == "f64" ||
  s == "u8" || s == "u16" || s == "u32" || s == "u64" ||
  s == "i8" || s == "i16" || s == "i32" || s == "i64" ||
  s == "rat"

def evalFloat (expBits fracBits bias : Nat) (ta tb : GenericTensor) : CaseResult :=
  match broadcastShape ta.shape tb.shape with
  | none => CaseResult.err "incompatible shapes"
  | some sc =>
      let totalBits := 1 + expBits + fracBits
      let maxBits := pow2 totalBits
      match broadcastMap sc ta.shape tb.shape ta.data tb.data (fun sa sb =>
          match parseNatStr sa, parseNatStr sb with
          | some ba, some bb =>
              if (ba < maxBits) ∧ (bb < maxBits) then
                let va := decodeFloat expBits fracBits bias ba
                let vb := decodeFloat expBits fracBits bias bb
                let cbits := floatSubBits expBits fracBits bias va vb
                some (toString cbits)
              else none
          | _, _ => none) with
      | some data => CaseResult.ok ta.kind sc data
      | none => CaseResult.malformed "invalid float data"

def evalInt (n : Nat) (ta tb : GenericTensor) : CaseResult :=
  match broadcastShape ta.shape tb.shape with
  | none => CaseResult.err "incompatible shapes"
  | some sc =>
      let modulus := pow2 n
      match broadcastMap sc ta.shape tb.shape ta.data tb.data (fun sa sb =>
          match parseNatStr sa, parseNatStr sb with
          | some ba, some bb =>
              if (ba < modulus) ∧ (bb < modulus) then
                let res := if bb ≤ ba then ba - bb else ba + modulus - bb
                some (toString res)
              else none
          | _, _ => none) with
      | some data => CaseResult.ok ta.kind sc data
      | none => CaseResult.malformed "invalid integer data"

def evalRat (ta tb : GenericTensor) : CaseResult :=
  match broadcastShape ta.shape tb.shape with
  | none => CaseResult.err "incompatible shapes"
  | some sc =>
      match broadcastMap sc ta.shape tb.shape ta.data tb.data (fun sa sb =>
          match parseRat sa, parseRat sb with
          | some (na, da), some (nb, db) =>
              let num := na * Int.ofNat db - nb * Int.ofNat da
              let den := da * db
              let res :=
                if num = 0 then "0/1"
                else
                  let g := Nat.gcd (intAbsNat num) den
                  let num' := num / Int.ofNat g
                  let den' := den / g
                  toString num' ++ "/" ++ toString den'
              some res
          | _, _ => none) with
      | some data => CaseResult.ok ta.kind sc data
      | none => CaseResult.malformed "invalid rational data"

def evalSub (ts : List GenericTensor) : CaseResult :=
  match ts with
  | [ta, tb] =>
      if isSupportedKind ta.kind = false then
        CaseResult.unsupported ("unsupported kind " ++ ta.kind)
      else if isSupportedKind tb.kind = false then
        CaseResult.unsupported ("unsupported kind " ++ tb.kind)
      else if ta.kind ≠ tb.kind then
        CaseResult.err "type mismatch"
      else if ta.data.length != shapeProd ta.shape || tb.data.length != shapeProd tb.shape then
        CaseResult.malformed "data length does not match shape"
      else
        match ta.kind with
        | "f32" => evalFloat 8 23 127 ta tb
        | "f64" => evalFloat 11 52 1023 ta tb
        | "u8" => evalInt 8 ta tb
        | "u16" => evalInt 16 ta tb
        | "u32" => evalInt 32 ta tb
        | "u64" => evalInt 64 ta tb
        | "i8" => evalInt 8 ta tb
        | "i16" => evalInt 16 ta tb
        | "i32" => evalInt 32 ta tb
        | "i64" => evalInt 64 ta tb
        | "rat" => evalRat ta tb
        | _ => CaseResult.unsupported ("unsupported kind " ++ ta.kind)
  | _ => CaseResult.err "Sub expects two tensors"

def parseCase (line : String) : CaseResult :=
  let fields := line.splitOn "\t"
  match fields with
  | _id :: _attrs :: arityStr :: rest =>
      match parseNatStr arityStr with
      | none => CaseResult.malformed "invalid arity"
      | some arity =>
          if rest.length ≠ arity then
            CaseResult.malformed "tensor count does not match arity"
          else
            let rec parseTensors (fs : List String) : Option (List GenericTensor) :=
              match fs with
              | [] => some []
              | f :: fr =>
                  match parseTensorField f with
                  | none => none
                  | some t =>
                      match parseTensors fr with
                      | none => none
                      | some ts => some (t :: ts)
            match parseTensors rest with
            | none => CaseResult.malformed "invalid tensor field"
            | some ts =>
                if arity ≠ 2 then CaseResult.err "Sub expects arity 2"
                else evalSub ts
  | _ => CaseResult.malformed "expected id, attrs, arity, tensors"

def joinCommaTail : List String -> String
  | [] => ""
  | y :: ys => "," ++ y ++ joinCommaTail ys

def joinComma (xs : List String) : String :=
  match xs with
  | [] => "-"
  | x :: xs => x ++ joinCommaTail xs

def formatShape (s : List Nat) : String :=
  joinComma (s.map (fun n => toString n))

def formatCase (id : String) (r : CaseResult) : String :=
  match r with
  | CaseResult.ok kind shape data =>
      id ++ "\tOK\t1\t" ++ kind ++ "|" ++ formatShape shape ++ "|" ++ joinComma data
  | CaseResult.unsupported reason =>
      id ++ "\tUNSUPPORTED\t0\t" ++ reason
  | CaseResult.err reason =>
      id ++ "\tERR\t0\t" ++ reason
  | CaseResult.malformed msg =>
      id ++ "\tMALFORMED\t0\t" ++ msg

def stripCr (s : String) : String :=
  match s.toList.reverse with
  | '\r' :: rest => String.ofList rest.reverse
  | _ => s

def splitLines (s : String) : List String :=
  let rec go (acc : List Char) (lines : List String) (cs : List Char) : List String :=
    match cs with
    | [] =>
        if acc.isEmpty then lines.reverse
        else (stripCr (String.ofList acc.reverse)) :: lines.reverse
    | c :: cs' =>
        if c = '\n' then
          go [] (stripCr (String.ofList acc.reverse) :: lines) cs'
        else
          go (c :: acc) lines cs'
  go [] [] s.toList

def processLine (line : String) : String :=
  let id := match line.splitOn "\t" with
    | [] => ""
    | f :: _ => f
  formatCase id (parseCase line)

end SubRunner

open SubRunner

def main : IO Unit := do
  let stdin ← IO.getStdin
  let input ← stdin.readToEnd
  let lines := splitLines input
  for line in lines do
    IO.println (processLine line)
