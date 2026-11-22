/-
The C code generator — a pretty-printer for the C11 AST.

`renderExpr` is precedence-aware: every expression has a precedence, and a
child is parenthesized exactly when its precedence is below the minimum its
position allows.  The table (C11 6.5) is

| construct                                | precedence |
|------------------------------------------|------------|
| `,`                                      | 1          |
| `=`, compound assignment (right-assoc)   | 2          |
| `?:` (right-assoc)                       | 3          |
| `||`                                     | 4          |
| `&&`                                     | 5          |
| `|`                                      | 6          |
| `^`                                      | 7          |
| `&`                                      | 8          |
| `==`, `!=`                               | 9          |
| `<`, `<=`, `>`, `>=`                     | 10         |
| `<<`, `>>`                               | 11         |
| `+`, `-`                                 | 12         |
| `*`, `/`, `%`                            | 13         |
| unary prefix, casts                      | 14         |
| postfix `[] . -> () ++ --`               | 15         |
| primary                                  | 16         |

Two details keep the output lexically safe: a prefix operator renders its
operand at 15 (so `-(-x)` and `&(&x)`, never `--x` or `&&x`), and a negative
integer literal renders in parentheses (so `(-1) + x`).  Control-flow bodies
are always braced, so a dangling `else` cannot occur.
-/

import CGen.Ast

namespace CGen

/-! ### Types -/

/-- The type specifier of a type, without a declarator. -/
def baseName : CType → String
  | .void => "void"
  | .bool => "_Bool"
  | .int bits true => "int" ++ toString bits ++ "_t"
  | .int bits false => "uint" ++ toString bits ++ "_t"
  | .float 32 => "float"
  | .float 64 => "double"
  | .float bits => "float" ++ toString bits
  | .size => "size_t"
  | .named name => name
  | .ptr _ => "void *"
  | .const base => "const " ++ baseName base

/-- A declaration of `name` with type `t`, placing the declarator where C
requires it: `declType (ptr (const (float 32))) "a"` is `const float *a`. -/
def declType : CType → String → String
  | .ptr pointee, name => declType pointee ("*" ++ name)
  | .const base, name => "const " ++ declType base name
  | t, name => if name.isEmpty then baseName t else baseName t ++ " " ++ name

/-- A type used in a cast or a type name: `declType t ""`. -/
def renderType (t : CType) : String := declType t ""

/-! ### Expressions -/

private def precComma : Nat := 1
private def precAssign : Nat := 2
private def precTernary : Nat := 3
private def precLogOr : Nat := 4
private def precLogAnd : Nat := 5
private def precBitOr : Nat := 6
private def precBitXor : Nat := 7
private def precBitAnd : Nat := 8
private def precEq : Nat := 9
private def precRel : Nat := 10
private def precShift : Nat := 11
private def precAdd : Nat := 12
private def precMul : Nat := 13
private def precUnary : Nat := 14
private def precPostfix : Nat := 15

/-- The precedence of a binary operator (assignment and comma included). -/
def binOpPrec : BinOp → Nat
  | .comma => precComma
  | .assign => precAssign
  | .logOr => precLogOr
  | .logAnd => precLogAnd
  | .bitOr => precBitOr
  | .bitXor => precBitXor
  | .bitAnd => precBitAnd
  | .eq | .ne => precEq
  | .lt | .le | .gt | .ge => precRel
  | .shl | .shr => precShift
  | .add | .sub => precAdd
  | .mul | .div | .mod => precMul

private def binOpSym : BinOp → String
  | .add => "+" | .sub => "-" | .mul => "*" | .div => "/" | .mod => "%"
  | .shl => "<<" | .shr => ">>"
  | .bitAnd => "&" | .bitOr => "|" | .bitXor => "^"
  | .lt => "<" | .le => "<=" | .gt => ">" | .ge => ">="
  | .eq => "==" | .ne => "!="
  | .logAnd => "&&" | .logOr => "||"
  | .assign => "=" | .comma => ","

private def unOpSym : UnOp → String
  | .neg => "-" | .logNot => "!" | .bitNot => "~"
  | .deref => "*" | .addr => "&"
  | .preInc => "++" | .preDec => "--"
  | .postInc => "" | .postDec => ""

/-- Parenthesize `s` when its precedence is below the minimum. -/
private def paren (need : Bool) (s : String) : String :=
  if need then "(" ++ s ++ ")" else s

/-- Render an expression in a context whose minimum precedence is `minPrec`.
`renderExpr 0` renders a full expression. -/
def renderExpr (minPrec : Nat) : Expr → String
  | .lit text => text
  | .intLit value => if value < 0 then "(" ++ toString value ++ ")" else toString value
  | .var name => name
  | .index base offset =>
      paren (precPostfix < minPrec)
        (renderExpr precPostfix base ++ "[" ++ renderExpr precComma offset ++ "]")
  | .field base name arrow =>
      paren (precPostfix < minPrec)
        (renderExpr precPostfix base ++ (if arrow then "->" else ".") ++ name)
  | .unary .postInc arg =>
      paren (precPostfix < minPrec) (renderExpr precPostfix arg ++ "++")
  | .unary .postDec arg =>
      paren (precPostfix < minPrec) (renderExpr precPostfix arg ++ "--")
  | .unary op arg =>
      paren (precUnary < minPrec) (unOpSym op ++ renderExpr precPostfix arg)
  | .binary .comma lhs rhs =>
      paren (precComma < minPrec)
        (renderExpr precComma lhs ++ ", " ++ renderExpr precAssign rhs)
  | .binary .assign lhs rhs =>
      paren (precAssign < minPrec)
        (renderExpr precUnary lhs ++ " = " ++ renderExpr precAssign rhs)
  | .binary op lhs rhs =>
      let p := binOpPrec op
      paren (p < minPrec)
        (renderExpr p lhs ++ " " ++ binOpSym op ++ " " ++ renderExpr (p + 1) rhs)
  | .cast t arg =>
      paren (precUnary < minPrec)
        ("(" ++ renderType t ++ ")" ++ renderExpr precPostfix arg)
  | .call callee args =>
      paren (precPostfix < minPrec)
        (callee ++ "(" ++ ", ".intercalate (args.map (renderExpr precAssign)) ++ ")")
  | .ternary cond thenB elseB =>
      paren (precTernary < minPrec)
        (renderExpr precLogOr cond ++ " ? " ++ renderExpr precAssign thenB ++
          " : " ++ renderExpr precTernary elseB)
  | .sizeofType t =>
      paren (precUnary < minPrec) ("sizeof(" ++ renderType t ++ ")")

/-! ### Statements -/

/-- Two spaces per level. -/
def indent (level : Nat) : String := String.join (List.replicate level "  ")

private def escapeComment (text : String) : String := text.replace "*/" "* /"

/-- The initializer suffix of a declaration: `""` or `" = <expr>"`. -/
def renderInit : Option Expr → String
  | none => ""
  | some e => " = " ++ renderExpr precAssign e

mutual
  /-- Render a statement at an indentation level, without a trailing newline.
  Multi-line statements embed their own newlines and indentation. -/
  partial def renderStmt (level : Nat) (s : Stmt) : String :=
    match s with
    | .expr e => indent level ++ renderExpr 0 e ++ ";"
    | .decl t name init => indent level ++ declType t name ++ renderInit init ++ ";"
    | .ifThen cond thenB elseB =>
        indent level ++ "if (" ++ renderExpr 0 cond ++ ") " ++ renderBody level thenB ++
          (match elseB with
           | none => ""
           | some e => " else " ++ renderBody level e)
    | .forLoop init cond step body =>
        indent level ++ "for (" ++ renderForClause init ++ "; " ++
          (match cond with | none => "" | some e => renderExpr 0 e) ++ "; " ++
          (match step with | none => "" | some e => renderExpr 0 e) ++ ") " ++
          renderBody level body
    | .whileLoop cond body =>
        indent level ++ "while (" ++ renderExpr 0 cond ++ ") " ++ renderBody level body
    | .ret value =>
        indent level ++ "return" ++
          (match value with | none => "" | some e => " " ++ renderExpr 0 e) ++ ";"
    | .block stmts =>
        if stmts.isEmpty then indent level ++ "{}"
        else indent level ++ "{\n" ++ renderStmts (level + 1) stmts ++ "\n" ++ indent level ++ "}"
    | .comment text => indent level ++ "/* " ++ escapeComment text ++ " */"
    | .nop => indent level ++ ";"

  /-- Render a list of statements, one per line. -/
  partial def renderStmts (level : Nat) (stmts : List Stmt) : String :=
    String.intercalate "\n" (stmts.map (renderStmt level))

  /-- Render a controlled statement as a braced body.  Braces are always
  emitted, so an `else` can never bind to an inner `if`. -/
  partial def renderBody (level : Nat) (s : Stmt) : String :=
    match s with
    | .block stmts =>
        if stmts.isEmpty then "{}"
        else "{\n" ++ renderStmts (level + 1) stmts ++ "\n" ++ indent level ++ "}"
    | _ => "{\n" ++ renderStmt (level + 1) s ++ "\n" ++ indent level ++ "}"

  /-- A `for` header clause: a declaration or an expression, without its
  terminating semicolon. -/
  partial def renderForClause : Option Stmt → String
    | none => ""
    | some (.decl t name init) => declType t name ++ renderInit init
    | some (.expr e) => renderExpr 0 e
    | some s => renderStmt 0 s
end

/-! ### Functions and translation units -/

/-- The parameter list, `(void)` when there are no parameters. -/
def renderParams : List Param → String
  | [] => "void"
  | params => ", ".intercalate (params.map (fun p => declType p.type p.name))

/-- A function's name, return type and parameters, without a body. -/
def renderFunHeader (fn : Fun) : String :=
  declType fn.retType fn.name ++ "(" ++ renderParams fn.params ++ ")"

/-- A function definition, without a trailing newline. -/
def renderFun (fn : Fun) : String :=
  renderFunHeader fn ++ " " ++ renderBody 0 (.block fn.body)

/-- The derivation comments above a function's prototype and definition. -/
private def renderFunComment : List String → String
  | [] => ""
  | lines => String.join (lines.map (fun l => "/* " ++ escapeComment l ++ " */\n"))

/-- A top-level declaration, without a trailing newline. -/
def renderDecl : Decl → String
  | .typedef t name => "typedef " ++ declType t name ++ ";"
  | .structDef name fields =>
      "typedef struct {\n" ++
        String.join (fields.map (fun f => indent 1 ++ declType f.type f.name ++ ";\n")) ++
        "} " ++ name ++ ";"
  | .funDecl fn => renderFunComment fn.comment ++ renderFunHeader fn ++ ";"
  | .funDef fn => renderFunComment fn.comment ++ renderFun fn
  | .comment text => "/* " ++ escapeComment text ++ " */"

/-- The separator between two declarations: prototypes are kept on adjacent
lines, everything else is separated by a blank line. -/
private def declSeparator : Decl → Decl → String
  | .funDecl _, .funDecl _ => "\n"
  | _, _ => "\n\n"

private partial def joinDecls : List Decl → String
  | [] => ""
  | [d] => renderDecl d
  | d₁ :: d₂ :: rest => renderDecl d₁ ++ declSeparator d₁ d₂ ++ joinDecls (d₂ :: rest)

/-- The banner comment block.  Empty lines stay empty. -/
def renderBanner (lines : List String) : String :=
  match lines with
  | [] => ""
  | _ =>
      "/*\n" ++
        String.join (lines.map (fun l => if l.isEmpty then " *\n" else " * " ++ l ++ "\n")) ++
        " */\n"

/-- The includes, one per line.  A name that is already delimited — `<stddef.h>`,
`"sonnx_kernels.h"` — is rendered as it stands; a bare name is a system header
and is wrapped in angle brackets. -/
def renderIncludes (headers : List String) : String :=
  String.join (headers.map (fun h =>
    if h.startsWith "<" || h.startsWith "\"" then "#include " ++ h ++ "\n"
    else "#include <" ++ h ++ ">\n"))

/-- A translation unit: banner, includes, then the declarations, ending in a
newline. -/
def renderFile (f : CFile) : String :=
  let header := renderBanner f.banner ++ renderIncludes f.includes
  let decls := joinDecls f.decls
  let text :=
    if header.isEmpty then decls
    else if decls.isEmpty then header
    else header ++ "\n" ++ decls
  if text.isEmpty then "" else text ++ "\n"

/-! ### Printing -/

instance : ToString CType := ⟨renderType⟩
instance : ToString Expr := ⟨renderExpr 0⟩
instance : ToString Stmt := ⟨renderStmt 0⟩
instance : ToString Fun := ⟨renderFun⟩
instance : ToString Decl := ⟨renderDecl⟩
instance : ToString CFile := ⟨renderFile⟩

end CGen
