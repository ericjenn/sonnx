/-
The C code generator — the abstract syntax of C11.

This module defines the fragment of C11 that the operator kernels need: types,
expressions with their operators, statements, function definitions, and a
translation unit.  It is independent of `Sonnx`; the integer semantics the
kernels implement are those of `Sonnx.uintWrap` and `Sonnx.sintWrap`, and are
referenced only in documentation.

The AST is closed under the constructs the printer knows how to parenthesize
(see `CGen.Print`).  Extending the generator means adding statements to
`CGen.Kernel`, not new syntax here.
-/

namespace CGen

/-- A C type.  `int 8 true` is `int8_t`, `int 8 false` is `uint8_t`, `float 32`
is `float`, `float 64` is `double`.  Pointer and qualified types nest:
`ptr (const (float 32))` renders as `const float *`.  `named` is a typedef'd or
structure type. -/
inductive CType
  | void
  | bool
  | int (bits : Nat) (signed : Bool)
  | float (bits : Nat)
  | size
  | ptr (pointee : CType)
  | const (base : CType)
  | named (name : String)
  deriving Repr, Inhabited, DecidableEq

/-- Unary operators.  `postInc` and `postDec` are postfix; the rest are prefix. -/
inductive UnOp
  | neg
  | logNot
  | bitNot
  | deref
  | addr
  | preInc
  | preDec
  | postInc
  | postDec
  deriving Repr, Inhabited, DecidableEq

/-- Binary operators.  `assign` is right-associative and `comma` is the
lowest-precedence operator. -/
inductive BinOp
  | add | sub | mul | div | mod | shl | shr
  | bitAnd | bitOr | bitXor
  | lt | le | gt | ge | eq | ne
  | logAnd | logOr
  | assign
  | comma
  deriving Repr, Inhabited, DecidableEq

/-- A C expression.  `lit` is an atom printed verbatim (a numeric suffix, a
character constant, an already-formatted token): the caller guarantees that it
is atomic.  `intLit` renders a negative value in parentheses so that it cannot
merge with an adjacent unary minus. -/
inductive Expr
  | lit (text : String)
  | intLit (value : Int)
  | var (name : String)
  | index (base : Expr) (offset : Expr)
  | field (base : Expr) (name : String) (arrow : Bool)
  | unary (op : UnOp) (arg : Expr)
  | binary (op : BinOp) (lhs : Expr) (rhs : Expr)
  | cast (type : CType) (arg : Expr)
  | call (callee : String) (args : List Expr)
  | ternary (cond : Expr) (thenB : Expr) (elseB : Expr)
  | sizeofType (type : CType)
  deriving Repr, Inhabited

/-- A C statement.  `ret` is the `return` statement (`return` is a Lean
keyword).  A `forLoop`'s init clause may be a declaration, rendered without its
terminating semicolon. -/
inductive Stmt
  | expr (e : Expr)
  | decl (type : CType) (name : String) (init : Option Expr)
  | ifThen (cond : Expr) (thenB : Stmt) (elseB : Option Stmt)
  | forLoop (init : Option Stmt) (cond : Option Expr) (step : Option Expr) (body : Stmt)
  | whileLoop (cond : Expr) (body : Stmt)
  | ret (value : Option Expr)
  | block (body : List Stmt)
  | comment (text : String)
  | nop
  deriving Repr, Inhabited

/-- A function parameter. -/
structure Param where
  type : CType
  name : String
  deriving Repr, Inhabited

/-- A function: its name, return type, parameters and body. -/
structure Fun where
  name : String
  retType : CType
  params : List Param
  body : List Stmt
  deriving Repr, Inhabited

/-- One field of a struct definition. -/
structure Field where
  type : CType
  name : String
  deriving Repr, Inhabited

/-- A top-level declaration.  `structDef` renders a typedef of an anonymous
struct: the tensor type the kernels receive. -/
inductive Decl
  | typedef (type : CType) (name : String)
  | structDef (name : String) (fields : List Field)
  | funDecl (fn : Fun)
  | funDef (fn : Fun)
  | comment (text : String)
  deriving Repr, Inhabited

/-- A translation unit: an optional banner comment, the system headers to
include, and the declarations. -/
structure CFile where
  banner : List String := []
  includes : List String := []
  decls : List Decl := []
  deriving Repr, Inhabited

end CGen
