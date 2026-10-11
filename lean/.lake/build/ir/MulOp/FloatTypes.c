// Lean compiler output
// Module: MulOp.FloatTypes
// Imports: public import Init public meta import Init public import MulOp.MulCore
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
double l_Float_ofScientific(lean_object*, uint8_t, lean_object*);
double lean_float_negate(double);
double lean_float_div(double, double);
static lean_once_cell_t lp_mul__op_MulOp_NaN___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mul__op_MulOp_NaN___closed__0;
static lean_once_cell_t lp_mul__op_MulOp_NaN___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mul__op_MulOp_NaN___closed__1;
LEAN_EXPORT double lp_mul__op_MulOp_NaN;
static lean_once_cell_t lp_mul__op_MulOp_Inf___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mul__op_MulOp_Inf___closed__0;
static lean_once_cell_t lp_mul__op_MulOp_Inf___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mul__op_MulOp_Inf___closed__1;
LEAN_EXPORT double lp_mul__op_MulOp_Inf;
static const lean_ctor_object lp_mul__op_MulOp_fex2A___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mul__op_MulOp_fex2A___closed__0 = (const lean_object*)&lp_mul__op_MulOp_fex2A___closed__0_value;
static lean_once_cell_t lp_mul__op_MulOp_fex2A___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mul__op_MulOp_fex2A___closed__1;
static lean_once_cell_t lp_mul__op_MulOp_fex2A___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mul__op_MulOp_fex2A___closed__2;
LEAN_EXPORT lean_object* lp_mul__op_MulOp_fex2A___closed__3___boxed__const__1;
static lean_once_cell_t lp_mul__op_MulOp_fex2A___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mul__op_MulOp_fex2A___closed__3;
static lean_once_cell_t lp_mul__op_MulOp_fex2A___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mul__op_MulOp_fex2A___closed__4;
LEAN_EXPORT lean_object* lp_mul__op_MulOp_fex2A___closed__5___boxed__const__1;
static lean_once_cell_t lp_mul__op_MulOp_fex2A___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mul__op_MulOp_fex2A___closed__5;
LEAN_EXPORT lean_object* lp_mul__op_MulOp_fex2A___closed__6___boxed__const__1;
static lean_once_cell_t lp_mul__op_MulOp_fex2A___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mul__op_MulOp_fex2A___closed__6;
LEAN_EXPORT lean_object* lp_mul__op_MulOp_fex2A___closed__7___boxed__const__1;
static lean_once_cell_t lp_mul__op_MulOp_fex2A___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mul__op_MulOp_fex2A___closed__7;
LEAN_EXPORT lean_object* lp_mul__op_MulOp_fex2A___closed__8___boxed__const__1;
static lean_once_cell_t lp_mul__op_MulOp_fex2A___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mul__op_MulOp_fex2A___closed__8;
static lean_once_cell_t lp_mul__op_MulOp_fex2A___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mul__op_MulOp_fex2A___closed__9;
LEAN_EXPORT lean_object* lp_mul__op_MulOp_fex2A;
static lean_once_cell_t lp_mul__op_MulOp_fex2B___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mul__op_MulOp_fex2B___closed__0;
static lean_once_cell_t lp_mul__op_MulOp_fex2B___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mul__op_MulOp_fex2B___closed__1;
static lean_once_cell_t lp_mul__op_MulOp_fex2B___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mul__op_MulOp_fex2B___closed__2;
static lean_once_cell_t lp_mul__op_MulOp_fex2B___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mul__op_MulOp_fex2B___closed__3;
static lean_once_cell_t lp_mul__op_MulOp_fex2B___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mul__op_MulOp_fex2B___closed__4;
LEAN_EXPORT lean_object* lp_mul__op_MulOp_fex2B___closed__5___boxed__const__1;
static lean_once_cell_t lp_mul__op_MulOp_fex2B___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mul__op_MulOp_fex2B___closed__5;
LEAN_EXPORT lean_object* lp_mul__op_MulOp_fex2B___closed__6___boxed__const__1;
static lean_once_cell_t lp_mul__op_MulOp_fex2B___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mul__op_MulOp_fex2B___closed__6;
LEAN_EXPORT lean_object* lp_mul__op_MulOp_fex2B___closed__7___boxed__const__1;
static lean_once_cell_t lp_mul__op_MulOp_fex2B___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mul__op_MulOp_fex2B___closed__7;
static lean_once_cell_t lp_mul__op_MulOp_fex2B___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mul__op_MulOp_fex2B___closed__8;
static lean_once_cell_t lp_mul__op_MulOp_fex2B___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mul__op_MulOp_fex2B___closed__9;
static lean_once_cell_t lp_mul__op_MulOp_fex2B___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mul__op_MulOp_fex2B___closed__10;
LEAN_EXPORT lean_object* lp_mul__op_MulOp_fex2B;
static double _init_lp_mul__op_MulOp_NaN___closed__0(void){
_start:
{
lean_object* v___x_1_; uint8_t v___x_2_; lean_object* v___x_3_; double v___x_4_; 
v___x_1_ = lean_unsigned_to_nat(1u);
v___x_2_ = 1;
v___x_3_ = lean_unsigned_to_nat(0u);
v___x_4_ = l_Float_ofScientific(v___x_3_, v___x_2_, v___x_1_);
return v___x_4_;
}
}
static double _init_lp_mul__op_MulOp_NaN___closed__1(void){
_start:
{
double v___x_5_; double v___x_6_; 
v___x_5_ = lean_float_once(&lp_mul__op_MulOp_NaN___closed__0, &lp_mul__op_MulOp_NaN___closed__0_once, _init_lp_mul__op_MulOp_NaN___closed__0);
v___x_6_ = lean_float_div(v___x_5_, v___x_5_);
return v___x_6_;
}
}
static double _init_lp_mul__op_MulOp_NaN(void){
_start:
{
double v___x_7_; 
v___x_7_ = lean_float_once(&lp_mul__op_MulOp_NaN___closed__1, &lp_mul__op_MulOp_NaN___closed__1_once, _init_lp_mul__op_MulOp_NaN___closed__1);
return v___x_7_;
}
}
static double _init_lp_mul__op_MulOp_Inf___closed__0(void){
_start:
{
lean_object* v___x_8_; uint8_t v___x_9_; lean_object* v___x_10_; double v___x_11_; 
v___x_8_ = lean_unsigned_to_nat(1u);
v___x_9_ = 1;
v___x_10_ = lean_unsigned_to_nat(10u);
v___x_11_ = l_Float_ofScientific(v___x_10_, v___x_9_, v___x_8_);
return v___x_11_;
}
}
static double _init_lp_mul__op_MulOp_Inf___closed__1(void){
_start:
{
double v___x_12_; double v___x_13_; double v___x_14_; 
v___x_12_ = lean_float_once(&lp_mul__op_MulOp_NaN___closed__0, &lp_mul__op_MulOp_NaN___closed__0_once, _init_lp_mul__op_MulOp_NaN___closed__0);
v___x_13_ = lean_float_once(&lp_mul__op_MulOp_Inf___closed__0, &lp_mul__op_MulOp_Inf___closed__0_once, _init_lp_mul__op_MulOp_Inf___closed__0);
v___x_14_ = lean_float_div(v___x_13_, v___x_12_);
return v___x_14_;
}
}
static double _init_lp_mul__op_MulOp_Inf(void){
_start:
{
double v___x_15_; 
v___x_15_ = lean_float_once(&lp_mul__op_MulOp_Inf___closed__1, &lp_mul__op_MulOp_Inf___closed__1_once, _init_lp_mul__op_MulOp_Inf___closed__1);
return v___x_15_;
}
}
static double _init_lp_mul__op_MulOp_fex2A___closed__1(void){
_start:
{
double v___x_19_; double v___x_20_; 
v___x_19_ = lp_mul__op_MulOp_Inf;
v___x_20_ = lean_float_negate(v___x_19_);
return v___x_20_;
}
}
static double _init_lp_mul__op_MulOp_fex2A___closed__2(void){
_start:
{
double v___x_21_; double v___x_22_; 
v___x_21_ = lean_float_once(&lp_mul__op_MulOp_NaN___closed__0, &lp_mul__op_MulOp_NaN___closed__0_once, _init_lp_mul__op_MulOp_NaN___closed__0);
v___x_22_ = lean_float_negate(v___x_21_);
return v___x_22_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2A___closed__3___boxed__const__1(void){
_start:
{
double v___x_23_; lean_object* v___x_24_; 
v___x_23_ = lean_float_once(&lp_mul__op_MulOp_fex2A___closed__2, &lp_mul__op_MulOp_fex2A___closed__2_once, _init_lp_mul__op_MulOp_fex2A___closed__2);
v___x_24_ = lean_box_float(v___x_23_);
return v___x_24_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2A___closed__3(void){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; 
v___x_25_ = lean_box(0);
v___x_26_ = lp_mul__op_MulOp_fex2A___closed__3___boxed__const__1;
v___x_27_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_27_, 0, v___x_26_);
lean_ctor_set(v___x_27_, 1, v___x_25_);
return v___x_27_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2A___closed__4(void){
_start:
{
lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; 
v___x_28_ = lean_obj_once(&lp_mul__op_MulOp_fex2A___closed__3, &lp_mul__op_MulOp_fex2A___closed__3_once, _init_lp_mul__op_MulOp_fex2A___closed__3);
v___x_29_ = lp_mul__op_MulOp_fex2A___closed__3___boxed__const__1;
v___x_30_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_30_, 0, v___x_29_);
lean_ctor_set(v___x_30_, 1, v___x_28_);
return v___x_30_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2A___closed__5___boxed__const__1(void){
_start:
{
double v___x_31_; lean_object* v___x_32_; 
v___x_31_ = lean_float_once(&lp_mul__op_MulOp_NaN___closed__0, &lp_mul__op_MulOp_NaN___closed__0_once, _init_lp_mul__op_MulOp_NaN___closed__0);
v___x_32_ = lean_box_float(v___x_31_);
return v___x_32_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2A___closed__5(void){
_start:
{
lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; 
v___x_33_ = lean_obj_once(&lp_mul__op_MulOp_fex2A___closed__4, &lp_mul__op_MulOp_fex2A___closed__4_once, _init_lp_mul__op_MulOp_fex2A___closed__4);
v___x_34_ = lp_mul__op_MulOp_fex2A___closed__5___boxed__const__1;
v___x_35_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_35_, 0, v___x_34_);
lean_ctor_set(v___x_35_, 1, v___x_33_);
return v___x_35_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2A___closed__6___boxed__const__1(void){
_start:
{
double v___x_36_; lean_object* v___x_37_; 
v___x_36_ = lean_float_once(&lp_mul__op_MulOp_fex2A___closed__1, &lp_mul__op_MulOp_fex2A___closed__1_once, _init_lp_mul__op_MulOp_fex2A___closed__1);
v___x_37_ = lean_box_float(v___x_36_);
return v___x_37_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2A___closed__6(void){
_start:
{
lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; 
v___x_38_ = lean_obj_once(&lp_mul__op_MulOp_fex2A___closed__5, &lp_mul__op_MulOp_fex2A___closed__5_once, _init_lp_mul__op_MulOp_fex2A___closed__5);
v___x_39_ = lp_mul__op_MulOp_fex2A___closed__6___boxed__const__1;
v___x_40_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_40_, 0, v___x_39_);
lean_ctor_set(v___x_40_, 1, v___x_38_);
return v___x_40_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2A___closed__7___boxed__const__1(void){
_start:
{
double v___x_41_; lean_object* v___x_42_; 
v___x_41_ = lp_mul__op_MulOp_Inf;
v___x_42_ = lean_box_float(v___x_41_);
return v___x_42_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2A___closed__7(void){
_start:
{
lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v___x_43_ = lean_obj_once(&lp_mul__op_MulOp_fex2A___closed__6, &lp_mul__op_MulOp_fex2A___closed__6_once, _init_lp_mul__op_MulOp_fex2A___closed__6);
v___x_44_ = lp_mul__op_MulOp_fex2A___closed__7___boxed__const__1;
v___x_45_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_45_, 0, v___x_44_);
lean_ctor_set(v___x_45_, 1, v___x_43_);
return v___x_45_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2A___closed__8___boxed__const__1(void){
_start:
{
double v___x_46_; lean_object* v___x_47_; 
v___x_46_ = lean_float_once(&lp_mul__op_MulOp_Inf___closed__0, &lp_mul__op_MulOp_Inf___closed__0_once, _init_lp_mul__op_MulOp_Inf___closed__0);
v___x_47_ = lean_box_float(v___x_46_);
return v___x_47_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2A___closed__8(void){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; 
v___x_48_ = lean_obj_once(&lp_mul__op_MulOp_fex2A___closed__7, &lp_mul__op_MulOp_fex2A___closed__7_once, _init_lp_mul__op_MulOp_fex2A___closed__7);
v___x_49_ = lp_mul__op_MulOp_fex2A___closed__8___boxed__const__1;
v___x_50_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_50_, 0, v___x_49_);
lean_ctor_set(v___x_50_, 1, v___x_48_);
return v___x_50_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2A___closed__9(void){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; 
v___x_51_ = lean_obj_once(&lp_mul__op_MulOp_fex2A___closed__8, &lp_mul__op_MulOp_fex2A___closed__8_once, _init_lp_mul__op_MulOp_fex2A___closed__8);
v___x_52_ = ((lean_object*)(lp_mul__op_MulOp_fex2A___closed__0));
v___x_53_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_53_, 0, v___x_52_);
lean_ctor_set(v___x_53_, 1, v___x_51_);
return v___x_53_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2A(void){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lean_obj_once(&lp_mul__op_MulOp_fex2A___closed__9, &lp_mul__op_MulOp_fex2A___closed__9_once, _init_lp_mul__op_MulOp_fex2A___closed__9);
return v___x_54_;
}
}
static double _init_lp_mul__op_MulOp_fex2B___closed__0(void){
_start:
{
lean_object* v___x_55_; uint8_t v___x_56_; lean_object* v___x_57_; double v___x_58_; 
v___x_55_ = lean_unsigned_to_nat(1u);
v___x_56_ = 1;
v___x_57_ = lean_unsigned_to_nat(20u);
v___x_58_ = l_Float_ofScientific(v___x_57_, v___x_56_, v___x_55_);
return v___x_58_;
}
}
static double _init_lp_mul__op_MulOp_fex2B___closed__1(void){
_start:
{
double v___x_59_; double v___x_60_; 
v___x_59_ = lean_float_once(&lp_mul__op_MulOp_fex2B___closed__0, &lp_mul__op_MulOp_fex2B___closed__0_once, _init_lp_mul__op_MulOp_fex2B___closed__0);
v___x_60_ = lean_float_negate(v___x_59_);
return v___x_60_;
}
}
static double _init_lp_mul__op_MulOp_fex2B___closed__2(void){
_start:
{
lean_object* v___x_61_; uint8_t v___x_62_; lean_object* v___x_63_; double v___x_64_; 
v___x_61_ = lean_unsigned_to_nat(1u);
v___x_62_ = 1;
v___x_63_ = lean_unsigned_to_nat(30u);
v___x_64_ = l_Float_ofScientific(v___x_63_, v___x_62_, v___x_61_);
return v___x_64_;
}
}
static double _init_lp_mul__op_MulOp_fex2B___closed__3(void){
_start:
{
double v___x_65_; double v___x_66_; 
v___x_65_ = lean_float_once(&lp_mul__op_MulOp_fex2B___closed__2, &lp_mul__op_MulOp_fex2B___closed__2_once, _init_lp_mul__op_MulOp_fex2B___closed__2);
v___x_66_ = lean_float_negate(v___x_65_);
return v___x_66_;
}
}
static double _init_lp_mul__op_MulOp_fex2B___closed__4(void){
_start:
{
lean_object* v___x_67_; uint8_t v___x_68_; lean_object* v___x_69_; double v___x_70_; 
v___x_67_ = lean_unsigned_to_nat(1u);
v___x_68_ = 1;
v___x_69_ = lean_unsigned_to_nat(50u);
v___x_70_ = l_Float_ofScientific(v___x_69_, v___x_68_, v___x_67_);
return v___x_70_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2B___closed__5___boxed__const__1(void){
_start:
{
double v___x_71_; lean_object* v___x_72_; 
v___x_71_ = lean_float_once(&lp_mul__op_MulOp_fex2B___closed__4, &lp_mul__op_MulOp_fex2B___closed__4_once, _init_lp_mul__op_MulOp_fex2B___closed__4);
v___x_72_ = lean_box_float(v___x_71_);
return v___x_72_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2B___closed__5(void){
_start:
{
lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; 
v___x_73_ = lean_obj_once(&lp_mul__op_MulOp_fex2A___closed__3, &lp_mul__op_MulOp_fex2A___closed__3_once, _init_lp_mul__op_MulOp_fex2A___closed__3);
v___x_74_ = lp_mul__op_MulOp_fex2B___closed__5___boxed__const__1;
v___x_75_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_75_, 0, v___x_74_);
lean_ctor_set(v___x_75_, 1, v___x_73_);
return v___x_75_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2B___closed__6___boxed__const__1(void){
_start:
{
double v___x_76_; lean_object* v___x_77_; 
v___x_76_ = lean_float_once(&lp_mul__op_MulOp_fex2B___closed__3, &lp_mul__op_MulOp_fex2B___closed__3_once, _init_lp_mul__op_MulOp_fex2B___closed__3);
v___x_77_ = lean_box_float(v___x_76_);
return v___x_77_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2B___closed__6(void){
_start:
{
lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_78_ = lean_obj_once(&lp_mul__op_MulOp_fex2B___closed__5, &lp_mul__op_MulOp_fex2B___closed__5_once, _init_lp_mul__op_MulOp_fex2B___closed__5);
v___x_79_ = lp_mul__op_MulOp_fex2B___closed__6___boxed__const__1;
v___x_80_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_80_, 0, v___x_79_);
lean_ctor_set(v___x_80_, 1, v___x_78_);
return v___x_80_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2B___closed__7___boxed__const__1(void){
_start:
{
double v___x_81_; lean_object* v___x_82_; 
v___x_81_ = lean_float_once(&lp_mul__op_MulOp_fex2B___closed__1, &lp_mul__op_MulOp_fex2B___closed__1_once, _init_lp_mul__op_MulOp_fex2B___closed__1);
v___x_82_ = lean_box_float(v___x_81_);
return v___x_82_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2B___closed__7(void){
_start:
{
lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; 
v___x_83_ = lean_obj_once(&lp_mul__op_MulOp_fex2B___closed__6, &lp_mul__op_MulOp_fex2B___closed__6_once, _init_lp_mul__op_MulOp_fex2B___closed__6);
v___x_84_ = lp_mul__op_MulOp_fex2B___closed__7___boxed__const__1;
v___x_85_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_85_, 0, v___x_84_);
lean_ctor_set(v___x_85_, 1, v___x_83_);
return v___x_85_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2B___closed__8(void){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; 
v___x_86_ = lean_obj_once(&lp_mul__op_MulOp_fex2B___closed__7, &lp_mul__op_MulOp_fex2B___closed__7_once, _init_lp_mul__op_MulOp_fex2B___closed__7);
v___x_87_ = lp_mul__op_MulOp_fex2A___closed__5___boxed__const__1;
v___x_88_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_88_, 0, v___x_87_);
lean_ctor_set(v___x_88_, 1, v___x_86_);
return v___x_88_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2B___closed__9(void){
_start:
{
lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; 
v___x_89_ = lean_obj_once(&lp_mul__op_MulOp_fex2B___closed__8, &lp_mul__op_MulOp_fex2B___closed__8_once, _init_lp_mul__op_MulOp_fex2B___closed__8);
v___x_90_ = lp_mul__op_MulOp_fex2A___closed__8___boxed__const__1;
v___x_91_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_91_, 0, v___x_90_);
lean_ctor_set(v___x_91_, 1, v___x_89_);
return v___x_91_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2B___closed__10(void){
_start:
{
lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; 
v___x_92_ = lean_obj_once(&lp_mul__op_MulOp_fex2B___closed__9, &lp_mul__op_MulOp_fex2B___closed__9_once, _init_lp_mul__op_MulOp_fex2B___closed__9);
v___x_93_ = ((lean_object*)(lp_mul__op_MulOp_fex2A___closed__0));
v___x_94_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_94_, 0, v___x_93_);
lean_ctor_set(v___x_94_, 1, v___x_92_);
return v___x_94_;
}
}
static lean_object* _init_lp_mul__op_MulOp_fex2B(void){
_start:
{
lean_object* v___x_95_; 
v___x_95_ = lean_obj_once(&lp_mul__op_MulOp_fex2B___closed__10, &lp_mul__op_MulOp_fex2B___closed__10_once, _init_lp_mul__op_MulOp_fex2B___closed__10);
return v___x_95_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mul__op_MulOp_MulCore(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mul__op_MulOp_FloatTypes(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mul__op_MulOp_MulCore(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mul__op_MulOp_NaN = _init_lp_mul__op_MulOp_NaN();
lp_mul__op_MulOp_Inf = _init_lp_mul__op_MulOp_Inf();
lp_mul__op_MulOp_fex2A___closed__3___boxed__const__1 = _init_lp_mul__op_MulOp_fex2A___closed__3___boxed__const__1();
lean_mark_persistent(lp_mul__op_MulOp_fex2A___closed__3___boxed__const__1);
lp_mul__op_MulOp_fex2A___closed__5___boxed__const__1 = _init_lp_mul__op_MulOp_fex2A___closed__5___boxed__const__1();
lean_mark_persistent(lp_mul__op_MulOp_fex2A___closed__5___boxed__const__1);
lp_mul__op_MulOp_fex2A___closed__6___boxed__const__1 = _init_lp_mul__op_MulOp_fex2A___closed__6___boxed__const__1();
lean_mark_persistent(lp_mul__op_MulOp_fex2A___closed__6___boxed__const__1);
lp_mul__op_MulOp_fex2A___closed__7___boxed__const__1 = _init_lp_mul__op_MulOp_fex2A___closed__7___boxed__const__1();
lean_mark_persistent(lp_mul__op_MulOp_fex2A___closed__7___boxed__const__1);
lp_mul__op_MulOp_fex2A___closed__8___boxed__const__1 = _init_lp_mul__op_MulOp_fex2A___closed__8___boxed__const__1();
lean_mark_persistent(lp_mul__op_MulOp_fex2A___closed__8___boxed__const__1);
lp_mul__op_MulOp_fex2A = _init_lp_mul__op_MulOp_fex2A();
lean_mark_persistent(lp_mul__op_MulOp_fex2A);
lp_mul__op_MulOp_fex2B___closed__5___boxed__const__1 = _init_lp_mul__op_MulOp_fex2B___closed__5___boxed__const__1();
lean_mark_persistent(lp_mul__op_MulOp_fex2B___closed__5___boxed__const__1);
lp_mul__op_MulOp_fex2B___closed__6___boxed__const__1 = _init_lp_mul__op_MulOp_fex2B___closed__6___boxed__const__1();
lean_mark_persistent(lp_mul__op_MulOp_fex2B___closed__6___boxed__const__1);
lp_mul__op_MulOp_fex2B___closed__7___boxed__const__1 = _init_lp_mul__op_MulOp_fex2B___closed__7___boxed__const__1();
lean_mark_persistent(lp_mul__op_MulOp_fex2B___closed__7___boxed__const__1);
lp_mul__op_MulOp_fex2B = _init_lp_mul__op_MulOp_fex2B();
lean_mark_persistent(lp_mul__op_MulOp_fex2B);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
