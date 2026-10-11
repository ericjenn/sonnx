// Lean compiler output
// Module: MulOp.IntTypes
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
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_int_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instInhabitedUIntN(lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instInhabitedUIntN___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instMulUIntN___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instMulUIntN___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instMulUIntN(lean_object*);
LEAN_EXPORT uint8_t lp_mul__op_MulOp_instDecidableEqUIntN___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instDecidableEqUIntN___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mul__op_MulOp_instDecidableEqUIntN(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instDecidableEqUIntN___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instInhabitedSIntN(lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instInhabitedSIntN___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instMulSIntN(lean_object*);
LEAN_EXPORT uint8_t lp_mul__op_MulOp_instDecidableEqSIntN___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instDecidableEqSIntN___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mul__op_MulOp_instDecidableEqSIntN(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instDecidableEqSIntN___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_SIntN_toInt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_SIntN_toInt___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instInhabitedUIntN(lean_object* v_n_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instInhabitedUIntN___boxed(lean_object* v_n_3_){
_start:
{
lean_object* v_res_4_; 
v_res_4_ = lp_mul__op_MulOp_instInhabitedUIntN(v_n_3_);
lean_dec(v_n_3_);
return v_res_4_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instMulUIntN___lam__0(lean_object* v_n_5_, lean_object* v_a_6_, lean_object* v_b_7_){
_start:
{
lean_object* v___x_8_; lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_8_ = lean_nat_mul(v_a_6_, v_b_7_);
v___x_9_ = lean_unsigned_to_nat(2u);
v___x_10_ = lean_nat_pow(v___x_9_, v_n_5_);
v___x_11_ = lean_nat_mod(v___x_8_, v___x_10_);
lean_dec(v___x_10_);
lean_dec(v___x_8_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instMulUIntN___lam__0___boxed(lean_object* v_n_12_, lean_object* v_a_13_, lean_object* v_b_14_){
_start:
{
lean_object* v_res_15_; 
v_res_15_ = lp_mul__op_MulOp_instMulUIntN___lam__0(v_n_12_, v_a_13_, v_b_14_);
lean_dec(v_b_14_);
lean_dec(v_a_13_);
lean_dec(v_n_12_);
return v_res_15_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instMulUIntN(lean_object* v_n_16_){
_start:
{
lean_object* v___f_17_; 
v___f_17_ = lean_alloc_closure((void*)(lp_mul__op_MulOp_instMulUIntN___lam__0___boxed), 3, 1);
lean_closure_set(v___f_17_, 0, v_n_16_);
return v___f_17_;
}
}
LEAN_EXPORT uint8_t lp_mul__op_MulOp_instDecidableEqUIntN___redArg(lean_object* v_a_18_, lean_object* v_b_19_){
_start:
{
uint8_t v___x_20_; 
v___x_20_ = lean_nat_dec_eq(v_a_18_, v_b_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instDecidableEqUIntN___redArg___boxed(lean_object* v_a_21_, lean_object* v_b_22_){
_start:
{
uint8_t v_res_23_; lean_object* v_r_24_; 
v_res_23_ = lp_mul__op_MulOp_instDecidableEqUIntN___redArg(v_a_21_, v_b_22_);
lean_dec(v_b_22_);
lean_dec(v_a_21_);
v_r_24_ = lean_box(v_res_23_);
return v_r_24_;
}
}
LEAN_EXPORT uint8_t lp_mul__op_MulOp_instDecidableEqUIntN(lean_object* v_n_25_, lean_object* v_a_26_, lean_object* v_b_27_){
_start:
{
uint8_t v___x_28_; 
v___x_28_ = lean_nat_dec_eq(v_a_26_, v_b_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instDecidableEqUIntN___boxed(lean_object* v_n_29_, lean_object* v_a_30_, lean_object* v_b_31_){
_start:
{
uint8_t v_res_32_; lean_object* v_r_33_; 
v_res_32_ = lp_mul__op_MulOp_instDecidableEqUIntN(v_n_29_, v_a_30_, v_b_31_);
lean_dec(v_b_31_);
lean_dec(v_a_30_);
lean_dec(v_n_29_);
v_r_33_ = lean_box(v_res_32_);
return v_r_33_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instInhabitedSIntN(lean_object* v_n_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lean_unsigned_to_nat(0u);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instInhabitedSIntN___boxed(lean_object* v_n_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_mul__op_MulOp_instInhabitedSIntN(v_n_36_);
lean_dec(v_n_36_);
return v_res_37_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instMulSIntN(lean_object* v_n_38_){
_start:
{
lean_object* v___f_39_; 
v___f_39_ = lean_alloc_closure((void*)(lp_mul__op_MulOp_instMulUIntN___lam__0___boxed), 3, 1);
lean_closure_set(v___f_39_, 0, v_n_38_);
return v___f_39_;
}
}
LEAN_EXPORT uint8_t lp_mul__op_MulOp_instDecidableEqSIntN___redArg(lean_object* v_a_40_, lean_object* v_b_41_){
_start:
{
uint8_t v___x_42_; 
v___x_42_ = lean_nat_dec_eq(v_a_40_, v_b_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instDecidableEqSIntN___redArg___boxed(lean_object* v_a_43_, lean_object* v_b_44_){
_start:
{
uint8_t v_res_45_; lean_object* v_r_46_; 
v_res_45_ = lp_mul__op_MulOp_instDecidableEqSIntN___redArg(v_a_43_, v_b_44_);
lean_dec(v_b_44_);
lean_dec(v_a_43_);
v_r_46_ = lean_box(v_res_45_);
return v_r_46_;
}
}
LEAN_EXPORT uint8_t lp_mul__op_MulOp_instDecidableEqSIntN(lean_object* v_n_47_, lean_object* v_a_48_, lean_object* v_b_49_){
_start:
{
uint8_t v___x_50_; 
v___x_50_ = lean_nat_dec_eq(v_a_48_, v_b_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instDecidableEqSIntN___boxed(lean_object* v_n_51_, lean_object* v_a_52_, lean_object* v_b_53_){
_start:
{
uint8_t v_res_54_; lean_object* v_r_55_; 
v_res_54_ = lp_mul__op_MulOp_instDecidableEqSIntN(v_n_51_, v_a_52_, v_b_53_);
lean_dec(v_b_53_);
lean_dec(v_a_52_);
lean_dec(v_n_51_);
v_r_55_ = lean_box(v_res_54_);
return v_r_55_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_SIntN_toInt(lean_object* v_n_56_, lean_object* v_a_57_){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; uint8_t v___x_62_; 
v___x_58_ = lean_unsigned_to_nat(2u);
v___x_59_ = lean_unsigned_to_nat(1u);
v___x_60_ = lean_nat_sub(v_n_56_, v___x_59_);
v___x_61_ = lean_nat_pow(v___x_58_, v___x_60_);
lean_dec(v___x_60_);
v___x_62_ = lean_nat_dec_lt(v_a_57_, v___x_61_);
lean_dec(v___x_61_);
if (v___x_62_ == 0)
{
lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
v___x_63_ = lean_nat_to_int(v_a_57_);
v___x_64_ = lean_nat_pow(v___x_58_, v_n_56_);
v___x_65_ = lean_nat_to_int(v___x_64_);
v___x_66_ = lean_int_sub(v___x_63_, v___x_65_);
lean_dec(v___x_65_);
lean_dec(v___x_63_);
return v___x_66_;
}
else
{
lean_object* v___x_67_; 
v___x_67_ = lean_nat_to_int(v_a_57_);
return v___x_67_;
}
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_SIntN_toInt___boxed(lean_object* v_n_68_, lean_object* v_a_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_mul__op_MulOp_SIntN_toInt(v_n_68_, v_a_69_);
lean_dec(v_n_68_);
return v_res_70_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mul__op_MulOp_MulCore(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mul__op_MulOp_IntTypes(uint8_t builtin) {
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
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
