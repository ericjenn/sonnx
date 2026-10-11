// Lean compiler output
// Module: MulOp.Broadcast
// Imports: public import Init public meta import Init public import Init public import MulOp.Tensor
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_List_replicateTR___redArg(lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_zipWithTR_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_List_drop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_padTo(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_padTo___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_bcastDim(lean_object*, lean_object*);
static const lean_ctor_object lp_mul__op_MulOp_allSome___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mul__op_MulOp_allSome___redArg___closed__0 = (const lean_object*)&lp_mul__op_MulOp_allSome___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mul__op_MulOp_allSome___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_allSome(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op___private_MulOp_Broadcast_0__MulOp_allSome_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op___private_MulOp_Broadcast_0__MulOp_allSome_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_bcastIdx___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_bcastIdx___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mul__op_MulOp_bcastIdx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mul__op_MulOp_bcastIdx___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mul__op_MulOp_bcastIdx___closed__0 = (const lean_object*)&lp_mul__op_MulOp_bcastIdx___closed__0_value;
static const lean_array_object lp_mul__op_MulOp_bcastIdx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mul__op_MulOp_bcastIdx___closed__1 = (const lean_object*)&lp_mul__op_MulOp_bcastIdx___closed__1_value;
LEAN_EXPORT lean_object* lp_mul__op_MulOp_bcastIdx(lean_object*, lean_object*);
static const lean_closure_object lp_mul__op_MulOp_bcastShape___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mul__op_MulOp_bcastDim, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mul__op_MulOp_bcastShape___closed__0 = (const lean_object*)&lp_mul__op_MulOp_bcastShape___closed__0_value;
static const lean_array_object lp_mul__op_MulOp_bcastShape___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mul__op_MulOp_bcastShape___closed__1 = (const lean_object*)&lp_mul__op_MulOp_bcastShape___closed__1_value;
LEAN_EXPORT lean_object* lp_mul__op_MulOp_bcastShape(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_padTo(lean_object* v_k_1_, lean_object* v_s_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; 
v___x_3_ = l_List_lengthTR___redArg(v_s_2_);
v___x_4_ = lean_nat_sub(v_k_1_, v___x_3_);
lean_dec(v___x_3_);
v___x_5_ = lean_unsigned_to_nat(1u);
v___x_6_ = l_List_replicateTR___redArg(v___x_4_, v___x_5_);
v___x_7_ = l_List_appendTR___redArg(v___x_6_, v_s_2_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_padTo___boxed(lean_object* v_k_8_, lean_object* v_s_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mul__op_MulOp_padTo(v_k_8_, v_s_9_);
lean_dec(v_k_8_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_bcastDim(lean_object* v_a_11_, lean_object* v_b_12_){
_start:
{
uint8_t v___x_13_; 
v___x_13_ = lean_nat_dec_eq(v_a_11_, v_b_12_);
if (v___x_13_ == 0)
{
lean_object* v___x_14_; uint8_t v___x_15_; 
v___x_14_ = lean_unsigned_to_nat(1u);
v___x_15_ = lean_nat_dec_eq(v_a_11_, v___x_14_);
if (v___x_15_ == 0)
{
uint8_t v___x_16_; 
v___x_16_ = lean_nat_dec_eq(v_b_12_, v___x_14_);
lean_dec(v_b_12_);
if (v___x_16_ == 0)
{
lean_object* v___x_17_; 
lean_dec(v_a_11_);
v___x_17_ = lean_box(0);
return v___x_17_;
}
else
{
lean_object* v___x_18_; 
v___x_18_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_18_, 0, v_a_11_);
return v___x_18_;
}
}
else
{
lean_object* v___x_19_; 
lean_dec(v_a_11_);
v___x_19_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_19_, 0, v_b_12_);
return v___x_19_;
}
}
else
{
lean_object* v___x_20_; 
lean_dec(v_b_12_);
v___x_20_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_20_, 0, v_a_11_);
return v___x_20_;
}
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_allSome___redArg(lean_object* v_x_23_){
_start:
{
if (lean_obj_tag(v_x_23_) == 0)
{
lean_object* v___x_24_; 
v___x_24_ = ((lean_object*)(lp_mul__op_MulOp_allSome___redArg___closed__0));
return v___x_24_;
}
else
{
lean_object* v_head_25_; 
v_head_25_ = lean_ctor_get(v_x_23_, 0);
lean_inc(v_head_25_);
if (lean_obj_tag(v_head_25_) == 0)
{
lean_object* v___x_26_; 
lean_dec_ref_known(v_x_23_, 2);
v___x_26_ = lean_box(0);
return v___x_26_;
}
else
{
lean_object* v_tail_27_; lean_object* v___x_29_; uint8_t v_isShared_30_; uint8_t v_isSharedCheck_44_; 
v_tail_27_ = lean_ctor_get(v_x_23_, 1);
v_isSharedCheck_44_ = !lean_is_exclusive(v_x_23_);
if (v_isSharedCheck_44_ == 0)
{
lean_object* v_unused_45_; 
v_unused_45_ = lean_ctor_get(v_x_23_, 0);
lean_dec(v_unused_45_);
v___x_29_ = v_x_23_;
v_isShared_30_ = v_isSharedCheck_44_;
goto v_resetjp_28_;
}
else
{
lean_inc(v_tail_27_);
lean_dec(v_x_23_);
v___x_29_ = lean_box(0);
v_isShared_30_ = v_isSharedCheck_44_;
goto v_resetjp_28_;
}
v_resetjp_28_:
{
lean_object* v_val_31_; lean_object* v___x_32_; 
v_val_31_ = lean_ctor_get(v_head_25_, 0);
lean_inc(v_val_31_);
lean_dec_ref_known(v_head_25_, 1);
v___x_32_ = lp_mul__op_MulOp_allSome___redArg(v_tail_27_);
if (lean_obj_tag(v___x_32_) == 0)
{
lean_dec(v_val_31_);
lean_del_object(v___x_29_);
return v___x_32_;
}
else
{
lean_object* v_val_33_; lean_object* v___x_35_; uint8_t v_isShared_36_; uint8_t v_isSharedCheck_43_; 
v_val_33_ = lean_ctor_get(v___x_32_, 0);
v_isSharedCheck_43_ = !lean_is_exclusive(v___x_32_);
if (v_isSharedCheck_43_ == 0)
{
v___x_35_ = v___x_32_;
v_isShared_36_ = v_isSharedCheck_43_;
goto v_resetjp_34_;
}
else
{
lean_inc(v_val_33_);
lean_dec(v___x_32_);
v___x_35_ = lean_box(0);
v_isShared_36_ = v_isSharedCheck_43_;
goto v_resetjp_34_;
}
v_resetjp_34_:
{
lean_object* v___x_38_; 
if (v_isShared_30_ == 0)
{
lean_ctor_set(v___x_29_, 1, v_val_33_);
lean_ctor_set(v___x_29_, 0, v_val_31_);
v___x_38_ = v___x_29_;
goto v_reusejp_37_;
}
else
{
lean_object* v_reuseFailAlloc_42_; 
v_reuseFailAlloc_42_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_42_, 0, v_val_31_);
lean_ctor_set(v_reuseFailAlloc_42_, 1, v_val_33_);
v___x_38_ = v_reuseFailAlloc_42_;
goto v_reusejp_37_;
}
v_reusejp_37_:
{
lean_object* v___x_40_; 
if (v_isShared_36_ == 0)
{
lean_ctor_set(v___x_35_, 0, v___x_38_);
v___x_40_ = v___x_35_;
goto v_reusejp_39_;
}
else
{
lean_object* v_reuseFailAlloc_41_; 
v_reuseFailAlloc_41_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_41_, 0, v___x_38_);
v___x_40_ = v_reuseFailAlloc_41_;
goto v_reusejp_39_;
}
v_reusejp_39_:
{
return v___x_40_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_allSome(lean_object* v_00_u03b1_46_, lean_object* v_x_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_mul__op_MulOp_allSome___redArg(v_x_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mul__op___private_MulOp_Broadcast_0__MulOp_allSome_match__1_splitter___redArg(lean_object* v_x_49_, lean_object* v_h__1_50_, lean_object* v_h__2_51_, lean_object* v_h__3_52_){
_start:
{
if (lean_obj_tag(v_x_49_) == 0)
{
lean_object* v___x_53_; lean_object* v___x_54_; 
lean_dec(v_h__3_52_);
lean_dec(v_h__2_51_);
v___x_53_ = lean_box(0);
v___x_54_ = lean_apply_1(v_h__1_50_, v___x_53_);
return v___x_54_;
}
else
{
lean_object* v_head_55_; 
lean_dec(v_h__1_50_);
v_head_55_ = lean_ctor_get(v_x_49_, 0);
if (lean_obj_tag(v_head_55_) == 0)
{
lean_object* v_tail_56_; lean_object* v___x_57_; 
lean_dec(v_h__2_51_);
v_tail_56_ = lean_ctor_get(v_x_49_, 1);
lean_inc(v_tail_56_);
lean_dec_ref_known(v_x_49_, 2);
v___x_57_ = lean_apply_1(v_h__3_52_, v_tail_56_);
return v___x_57_;
}
else
{
lean_object* v_tail_58_; lean_object* v_val_59_; lean_object* v___x_60_; 
lean_inc_ref(v_head_55_);
lean_dec(v_h__3_52_);
v_tail_58_ = lean_ctor_get(v_x_49_, 1);
lean_inc(v_tail_58_);
lean_dec_ref_known(v_x_49_, 2);
v_val_59_ = lean_ctor_get(v_head_55_, 0);
lean_inc(v_val_59_);
lean_dec_ref_known(v_head_55_, 1);
v___x_60_ = lean_apply_2(v_h__2_51_, v_val_59_, v_tail_58_);
return v___x_60_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mul__op___private_MulOp_Broadcast_0__MulOp_allSome_match__1_splitter(lean_object* v_00_u03b1_61_, lean_object* v_motive_62_, lean_object* v_x_63_, lean_object* v_h__1_64_, lean_object* v_h__2_65_, lean_object* v_h__3_66_){
_start:
{
if (lean_obj_tag(v_x_63_) == 0)
{
lean_object* v___x_67_; lean_object* v___x_68_; 
lean_dec(v_h__3_66_);
lean_dec(v_h__2_65_);
v___x_67_ = lean_box(0);
v___x_68_ = lean_apply_1(v_h__1_64_, v___x_67_);
return v___x_68_;
}
else
{
lean_object* v_head_69_; 
lean_dec(v_h__1_64_);
v_head_69_ = lean_ctor_get(v_x_63_, 0);
if (lean_obj_tag(v_head_69_) == 0)
{
lean_object* v_tail_70_; lean_object* v___x_71_; 
lean_dec(v_h__2_65_);
v_tail_70_ = lean_ctor_get(v_x_63_, 1);
lean_inc(v_tail_70_);
lean_dec_ref_known(v_x_63_, 2);
v___x_71_ = lean_apply_1(v_h__3_66_, v_tail_70_);
return v___x_71_;
}
else
{
lean_object* v_tail_72_; lean_object* v_val_73_; lean_object* v___x_74_; 
lean_inc_ref(v_head_69_);
lean_dec(v_h__3_66_);
v_tail_72_ = lean_ctor_get(v_x_63_, 1);
lean_inc(v_tail_72_);
lean_dec_ref_known(v_x_63_, 2);
v_val_73_ = lean_ctor_get(v_head_69_, 0);
lean_inc(v_val_73_);
lean_dec_ref_known(v_head_69_, 1);
v___x_74_ = lean_apply_2(v_h__2_65_, v_val_73_, v_tail_72_);
return v___x_74_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_bcastIdx___lam__0(lean_object* v_x_75_, lean_object* v_d_76_){
_start:
{
lean_object* v___x_77_; uint8_t v___x_78_; 
v___x_77_ = lean_unsigned_to_nat(1u);
v___x_78_ = lean_nat_dec_eq(v_d_76_, v___x_77_);
if (v___x_78_ == 0)
{
lean_inc(v_x_75_);
return v_x_75_;
}
else
{
lean_object* v___x_79_; 
v___x_79_ = lean_unsigned_to_nat(0u);
return v___x_79_;
}
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_bcastIdx___lam__0___boxed(lean_object* v_x_80_, lean_object* v_d_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_mul__op_MulOp_bcastIdx___lam__0(v_x_80_, v_d_81_);
lean_dec(v_d_81_);
lean_dec(v_x_80_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_bcastIdx(lean_object* v_i_86_, lean_object* v_s_87_){
_start:
{
lean_object* v___f_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; 
v___f_88_ = ((lean_object*)(lp_mul__op_MulOp_bcastIdx___closed__0));
v___x_89_ = l_List_lengthTR___redArg(v_i_86_);
v___x_90_ = l_List_lengthTR___redArg(v_s_87_);
v___x_91_ = lean_nat_sub(v___x_89_, v___x_90_);
lean_dec(v___x_90_);
v___x_92_ = lp_mul__op_MulOp_padTo(v___x_89_, v_s_87_);
lean_dec(v___x_89_);
v___x_93_ = ((lean_object*)(lp_mul__op_MulOp_bcastIdx___closed__1));
v___x_94_ = l___private_Init_Data_List_Impl_0__List_zipWithTR_go___redArg(v___f_88_, v_i_86_, v___x_92_, v___x_93_);
v___x_95_ = l_List_drop___redArg(v___x_91_, v___x_94_);
lean_dec(v___x_94_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_bcastShape(lean_object* v_sA_99_, lean_object* v_sB_100_){
_start:
{
lean_object* v___x_101_; lean_object* v___y_103_; lean_object* v___x_109_; lean_object* v___x_110_; uint8_t v___x_111_; 
v___x_101_ = ((lean_object*)(lp_mul__op_MulOp_bcastShape___closed__0));
v___x_109_ = l_List_lengthTR___redArg(v_sA_99_);
v___x_110_ = l_List_lengthTR___redArg(v_sB_100_);
v___x_111_ = lean_nat_dec_le(v___x_109_, v___x_110_);
if (v___x_111_ == 0)
{
lean_dec(v___x_110_);
v___y_103_ = v___x_109_;
goto v___jp_102_;
}
else
{
lean_dec(v___x_109_);
v___y_103_ = v___x_110_;
goto v___jp_102_;
}
v___jp_102_:
{
lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; 
v___x_104_ = lp_mul__op_MulOp_padTo(v___y_103_, v_sA_99_);
v___x_105_ = lp_mul__op_MulOp_padTo(v___y_103_, v_sB_100_);
lean_dec(v___y_103_);
v___x_106_ = ((lean_object*)(lp_mul__op_MulOp_bcastShape___closed__1));
v___x_107_ = l___private_Init_Data_List_Impl_0__List_zipWithTR_go___redArg(v___x_101_, v___x_104_, v___x_105_, v___x_106_);
v___x_108_ = lp_mul__op_MulOp_allSome___redArg(v___x_107_);
return v___x_108_;
}
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mul__op_MulOp_Tensor(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mul__op_MulOp_Broadcast(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mul__op_MulOp_Tensor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
