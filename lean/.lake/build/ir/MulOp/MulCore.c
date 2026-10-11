// Lean compiler output
// Module: MulOp.MulCore
// Imports: public import Init public meta import Init public import MulOp.Broadcast
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
lean_object* lp_mul__op_MulOp_bcastShape(lean_object*, lean_object*);
lean_object* lp_mul__op_MulOp_bcastIdx(lean_object*, lean_object*);
lean_object* lp_mul__op_MulOp_Tensor_read___redArg(lean_object*, lean_object*);
lean_object* l_List_range(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_List_mapTR_loop___at___00MulOp_indices_spec__0(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mul__op_MulOp_indices___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mul__op_MulOp_indices___closed__0 = (const lean_object*)&lp_mul__op_MulOp_indices___closed__0_value;
LEAN_EXPORT lean_object* lp_mul__op___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00MulOp_indices_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_indices(lean_object*);
LEAN_EXPORT lean_object* lp_mul__op___private_MulOp_MulCore_0__MulOp_indices_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op___private_MulOp_MulCore_0__MulOp_indices_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_mulAt___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_mulAt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_mulData___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_mulData(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_mul___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_mul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_List_mapTR_loop___at___00MulOp_indices_spec__0(lean_object* v_k_1_, lean_object* v_a_2_, lean_object* v_a_3_){
_start:
{
if (lean_obj_tag(v_a_2_) == 0)
{
lean_object* v___x_4_; 
lean_dec(v_k_1_);
v___x_4_ = l_List_reverse___redArg(v_a_3_);
return v___x_4_;
}
else
{
lean_object* v_head_5_; lean_object* v_tail_6_; lean_object* v___x_8_; uint8_t v_isShared_9_; uint8_t v_isSharedCheck_15_; 
v_head_5_ = lean_ctor_get(v_a_2_, 0);
v_tail_6_ = lean_ctor_get(v_a_2_, 1);
v_isSharedCheck_15_ = !lean_is_exclusive(v_a_2_);
if (v_isSharedCheck_15_ == 0)
{
v___x_8_ = v_a_2_;
v_isShared_9_ = v_isSharedCheck_15_;
goto v_resetjp_7_;
}
else
{
lean_inc(v_tail_6_);
lean_inc(v_head_5_);
lean_dec(v_a_2_);
v___x_8_ = lean_box(0);
v_isShared_9_ = v_isSharedCheck_15_;
goto v_resetjp_7_;
}
v_resetjp_7_:
{
lean_object* v___x_11_; 
lean_inc(v_k_1_);
if (v_isShared_9_ == 0)
{
lean_ctor_set(v___x_8_, 1, v_head_5_);
lean_ctor_set(v___x_8_, 0, v_k_1_);
v___x_11_ = v___x_8_;
goto v_reusejp_10_;
}
else
{
lean_object* v_reuseFailAlloc_14_; 
v_reuseFailAlloc_14_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_14_, 0, v_k_1_);
lean_ctor_set(v_reuseFailAlloc_14_, 1, v_head_5_);
v___x_11_ = v_reuseFailAlloc_14_;
goto v_reusejp_10_;
}
v_reusejp_10_:
{
lean_object* v___x_12_; 
v___x_12_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_12_, 0, v___x_11_);
lean_ctor_set(v___x_12_, 1, v_a_3_);
v_a_2_ = v_tail_6_;
v_a_3_ = v___x_12_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mul__op___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00MulOp_indices_spec__1(lean_object* v_tail_18_, lean_object* v_a_19_, lean_object* v_a_20_){
_start:
{
if (lean_obj_tag(v_a_19_) == 0)
{
lean_object* v___x_21_; 
lean_dec(v_tail_18_);
v___x_21_ = lean_array_to_list(v_a_20_);
return v___x_21_;
}
else
{
lean_object* v_head_22_; lean_object* v_tail_23_; lean_object* v___x_24_; lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; 
v_head_22_ = lean_ctor_get(v_a_19_, 0);
lean_inc(v_head_22_);
v_tail_23_ = lean_ctor_get(v_a_19_, 1);
lean_inc(v_tail_23_);
lean_dec_ref_known(v_a_19_, 2);
lean_inc(v_tail_18_);
v___x_24_ = lp_mul__op_MulOp_indices(v_tail_18_);
v___x_25_ = lean_box(0);
v___x_26_ = lp_mul__op_List_mapTR_loop___at___00MulOp_indices_spec__0(v_head_22_, v___x_24_, v___x_25_);
v___x_27_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_20_, v___x_26_);
v_a_19_ = v_tail_23_;
v_a_20_ = v___x_27_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_indices(lean_object* v_x_29_){
_start:
{
if (lean_obj_tag(v_x_29_) == 0)
{
lean_object* v___x_30_; lean_object* v___x_31_; 
v___x_30_ = lean_box(0);
v___x_31_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_31_, 0, v_x_29_);
lean_ctor_set(v___x_31_, 1, v___x_30_);
return v___x_31_;
}
else
{
lean_object* v_head_32_; lean_object* v_tail_33_; lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; 
v_head_32_ = lean_ctor_get(v_x_29_, 0);
lean_inc(v_head_32_);
v_tail_33_ = lean_ctor_get(v_x_29_, 1);
lean_inc(v_tail_33_);
lean_dec_ref_known(v_x_29_, 2);
v___x_34_ = l_List_range(v_head_32_);
v___x_35_ = ((lean_object*)(lp_mul__op_MulOp_indices___closed__0));
v___x_36_ = lp_mul__op___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00MulOp_indices_spec__1(v_tail_33_, v___x_34_, v___x_35_);
return v___x_36_;
}
}
}
LEAN_EXPORT lean_object* lp_mul__op___private_MulOp_MulCore_0__MulOp_indices_match__1_splitter___redArg(lean_object* v_x_37_, lean_object* v_h__1_38_, lean_object* v_h__2_39_){
_start:
{
if (lean_obj_tag(v_x_37_) == 0)
{
lean_object* v___x_40_; lean_object* v___x_41_; 
lean_dec(v_h__2_39_);
v___x_40_ = lean_box(0);
v___x_41_ = lean_apply_1(v_h__1_38_, v___x_40_);
return v___x_41_;
}
else
{
lean_object* v_head_42_; lean_object* v_tail_43_; lean_object* v___x_44_; 
lean_dec(v_h__1_38_);
v_head_42_ = lean_ctor_get(v_x_37_, 0);
lean_inc(v_head_42_);
v_tail_43_ = lean_ctor_get(v_x_37_, 1);
lean_inc(v_tail_43_);
lean_dec_ref_known(v_x_37_, 2);
v___x_44_ = lean_apply_2(v_h__2_39_, v_head_42_, v_tail_43_);
return v___x_44_;
}
}
}
LEAN_EXPORT lean_object* lp_mul__op___private_MulOp_MulCore_0__MulOp_indices_match__1_splitter(lean_object* v_motive_45_, lean_object* v_x_46_, lean_object* v_h__1_47_, lean_object* v_h__2_48_){
_start:
{
if (lean_obj_tag(v_x_46_) == 0)
{
lean_object* v___x_49_; lean_object* v___x_50_; 
lean_dec(v_h__2_48_);
v___x_49_ = lean_box(0);
v___x_50_ = lean_apply_1(v_h__1_47_, v___x_49_);
return v___x_50_;
}
else
{
lean_object* v_head_51_; lean_object* v_tail_52_; lean_object* v___x_53_; 
lean_dec(v_h__1_47_);
v_head_51_ = lean_ctor_get(v_x_46_, 0);
lean_inc(v_head_51_);
v_tail_52_ = lean_ctor_get(v_x_46_, 1);
lean_inc(v_tail_52_);
lean_dec_ref_known(v_x_46_, 2);
v___x_53_ = lean_apply_2(v_h__2_48_, v_head_51_, v_tail_52_);
return v___x_53_;
}
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_mulAt___redArg(lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_tA_56_, lean_object* v_tB_57_, lean_object* v_i_58_){
_start:
{
lean_object* v___y_60_; lean_object* v_shape_67_; lean_object* v___x_68_; lean_object* v___x_69_; 
v_shape_67_ = lean_ctor_get(v_tA_56_, 0);
lean_inc(v_shape_67_);
lean_inc(v_i_58_);
v___x_68_ = lp_mul__op_MulOp_bcastIdx(v_i_58_, v_shape_67_);
v___x_69_ = lp_mul__op_MulOp_Tensor_read___redArg(v_tA_56_, v___x_68_);
lean_dec(v___x_68_);
lean_dec_ref(v_tA_56_);
if (lean_obj_tag(v___x_69_) == 0)
{
lean_inc(v_inst_55_);
v___y_60_ = v_inst_55_;
goto v___jp_59_;
}
else
{
lean_object* v_val_70_; 
v_val_70_ = lean_ctor_get(v___x_69_, 0);
lean_inc(v_val_70_);
lean_dec_ref_known(v___x_69_, 1);
v___y_60_ = v_val_70_;
goto v___jp_59_;
}
v___jp_59_:
{
lean_object* v_shape_61_; lean_object* v___x_62_; lean_object* v___x_63_; 
v_shape_61_ = lean_ctor_get(v_tB_57_, 0);
lean_inc(v_shape_61_);
v___x_62_ = lp_mul__op_MulOp_bcastIdx(v_i_58_, v_shape_61_);
v___x_63_ = lp_mul__op_MulOp_Tensor_read___redArg(v_tB_57_, v___x_62_);
lean_dec(v___x_62_);
lean_dec_ref(v_tB_57_);
if (lean_obj_tag(v___x_63_) == 0)
{
lean_object* v___x_64_; 
v___x_64_ = lean_apply_2(v_inst_54_, v___y_60_, v_inst_55_);
return v___x_64_;
}
else
{
lean_object* v_val_65_; lean_object* v___x_66_; 
lean_dec(v_inst_55_);
v_val_65_ = lean_ctor_get(v___x_63_, 0);
lean_inc(v_val_65_);
lean_dec_ref_known(v___x_63_, 1);
v___x_66_ = lean_apply_2(v_inst_54_, v___y_60_, v_val_65_);
return v___x_66_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_mulAt(lean_object* v_00_u03b1_71_, lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_tA_74_, lean_object* v_tB_75_, lean_object* v_i_76_){
_start:
{
lean_object* v___x_77_; 
v___x_77_ = lp_mul__op_MulOp_mulAt___redArg(v_inst_72_, v_inst_73_, v_tA_74_, v_tB_75_, v_i_76_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_mulData___redArg(lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_tA_80_, lean_object* v_tB_81_, lean_object* v_sc_82_){
_start:
{
lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_83_ = lean_alloc_closure((void*)(lp_mul__op_MulOp_mulAt), 6, 5);
lean_closure_set(v___x_83_, 0, lean_box(0));
lean_closure_set(v___x_83_, 1, v_inst_78_);
lean_closure_set(v___x_83_, 2, v_inst_79_);
lean_closure_set(v___x_83_, 3, v_tA_80_);
lean_closure_set(v___x_83_, 4, v_tB_81_);
lean_inc(v_sc_82_);
v___x_84_ = lp_mul__op_MulOp_indices(v_sc_82_);
v___x_85_ = lean_box(0);
v___x_86_ = l_List_mapTR_loop___redArg(v___x_83_, v___x_84_, v___x_85_);
v___x_87_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_87_, 0, v_sc_82_);
lean_ctor_set(v___x_87_, 1, v___x_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_mulData(lean_object* v_00_u03b1_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_tA_91_, lean_object* v_tB_92_, lean_object* v_sc_93_){
_start:
{
lean_object* v___x_94_; 
v___x_94_ = lp_mul__op_MulOp_mulData___redArg(v_inst_89_, v_inst_90_, v_tA_91_, v_tB_92_, v_sc_93_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_mul___redArg(lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_tA_97_, lean_object* v_tB_98_){
_start:
{
lean_object* v_shape_99_; lean_object* v_shape_100_; lean_object* v___x_101_; 
v_shape_99_ = lean_ctor_get(v_tA_97_, 0);
v_shape_100_ = lean_ctor_get(v_tB_98_, 0);
lean_inc(v_shape_100_);
lean_inc(v_shape_99_);
v___x_101_ = lp_mul__op_MulOp_bcastShape(v_shape_99_, v_shape_100_);
if (lean_obj_tag(v___x_101_) == 0)
{
lean_object* v___x_102_; 
lean_dec_ref(v_tB_98_);
lean_dec_ref(v_tA_97_);
lean_dec(v_inst_96_);
lean_dec(v_inst_95_);
v___x_102_ = lean_box(0);
return v___x_102_;
}
else
{
lean_object* v_val_103_; lean_object* v___x_105_; uint8_t v_isShared_106_; uint8_t v_isSharedCheck_111_; 
v_val_103_ = lean_ctor_get(v___x_101_, 0);
v_isSharedCheck_111_ = !lean_is_exclusive(v___x_101_);
if (v_isSharedCheck_111_ == 0)
{
v___x_105_ = v___x_101_;
v_isShared_106_ = v_isSharedCheck_111_;
goto v_resetjp_104_;
}
else
{
lean_inc(v_val_103_);
lean_dec(v___x_101_);
v___x_105_ = lean_box(0);
v_isShared_106_ = v_isSharedCheck_111_;
goto v_resetjp_104_;
}
v_resetjp_104_:
{
lean_object* v___x_107_; lean_object* v___x_109_; 
v___x_107_ = lp_mul__op_MulOp_mulData___redArg(v_inst_95_, v_inst_96_, v_tA_97_, v_tB_98_, v_val_103_);
if (v_isShared_106_ == 0)
{
lean_ctor_set(v___x_105_, 0, v___x_107_);
v___x_109_ = v___x_105_;
goto v_reusejp_108_;
}
else
{
lean_object* v_reuseFailAlloc_110_; 
v_reuseFailAlloc_110_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_110_, 0, v___x_107_);
v___x_109_ = v_reuseFailAlloc_110_;
goto v_reusejp_108_;
}
v_reusejp_108_:
{
return v___x_109_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_mul(lean_object* v_00_u03b1_112_, lean_object* v_inst_113_, lean_object* v_inst_114_, lean_object* v_tA_115_, lean_object* v_tB_116_){
_start:
{
lean_object* v___x_117_; 
v___x_117_ = lp_mul__op_MulOp_mul___redArg(v_inst_113_, v_inst_114_, v_tA_115_, v_tB_116_);
return v___x_117_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mul__op_MulOp_Broadcast(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mul__op_MulOp_MulCore(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mul__op_MulOp_Broadcast(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
