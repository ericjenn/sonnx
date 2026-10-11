// Lean compiler output
// Module: MulOp.Tensor
// Imports: public import Init public meta import Init public import Init
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
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_size(lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_size___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mul__op_MulOp_validIdxB(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_validIdxB___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op___private_MulOp_Tensor_0__MulOp_validIdxB_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op___private_MulOp_Tensor_0__MulOp_validIdxB_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_offset(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_offset___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op___private_MulOp_Tensor_0__MulOp_offset_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op___private_MulOp_Tensor_0__MulOp_offset_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op___private_MulOp_Tensor_0__MulOp_size_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op___private_MulOp_Tensor_0__MulOp_size_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mul__op_MulOp_instInhabitedTensor___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mul__op_MulOp_instInhabitedTensor___closed__0 = (const lean_object*)&lp_mul__op_MulOp_instInhabitedTensor___closed__0_value;
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instInhabitedTensor(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instInhabitedTensor___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_Tensor_read___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_Tensor_read___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_Tensor_read(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_Tensor_read___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mul__op_MulOp_size(lean_object* v_x_1_){
_start:
{
if (lean_obj_tag(v_x_1_) == 0)
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(1u);
return v___x_2_;
}
else
{
lean_object* v_head_3_; lean_object* v_tail_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v_head_3_ = lean_ctor_get(v_x_1_, 0);
v_tail_4_ = lean_ctor_get(v_x_1_, 1);
v___x_5_ = lp_mul__op_MulOp_size(v_tail_4_);
v___x_6_ = lean_nat_mul(v_head_3_, v___x_5_);
lean_dec(v___x_5_);
return v___x_6_;
}
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_size___boxed(lean_object* v_x_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mul__op_MulOp_size(v_x_7_);
lean_dec(v_x_7_);
return v_res_8_;
}
}
LEAN_EXPORT uint8_t lp_mul__op_MulOp_validIdxB(lean_object* v_x_9_, lean_object* v_x_10_){
_start:
{
if (lean_obj_tag(v_x_9_) == 0)
{
if (lean_obj_tag(v_x_10_) == 0)
{
uint8_t v___x_11_; 
v___x_11_ = 1;
return v___x_11_;
}
else
{
uint8_t v___x_12_; 
v___x_12_ = 0;
return v___x_12_;
}
}
else
{
if (lean_obj_tag(v_x_10_) == 0)
{
uint8_t v___x_13_; 
v___x_13_ = 0;
return v___x_13_;
}
else
{
lean_object* v_head_14_; lean_object* v_tail_15_; lean_object* v_head_16_; lean_object* v_tail_17_; uint8_t v___x_18_; 
v_head_14_ = lean_ctor_get(v_x_9_, 0);
v_tail_15_ = lean_ctor_get(v_x_9_, 1);
v_head_16_ = lean_ctor_get(v_x_10_, 0);
v_tail_17_ = lean_ctor_get(v_x_10_, 1);
v___x_18_ = lean_nat_dec_lt(v_head_16_, v_head_14_);
if (v___x_18_ == 0)
{
return v___x_18_;
}
else
{
v_x_9_ = v_tail_15_;
v_x_10_ = v_tail_17_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_validIdxB___boxed(lean_object* v_x_20_, lean_object* v_x_21_){
_start:
{
uint8_t v_res_22_; lean_object* v_r_23_; 
v_res_22_ = lp_mul__op_MulOp_validIdxB(v_x_20_, v_x_21_);
lean_dec(v_x_21_);
lean_dec(v_x_20_);
v_r_23_ = lean_box(v_res_22_);
return v_r_23_;
}
}
LEAN_EXPORT lean_object* lp_mul__op___private_MulOp_Tensor_0__MulOp_validIdxB_match__1_splitter___redArg(lean_object* v_x_24_, lean_object* v_x_25_, lean_object* v_h__1_26_, lean_object* v_h__2_27_, lean_object* v_h__3_28_, lean_object* v_h__4_29_){
_start:
{
if (lean_obj_tag(v_x_24_) == 0)
{
lean_dec(v_h__4_29_);
lean_dec(v_h__3_28_);
if (lean_obj_tag(v_x_25_) == 0)
{
lean_object* v___x_30_; lean_object* v___x_31_; 
lean_dec(v_h__2_27_);
v___x_30_ = lean_box(0);
v___x_31_ = lean_apply_1(v_h__1_26_, v___x_30_);
return v___x_31_;
}
else
{
lean_object* v_head_32_; lean_object* v_tail_33_; lean_object* v___x_34_; 
lean_dec(v_h__1_26_);
v_head_32_ = lean_ctor_get(v_x_25_, 0);
lean_inc(v_head_32_);
v_tail_33_ = lean_ctor_get(v_x_25_, 1);
lean_inc(v_tail_33_);
lean_dec_ref_known(v_x_25_, 2);
v___x_34_ = lean_apply_2(v_h__2_27_, v_head_32_, v_tail_33_);
return v___x_34_;
}
}
else
{
lean_dec(v_h__2_27_);
lean_dec(v_h__1_26_);
if (lean_obj_tag(v_x_25_) == 0)
{
lean_object* v_head_35_; lean_object* v_tail_36_; lean_object* v___x_37_; 
lean_dec(v_h__4_29_);
v_head_35_ = lean_ctor_get(v_x_24_, 0);
lean_inc(v_head_35_);
v_tail_36_ = lean_ctor_get(v_x_24_, 1);
lean_inc(v_tail_36_);
lean_dec_ref_known(v_x_24_, 2);
v___x_37_ = lean_apply_2(v_h__3_28_, v_head_35_, v_tail_36_);
return v___x_37_;
}
else
{
lean_object* v_head_38_; lean_object* v_tail_39_; lean_object* v_head_40_; lean_object* v_tail_41_; lean_object* v___x_42_; 
lean_dec(v_h__3_28_);
v_head_38_ = lean_ctor_get(v_x_24_, 0);
lean_inc(v_head_38_);
v_tail_39_ = lean_ctor_get(v_x_24_, 1);
lean_inc(v_tail_39_);
lean_dec_ref_known(v_x_24_, 2);
v_head_40_ = lean_ctor_get(v_x_25_, 0);
lean_inc(v_head_40_);
v_tail_41_ = lean_ctor_get(v_x_25_, 1);
lean_inc(v_tail_41_);
lean_dec_ref_known(v_x_25_, 2);
v___x_42_ = lean_apply_4(v_h__4_29_, v_head_38_, v_tail_39_, v_head_40_, v_tail_41_);
return v___x_42_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mul__op___private_MulOp_Tensor_0__MulOp_validIdxB_match__1_splitter(lean_object* v_motive_43_, lean_object* v_x_44_, lean_object* v_x_45_, lean_object* v_h__1_46_, lean_object* v_h__2_47_, lean_object* v_h__3_48_, lean_object* v_h__4_49_){
_start:
{
if (lean_obj_tag(v_x_44_) == 0)
{
lean_dec(v_h__4_49_);
lean_dec(v_h__3_48_);
if (lean_obj_tag(v_x_45_) == 0)
{
lean_object* v___x_50_; lean_object* v___x_51_; 
lean_dec(v_h__2_47_);
v___x_50_ = lean_box(0);
v___x_51_ = lean_apply_1(v_h__1_46_, v___x_50_);
return v___x_51_;
}
else
{
lean_object* v_head_52_; lean_object* v_tail_53_; lean_object* v___x_54_; 
lean_dec(v_h__1_46_);
v_head_52_ = lean_ctor_get(v_x_45_, 0);
lean_inc(v_head_52_);
v_tail_53_ = lean_ctor_get(v_x_45_, 1);
lean_inc(v_tail_53_);
lean_dec_ref_known(v_x_45_, 2);
v___x_54_ = lean_apply_2(v_h__2_47_, v_head_52_, v_tail_53_);
return v___x_54_;
}
}
else
{
lean_dec(v_h__2_47_);
lean_dec(v_h__1_46_);
if (lean_obj_tag(v_x_45_) == 0)
{
lean_object* v_head_55_; lean_object* v_tail_56_; lean_object* v___x_57_; 
lean_dec(v_h__4_49_);
v_head_55_ = lean_ctor_get(v_x_44_, 0);
lean_inc(v_head_55_);
v_tail_56_ = lean_ctor_get(v_x_44_, 1);
lean_inc(v_tail_56_);
lean_dec_ref_known(v_x_44_, 2);
v___x_57_ = lean_apply_2(v_h__3_48_, v_head_55_, v_tail_56_);
return v___x_57_;
}
else
{
lean_object* v_head_58_; lean_object* v_tail_59_; lean_object* v_head_60_; lean_object* v_tail_61_; lean_object* v___x_62_; 
lean_dec(v_h__3_48_);
v_head_58_ = lean_ctor_get(v_x_44_, 0);
lean_inc(v_head_58_);
v_tail_59_ = lean_ctor_get(v_x_44_, 1);
lean_inc(v_tail_59_);
lean_dec_ref_known(v_x_44_, 2);
v_head_60_ = lean_ctor_get(v_x_45_, 0);
lean_inc(v_head_60_);
v_tail_61_ = lean_ctor_get(v_x_45_, 1);
lean_inc(v_tail_61_);
lean_dec_ref_known(v_x_45_, 2);
v___x_62_ = lean_apply_4(v_h__4_49_, v_head_58_, v_tail_59_, v_head_60_, v_tail_61_);
return v___x_62_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_offset(lean_object* v_x_63_, lean_object* v_x_64_){
_start:
{
if (lean_obj_tag(v_x_63_) == 0)
{
lean_object* v___x_65_; 
v___x_65_ = lean_unsigned_to_nat(0u);
return v___x_65_;
}
else
{
if (lean_obj_tag(v_x_64_) == 0)
{
lean_object* v___x_66_; 
v___x_66_ = lean_unsigned_to_nat(0u);
return v___x_66_;
}
else
{
lean_object* v_tail_67_; lean_object* v_head_68_; lean_object* v_tail_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; 
v_tail_67_ = lean_ctor_get(v_x_63_, 1);
v_head_68_ = lean_ctor_get(v_x_64_, 0);
v_tail_69_ = lean_ctor_get(v_x_64_, 1);
v___x_70_ = lp_mul__op_MulOp_size(v_tail_67_);
v___x_71_ = lean_nat_mul(v_head_68_, v___x_70_);
lean_dec(v___x_70_);
v___x_72_ = lp_mul__op_MulOp_offset(v_tail_67_, v_tail_69_);
v___x_73_ = lean_nat_add(v___x_71_, v___x_72_);
lean_dec(v___x_72_);
lean_dec(v___x_71_);
return v___x_73_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_offset___boxed(lean_object* v_x_74_, lean_object* v_x_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_mul__op_MulOp_offset(v_x_74_, v_x_75_);
lean_dec(v_x_75_);
lean_dec(v_x_74_);
return v_res_76_;
}
}
LEAN_EXPORT lean_object* lp_mul__op___private_MulOp_Tensor_0__MulOp_offset_match__1_splitter___redArg(lean_object* v_x_77_, lean_object* v_x_78_, lean_object* v_h__1_79_, lean_object* v_h__2_80_, lean_object* v_h__3_81_){
_start:
{
if (lean_obj_tag(v_x_77_) == 0)
{
lean_object* v___x_82_; 
lean_dec(v_h__3_81_);
lean_dec(v_h__2_80_);
v___x_82_ = lean_apply_1(v_h__1_79_, v_x_78_);
return v___x_82_;
}
else
{
lean_dec(v_h__1_79_);
if (lean_obj_tag(v_x_78_) == 0)
{
lean_object* v_head_83_; lean_object* v_tail_84_; lean_object* v___x_85_; 
lean_dec(v_h__3_81_);
v_head_83_ = lean_ctor_get(v_x_77_, 0);
lean_inc(v_head_83_);
v_tail_84_ = lean_ctor_get(v_x_77_, 1);
lean_inc(v_tail_84_);
lean_dec_ref_known(v_x_77_, 2);
v___x_85_ = lean_apply_2(v_h__2_80_, v_head_83_, v_tail_84_);
return v___x_85_;
}
else
{
lean_object* v_head_86_; lean_object* v_tail_87_; lean_object* v_head_88_; lean_object* v_tail_89_; lean_object* v___x_90_; 
lean_dec(v_h__2_80_);
v_head_86_ = lean_ctor_get(v_x_77_, 0);
lean_inc(v_head_86_);
v_tail_87_ = lean_ctor_get(v_x_77_, 1);
lean_inc(v_tail_87_);
lean_dec_ref_known(v_x_77_, 2);
v_head_88_ = lean_ctor_get(v_x_78_, 0);
lean_inc(v_head_88_);
v_tail_89_ = lean_ctor_get(v_x_78_, 1);
lean_inc(v_tail_89_);
lean_dec_ref_known(v_x_78_, 2);
v___x_90_ = lean_apply_4(v_h__3_81_, v_head_86_, v_tail_87_, v_head_88_, v_tail_89_);
return v___x_90_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mul__op___private_MulOp_Tensor_0__MulOp_offset_match__1_splitter(lean_object* v_motive_91_, lean_object* v_x_92_, lean_object* v_x_93_, lean_object* v_h__1_94_, lean_object* v_h__2_95_, lean_object* v_h__3_96_){
_start:
{
if (lean_obj_tag(v_x_92_) == 0)
{
lean_object* v___x_97_; 
lean_dec(v_h__3_96_);
lean_dec(v_h__2_95_);
v___x_97_ = lean_apply_1(v_h__1_94_, v_x_93_);
return v___x_97_;
}
else
{
lean_dec(v_h__1_94_);
if (lean_obj_tag(v_x_93_) == 0)
{
lean_object* v_head_98_; lean_object* v_tail_99_; lean_object* v___x_100_; 
lean_dec(v_h__3_96_);
v_head_98_ = lean_ctor_get(v_x_92_, 0);
lean_inc(v_head_98_);
v_tail_99_ = lean_ctor_get(v_x_92_, 1);
lean_inc(v_tail_99_);
lean_dec_ref_known(v_x_92_, 2);
v___x_100_ = lean_apply_2(v_h__2_95_, v_head_98_, v_tail_99_);
return v___x_100_;
}
else
{
lean_object* v_head_101_; lean_object* v_tail_102_; lean_object* v_head_103_; lean_object* v_tail_104_; lean_object* v___x_105_; 
lean_dec(v_h__2_95_);
v_head_101_ = lean_ctor_get(v_x_92_, 0);
lean_inc(v_head_101_);
v_tail_102_ = lean_ctor_get(v_x_92_, 1);
lean_inc(v_tail_102_);
lean_dec_ref_known(v_x_92_, 2);
v_head_103_ = lean_ctor_get(v_x_93_, 0);
lean_inc(v_head_103_);
v_tail_104_ = lean_ctor_get(v_x_93_, 1);
lean_inc(v_tail_104_);
lean_dec_ref_known(v_x_93_, 2);
v___x_105_ = lean_apply_4(v_h__3_96_, v_head_101_, v_tail_102_, v_head_103_, v_tail_104_);
return v___x_105_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mul__op___private_MulOp_Tensor_0__MulOp_size_match__1_splitter___redArg(lean_object* v_x_106_, lean_object* v_h__1_107_, lean_object* v_h__2_108_){
_start:
{
if (lean_obj_tag(v_x_106_) == 0)
{
lean_object* v___x_109_; lean_object* v___x_110_; 
lean_dec(v_h__2_108_);
v___x_109_ = lean_box(0);
v___x_110_ = lean_apply_1(v_h__1_107_, v___x_109_);
return v___x_110_;
}
else
{
lean_object* v_head_111_; lean_object* v_tail_112_; lean_object* v___x_113_; 
lean_dec(v_h__1_107_);
v_head_111_ = lean_ctor_get(v_x_106_, 0);
lean_inc(v_head_111_);
v_tail_112_ = lean_ctor_get(v_x_106_, 1);
lean_inc(v_tail_112_);
lean_dec_ref_known(v_x_106_, 2);
v___x_113_ = lean_apply_2(v_h__2_108_, v_head_111_, v_tail_112_);
return v___x_113_;
}
}
}
LEAN_EXPORT lean_object* lp_mul__op___private_MulOp_Tensor_0__MulOp_size_match__1_splitter(lean_object* v_motive_114_, lean_object* v_x_115_, lean_object* v_h__1_116_, lean_object* v_h__2_117_){
_start:
{
if (lean_obj_tag(v_x_115_) == 0)
{
lean_object* v___x_118_; lean_object* v___x_119_; 
lean_dec(v_h__2_117_);
v___x_118_ = lean_box(0);
v___x_119_ = lean_apply_1(v_h__1_116_, v___x_118_);
return v___x_119_;
}
else
{
lean_object* v_head_120_; lean_object* v_tail_121_; lean_object* v___x_122_; 
lean_dec(v_h__1_116_);
v_head_120_ = lean_ctor_get(v_x_115_, 0);
lean_inc(v_head_120_);
v_tail_121_ = lean_ctor_get(v_x_115_, 1);
lean_inc(v_tail_121_);
lean_dec_ref_known(v_x_115_, 2);
v___x_122_ = lean_apply_2(v_h__2_117_, v_head_120_, v_tail_121_);
return v___x_122_;
}
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instInhabitedTensor(lean_object* v_00_u03b1_125_, lean_object* v_inst_126_){
_start:
{
lean_object* v___x_127_; 
v___x_127_ = ((lean_object*)(lp_mul__op_MulOp_instInhabitedTensor___closed__0));
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_instInhabitedTensor___boxed(lean_object* v_00_u03b1_128_, lean_object* v_inst_129_){
_start:
{
lean_object* v_res_130_; 
v_res_130_ = lp_mul__op_MulOp_instInhabitedTensor(v_00_u03b1_128_, v_inst_129_);
lean_dec(v_inst_129_);
return v_res_130_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_Tensor_read___redArg(lean_object* v_t_131_, lean_object* v_i_132_){
_start:
{
lean_object* v_shape_133_; lean_object* v_data_134_; uint8_t v___x_135_; 
v_shape_133_ = lean_ctor_get(v_t_131_, 0);
v_data_134_ = lean_ctor_get(v_t_131_, 1);
v___x_135_ = lp_mul__op_MulOp_validIdxB(v_shape_133_, v_i_132_);
if (v___x_135_ == 0)
{
lean_object* v___x_136_; 
v___x_136_ = lean_box(0);
return v___x_136_;
}
else
{
lean_object* v___x_137_; lean_object* v___x_138_; 
v___x_137_ = lp_mul__op_MulOp_offset(v_shape_133_, v_i_132_);
v___x_138_ = l_List_get_x3fInternal___redArg(v_data_134_, v___x_137_);
return v___x_138_;
}
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_Tensor_read___redArg___boxed(lean_object* v_t_139_, lean_object* v_i_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_mul__op_MulOp_Tensor_read___redArg(v_t_139_, v_i_140_);
lean_dec(v_i_140_);
lean_dec_ref(v_t_139_);
return v_res_141_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_Tensor_read(lean_object* v_00_u03b1_142_, lean_object* v_t_143_, lean_object* v_i_144_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = lp_mul__op_MulOp_Tensor_read___redArg(v_t_143_, v_i_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mul__op_MulOp_Tensor_read___boxed(lean_object* v_00_u03b1_146_, lean_object* v_t_147_, lean_object* v_i_148_){
_start:
{
lean_object* v_res_149_; 
v_res_149_ = lp_mul__op_MulOp_Tensor_read(v_00_u03b1_146_, v_t_147_, v_i_148_);
lean_dec(v_i_148_);
lean_dec_ref(v_t_147_);
return v_res_149_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mul__op_MulOp_Tensor(uint8_t builtin) {
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
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
