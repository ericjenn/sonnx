// Lean compiler output
// Module: MulOp
// Imports: public import Init public meta import Init public import MulOp.Tensor public import MulOp.Broadcast public import MulOp.MulCore public import MulOp.IntTypes public import MulOp.FloatTypes public import MulOp.RealTypes public import MulOp.Examples
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
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mul__op_MulOp_Tensor(uint8_t builtin);
lean_object* initialize_mul__op_MulOp_Broadcast(uint8_t builtin);
lean_object* initialize_mul__op_MulOp_MulCore(uint8_t builtin);
lean_object* initialize_mul__op_MulOp_IntTypes(uint8_t builtin);
lean_object* initialize_mul__op_MulOp_FloatTypes(uint8_t builtin);
lean_object* initialize_mul__op_MulOp_RealTypes(uint8_t builtin);
lean_object* initialize_mul__op_MulOp_Examples(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mul__op_MulOp(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mul__op_MulOp_Tensor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mul__op_MulOp_Broadcast(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mul__op_MulOp_MulCore(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mul__op_MulOp_IntTypes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mul__op_MulOp_FloatTypes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mul__op_MulOp_RealTypes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mul__op_MulOp_Examples(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
