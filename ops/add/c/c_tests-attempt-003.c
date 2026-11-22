#include "sonnx_kernels.h"

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <limits.h>
#include <string.h>

#define MAXN 8

static int failures = 0;

/* Exact bit comparison; NaN is accepted as any NaN, signed zeros differ. */
static int same_f32(float x, float y) {
  uint32_t bx = 0, by = 0;
  if (isnan(y)) {
    return isnan(x) != 0;
  }
  memcpy(&bx, &x, sizeof bx);
  memcpy(&by, &y, sizeof by);
  return bx == by;
}

static int same_f64(double x, double y) {
  uint64_t bx = 0, by = 0;
  if (isnan(y)) {
    return isnan(x) != 0;
  }
  memcpy(&bx, &x, sizeof bx);
  memcpy(&by, &y, sizeof by);
  return bx == by;
}

static void check_f32(const char *nm, const float *got, const float *exp, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (!same_f32(got[i], exp[i])) {
      printf("FAIL %s: expected [%zu]=%a, got [%zu]=%a\n", nm, i, (double)exp[i],
             i, (double)got[i]);
      failures++;
      return;
    }
  }
  printf("ok %s\n", nm);
}

static void check_f64(const char *nm, const double *got, const double *exp, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (!same_f64(got[i], exp[i])) {
      printf("FAIL %s: expected [%zu]=%a, got [%zu]=%a\n", nm, i, exp[i], i, got[i]);
      failures++;
      return;
    }
  }
  printf("ok %s\n", nm);
}

#define DEF_CHECK_SINT(NAME, CTYPE)                                              \
  static void NAME(const char *nm, const CTYPE *got, const CTYPE *exp,           \
                   size_t n) {                                                    \
    for (size_t i = 0; i < n; i++) {                                             \
      if ((long long)got[i] != (long long)exp[i]) {                              \
        printf("FAIL %s: expected [%zu]=%lld, got [%zu]=%lld\n", nm, i,          \
               (long long)exp[i], i, (long long)got[i]);                         \
        failures++;                                                              \
        return;                                                                  \
      }                                                                          \
    }                                                                            \
    printf("ok %s\n", nm);                                                       \
  }

#define DEF_CHECK_UINT(NAME, CTYPE)                                              \
  static void NAME(const char *nm, const CTYPE *got, const CTYPE *exp,           \
                   size_t n) {                                                    \
    for (size_t i = 0; i < n; i++) {                                             \
      if ((unsigned long long)got[i] != (unsigned long long)exp[i]) {            \
        printf("FAIL %s: expected [%zu]=%llu, got [%zu]=%llu\n", nm, i,          \
               (unsigned long long)exp[i], i, (unsigned long long)got[i]);       \
        failures++;                                                              \
        return;                                                                  \
      }                                                                          \
    }                                                                            \
    printf("ok %s\n", nm);                                                       \
  }

DEF_CHECK_SINT(check_i8, int8_t)
DEF_CHECK_SINT(check_i16, int16_t)
DEF_CHECK_SINT(check_i32, int32_t)
DEF_CHECK_SINT(check_i64, int64_t)
DEF_CHECK_UINT(check_u8, uint8_t)
DEF_CHECK_UINT(check_u16, uint16_t)
DEF_CHECK_UINT(check_u32, uint32_t)
DEF_CHECK_UINT(check_u64, uint64_t)

#define DEF_RUNNER(NAME, CTYPE, KERNEL, CHECK)                                   \
  static void NAME(const char *nm, size_t nd, const size_t *shp,                 \
                   const CTYPE *ad, const CTYPE *bd, const CTYPE *ed, size_t n) {\
    CTYPE out[MAXN] = {0};                                                       \
    sonnx_tensor a = {.data = (void *)ad, .ndim = nd, .shape = shp};             \
    sonnx_tensor b = {.data = (void *)bd, .ndim = nd, .shape = shp};             \
    sonnx_tensor c = {.data = (void *)out, .ndim = nd, .shape = shp};            \
    KERNEL(&a, &b, &c);                                                          \
    CHECK(nm, out, ed, n);                                                       \
  }

DEF_RUNNER(run_f32, float, sonnx_add_f32, check_f32)
DEF_RUNNER(run_f64, double, sonnx_add_f64, check_f64)
DEF_RUNNER(run_i8, int8_t, sonnx_add_i8, check_i8)
DEF_RUNNER(run_i16, int16_t, sonnx_add_i16, check_i16)
DEF_RUNNER(run_i32, int32_t, sonnx_add_i32, check_i32)
DEF_RUNNER(run_i64, int64_t, sonnx_add_i64, check_i64)
DEF_RUNNER(run_u8, uint8_t, sonnx_add_u8, check_u8)
DEF_RUNNER(run_u16, uint16_t, sonnx_add_u16, check_u16)
DEF_RUNNER(run_u32, uint32_t, sonnx_add_u32, check_u32)
DEF_RUNNER(run_u64, uint64_t, sonnx_add_u64, check_u64)

int main(void) {
  /* ------------------------- f32 ------------------------- */

  { /* float example 1 (shape {3,2}) */
    const size_t sh[2] = {3, 2};
    const float A[6] = {3.0f, 4.5f, 16.0f, 1.0f, 25.5f, 24.25f};
    const float B[6] = {3.0f, 2.0f, 4.0f, 0.0f, 5.0f, 4.0f};
    const float E[6] = {6.0f, 6.5f, 20.0f, 1.0f, 30.5f, 28.25f};
    run_f32("f32_example1", 2, sh, A, B, E, 6);
  }

  { /* float example 2: infinities, NaN, signed zeros */
    const size_t sh[1] = {4};
    const float A[4] = {1.0f, INFINITY, 0.0f, -INFINITY};
    const float B[4] = {1.0f, -INFINITY, -0.0f, -INFINITY};
    const float E[4] = {2.0f, NAN, 0.0f, -INFINITY};
    run_f32("f32_example2", 1, sh, A, B, E, 4);
  }

  { /* overflow to infinity */
    const size_t sh[1] = {2};
    const float A[2] = {FLT_MAX, -FLT_MAX};
    const float B[2] = {FLT_MAX, -FLT_MAX};
    const float E[2] = {INFINITY, -INFINITY};
    run_f32("f32_overflow", 1, sh, A, B, E, 2);
  }

  { /* subnormal, exact sums below the smallest normal */
    const size_t sh[1] = {2};
    const float A[2] = {0x1p-149f, 0x1p-126f};
    const float B[2] = {0x1p-149f, -0x1p-149f};
    const float E[2] = {0x1p-148f, 0x1.fffffep-127f};
    run_f32("f32_subnormal", 1, sh, A, B, E, 2);
  }

  { /* signed zeros: (-0)+(-0) = -0 ; (-0)+(+0) = +0 */
    const size_t sh[1] = {2};
    const float A[2] = {-0.0f, -0.0f};
    const float B[2] = {-0.0f, 0.0f};
    const float E[2] = {-0.0f, 0.0f};
    run_f32("f32_signed_zero", 1, sh, A, B, E, 2);
  }

  { /* operands that cancel: null sum is +0 */
    const size_t sh[1] = {2};
    const float A[2] = {1.5f, 4.0f};
    const float B[2] = {-1.5f, -4.0f};
    const float E[2] = {0.0f, 0.0f};
    run_f32("f32_cancel_to_pos_zero", 1, sh, A, B, E, 2);
  }

  { /* rank-0 tensor: one element, shape NULL */
    const float A[1] = {1.5f};
    const float B[1] = {2.25f};
    const float E[1] = {3.75f};
    run_f32("f32_rank0", 0, NULL, A, B, E, 1);
  }

  { /* zero-sized dimension: C is empty, nothing may be written */
    const size_t sh[2] = {0, 3};
    float A[1] = {1.0f};
    float B[1] = {2.0f};
    float C[1];
    C[0] = 42.0f;
    sonnx_tensor a = {.data = A, .ndim = 2, .shape = sh};
    sonnx_tensor b = {.data = B, .ndim = 2, .shape = sh};
    sonnx_tensor c = {.data = C, .ndim = 2, .shape = sh};
    sonnx_add_f32(&a, &b, &c);
    if (C[0] == 42.0f) {
      printf("ok f32_zero_dim\n");
    } else {
      printf("FAIL f32_zero_dim: expected [0]=%a, got [0]=%a\n", 42.0, (double)C[0]);
      failures++;
    }
  }

  /* ------------------------- f64 ------------------------- */

  { /* float example 2 in double */
    const size_t sh[1] = {4};
    const double A[4] = {1.0, INFINITY, 0.0, -INFINITY};
    const double B[4] = {1.0, -INFINITY, -0.0, -INFINITY};
    const double E[4] = {2.0, NAN, 0.0, -INFINITY};
    run_f64("f64_example2", 1, sh, A, B, E, 4);
  }

  { /* float example 3: 0.1 + 0.2 rounded, and a null sum */
    const size_t sh[1] = {3};
    const double A[3] = {0.1, 1.0, 1.0};
    const double B[3] = {0.2, 2.0, -1.0};
    const double E[3] = {0.30000000000000004, 3.0, 0.0};
    run_f64("f64_example3", 1, sh, A, B, E, 3);
  }

  { /* overflow to infinity */
    const size_t sh[1] = {2};
    const double A[2] = {DBL_MAX, -DBL_MAX};
    const double B[2] = {DBL_MAX, -DBL_MAX};
    const double E[2] = {INFINITY, -INFINITY};
    run_f64("f64_overflow", 1, sh, A, B, E, 2);
  }

  { /* subnormal results, exact */
    const size_t sh[1] = {2};
    const double A[2] = {0x1p-1074, 0x1p-1022};
    const double B[2] = {0x1p-1074, -0x1p-1074};
    const double E[2] = {0x1p-1073, 0x1.fffffffffffffp-1023};
    run_f64("f64_subnormal", 1, sh, A, B, E, 2);
  }

  { /* signed zeros */
    const size_t sh[1] = {2};
    const double A[2] = {-0.0, -0.0};
    const double B[2] = {-0.0, 0.0};
    const double E[2] = {-0.0, 0.0};
    run_f64("f64_signed_zero", 1, sh, A, B, E, 2);
  }

  { /* roundTiesToEven: 2^53+1 ties to 2^53; 2^53+2 is exact */
    const size_t sh[1] = {2};
    const double A[2] = {0x1p53, 0x1p53};
    const double B[2] = {1.0, 2.0};
    const double E[2] = {0x1p53, 9007199254740994.0};
    run_f64("f64_round_ties_even", 1, sh, A, B, E, 2);
  }

  { /* rank-0 tensor */
    const double A[1] = {1.5};
    const double B[1] = {-0.25};
    const double E[1] = {1.25};
    run_f64("f64_rank0", 0, NULL, A, B, E, 1);
  }

  /* ------------------------- i8 -------------------------- */

  { /* int example 1 */
    const size_t sh[1] = {3};
    const int8_t A[3] = {-6, 100, -100};
    const int8_t B[3] = {-3, 100, -100};
    const int8_t E[3] = {-9, -56, 56};
    run_i8("i8_example1", 1, sh, A, B, E, 3);
  }

  { /* extremes and one step outside the range */
    const size_t sh[1] = {4};
    const int8_t A[4] = {INT8_MAX, INT8_MIN, INT8_MIN, INT8_MAX};
    const int8_t B[4] = {1, -1, INT8_MIN, INT8_MAX};
    const int8_t E[4] = {INT8_MIN, INT8_MAX, 0, -2};
    run_i8("i8_bounds", 1, sh, A, B, E, 4);
  }

  { /* several elements, row-major order preserved */
    const size_t sh[2] = {2, 2};
    const int8_t A[4] = {1, 2, 3, 4};
    const int8_t B[4] = {10, 20, 30, 40};
    const int8_t E[4] = {11, 22, 33, 44};
    run_i8("i8_shape_2x2", 2, sh, A, B, E, 4);
  }

  /* ------------------------- i16 ------------------------- */

  {
    const size_t sh[1] = {5};
    const int16_t A[5] = {INT16_MAX, INT16_MIN, 1000, INT16_MAX, -1};
    const int16_t B[5] = {1, -1, 1000, INT16_MAX, 1};
    const int16_t E[5] = {INT16_MIN, INT16_MAX, 2000, -2, 0};
    run_i16("i16_bounds", 1, sh, A, B, E, 5);
  }

  /* ------------------------- i32 ------------------------- */

  {
    const size_t sh[1] = {4};
    const int32_t A[4] = {INT32_MAX, INT32_MIN, -123456, INT32_MIN};
    const int32_t B[4] = {1, -1, 123456, INT32_MIN};
    const int32_t E[4] = {INT32_MIN, INT32_MAX, 0, 0};
    run_i32("i32_bounds", 1, sh, A, B, E, 4);
  }

  {
    const size_t sh[2] = {2, 3};
    const int32_t A[6] = {INT32_MAX, INT32_MIN, 0, 1, -1, 100};
    const int32_t B[6] = {1, -1, 0, -1, 1, -100};
    const int32_t E[6] = {INT32_MIN, INT32_MAX, 0, 0, 0, 0};
    run_i32("i32_shape_2x3", 2, sh, A, B, E, 6);
  }

  { /* zero-sized dimension */
    const size_t sh[2] = {3, 0};
    int32_t A[1] = {7};
    int32_t B[1] = {9};
    int32_t C[1];
    C[0] = -12345;
    sonnx_tensor a = {.data = A, .ndim = 2, .shape = sh};
    sonnx_tensor b = {.data = B, .ndim = 2, .shape = sh};
    sonnx_tensor c = {.data = C, .ndim = 2, .shape = sh};
    sonnx_add_i32(&a, &b, &c);
    if (C[0] == -12345) {
      printf("ok i32_zero_dim\n");
    } else {
      printf("FAIL i32_zero_dim: expected [0]=%d, got [0]=%d\n", -12345, (int)C[0]);
      failures++;
    }
  }

  /* ------------------------- i64 ------------------------- */

  {
    const size_t sh[2] = {3, 2};
    const int64_t A[6] = {INT64_MAX, INT64_MIN, INT64_MIN, 0, -1, 1234567890123LL};
    const int64_t B[6] = {1, -1, INT64_MIN, 0, 1, -1234567890123LL};
    const int64_t E[6] = {INT64_MIN, INT64_MAX, 0, 0, 0, 0};
    run_i64("i64_bounds", 2, sh, A, B, E, 6);
  }

  /* ------------------------- u8 -------------------------- */

  { /* uint example 1 */
    const size_t sh[1] = {3};
    const uint8_t A[3] = {6, 200, 35};
    const uint8_t B[3] = {3, 100, 5};
    const uint8_t E[3] = {9, 44, 40};
    run_u8("u8_example1", 1, sh, A, B, E, 3);
  }

  {
    const size_t sh[1] = {5};
    const uint8_t A[5] = {UINT8_MAX, 200, UINT8_MAX, 0, 0};
    const uint8_t B[5] = {1, 100, UINT8_MAX, 0, UINT8_MAX};
    const uint8_t E[5] = {0, 44, 254, 0, UINT8_MAX};
    run_u8("u8_bounds", 1, sh, A, B, E, 5);
  }

  /* ------------------------- u16 ------------------------- */

  {
    const size_t sh[1] = {4};
    const uint16_t A[4] = {UINT16_MAX, 40000, 1, 0};
    const uint16_t B[4] = {1, 40000, UINT16_MAX, 0};
    const uint16_t E[4] = {0, 14464, 0, 0};
    run_u16("u16_bounds", 1, sh, A, B, E, 4);
  }

  /* ------------------------- u32 ------------------------- */

  {
    const size_t sh[1] = {4};
    const uint32_t A[4] = {UINT32_MAX, UINT32_MAX, 0, 0};
    const uint32_t B[4] = {1, UINT32_MAX, 0, UINT32_MAX};
    const uint32_t E[4] = {0, UINT32_MAX - 1, 0, UINT32_MAX};
    run_u32("u32_bounds", 1, sh, A, B, E, 4);
  }

  /* ------------------------- u64 ------------------------- */

  {
    const size_t sh[1] = {4};
    const uint64_t A[4] = {UINT64_MAX, 0, UINT64_MAX, 0};
    const uint64_t B[4] = {1, 0, UINT64_MAX, UINT64_MAX};
    const uint64_t E[4] = {0, 0, UINT64_MAX - 1, UINT64_MAX};
    run_u64("u64_bounds", 1, sh, A, B, E, 4);
  }

  if (failures != 0) {
    fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  return 0;
}
