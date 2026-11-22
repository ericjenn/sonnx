#include "sonnx_kernels.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

static int g_fail = 0;

static int feq32(float e, float g) {
  if (isnan(e)) return isnan(g) != 0;
  if (e == 0.0f) return (g == 0.0f) && ((signbit(e) != 0) == (signbit(g) != 0));
  return e == g;
}

static int feq64(double e, double g) {
  if (isnan(e)) return isnan(g) != 0;
  if (e == 0.0) return (g == 0.0) && ((signbit(e) != 0) == (signbit(g) != 0));
  return e == g;
}

static void check_f32(const char *name, const float *exp, const float *got, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (!feq32(exp[i], got[i])) {
      printf("FAIL %s: expected %a, got %a (index %zu)\n", name,
             (double)exp[i], (double)got[i], i);
      g_fail++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_f64(const char *name, const double *exp, const double *got, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (!feq64(exp[i], got[i])) {
      printf("FAIL %s: expected %a, got %a (index %zu)\n", name,
             exp[i], got[i], i);
      g_fail++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_sint(const char *name, const int64_t *exp, const int64_t *got, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (exp[i] != got[i]) {
      printf("FAIL %s: expected %lld, got %lld (index %zu)\n", name,
             (long long)exp[i], (long long)got[i], i);
      g_fail++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_uint(const char *name, const uint64_t *exp, const uint64_t *got, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (exp[i] != got[i]) {
      printf("FAIL %s: expected %llu, got %llu (index %zu)\n", name,
             (unsigned long long)exp[i], (unsigned long long)got[i], i);
      g_fail++;
      return;
    }
  }
  printf("ok %s\n", name);
}

int main(void) {
  /* ---------------- f32 ---------------- */

  /* exact sums, including a zero made of opposite signs */
  {
    const size_t sh[1] = {3};
    float ad[3] = {1.5f, 2.25f, -3.5f};
    float bd[3] = {0.25f, -0.25f, 3.5f};
    float cd[3] = {0.0f, 0.0f, 0.0f};
    const float exp[3] = {1.75f, 2.0f, 0.0f};
    sonnx_tensor a = {.data = ad, .ndim = 1, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 1, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 1, .shape = sh};
    sonnx_add_f32(&a, &b, &c);
    check_f32("f32_basic", exp, cd, 3);
  }

  /* multi-element 2-D shape, loop bounds */
  {
    const size_t sh[2] = {2, 3};
    float ad[6] = {1.5f, -2.5f, 0.25f, 4.0f, -0.5f, 8.0f};
    float bd[6] = {0.5f, -2.5f, 0.75f, -3.5f, 0.5f, -7.5f};
    float cd[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const float exp[6] = {2.0f, -5.0f, 1.0f, 0.5f, 0.0f, 0.5f};
    sonnx_tensor a = {.data = ad, .ndim = 2, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 2, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 2, .shape = sh};
    sonnx_add_f32(&a, &b, &c);
    check_f32("f32_2x3", exp, cd, 6);
  }

  /* infinities and NaN */
  {
    const size_t sh[1] = {4};
    float ad[4] = {INFINITY, -INFINITY, 1.0f, NAN};
    float bd[4] = {-INFINITY, -INFINITY, -INFINITY, 1.0f};
    float cd[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float exp[4] = {NAN, -INFINITY, -INFINITY, NAN};
    sonnx_tensor a = {.data = ad, .ndim = 1, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 1, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 1, .shape = sh};
    sonnx_add_f32(&a, &b, &c);
    check_f32("f32_inf_nan", exp, cd, 4);
  }

  /* signed zeros */
  {
    const size_t sh[1] = {3};
    float ad[3] = {0.0f, -0.0f, 0.0f};
    float bd[3] = {-0.0f, -0.0f, 0.0f};
    float cd[3] = {1.0f, 1.0f, 1.0f};
    const float exp[3] = {0.0f, -0.0f, 0.0f};
    sonnx_tensor a = {.data = ad, .ndim = 1, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 1, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 1, .shape = sh};
    sonnx_add_f32(&a, &b, &c);
    check_f32("f32_signed_zeros", exp, cd, 3);
  }

  /* roundTiesToEven, subnormal result and overflow to infinity */
  {
    const size_t sh[1] = {4};
    float ad[4] = {1.0f, 1.0f, 0x1p-149f, 0x1.fffffep127f};
    float bd[4] = {0x1p-24f, 0x1.8p-24f, 0x1p-149f, 0x1.fffffep127f};
    float cd[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float exp[4] = {1.0f, 0x1.000002p0f, 0x1p-148f, INFINITY};
    sonnx_tensor a = {.data = ad, .ndim = 1, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 1, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 1, .shape = sh};
    sonnx_add_f32(&a, &b, &c);
    check_f32("f32_rounding_subnormal_overflow", exp, cd, 4);
  }

  /* empty result: the kernel must write nothing */
  {
    const size_t sh[2] = {0, 2};
    float ad[1] = {1.5f};
    float bd[1] = {2.5f};
    float cd[2] = {7.5f, 7.5f};
    const float exp[2] = {7.5f, 7.5f};
    sonnx_tensor a = {.data = ad, .ndim = 2, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 2, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 2, .shape = sh};
    sonnx_add_f32(&a, &b, &c);
    check_f32("f32_zero_sized_dim", exp, cd, 2);
  }

  /* ---------------- f64 ---------------- */

  /* float example 1, replayed exactly */
  {
    const size_t sh[2] = {3, 2};
    double ad[6] = {3.0, 4.5, 16.0, 1.0, 25.5, 24.25};
    double bd[6] = {3.0, 2.0, 4.0, 0.0, 5.0, 4.0};
    double cd[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const double exp[6] = {6.0, 6.5, 20.0, 1.0, 30.5, 28.25};
    sonnx_tensor a = {.data = ad, .ndim = 2, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 2, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 2, .shape = sh};
    sonnx_add_f64(&a, &b, &c);
    check_f64("f64_float_example1", exp, cd, 6);
  }

  /* float example 2, replayed exactly */
  {
    const size_t sh[1] = {4};
    double ad[4] = {1.0, INFINITY, 0.0, -INFINITY};
    double bd[4] = {1.0, -INFINITY, -0.0, -INFINITY};
    double cd[4] = {0.0, 0.0, 0.0, 0.0};
    const double exp[4] = {2.0, (double)NAN, 0.0, -INFINITY};
    sonnx_tensor a = {.data = ad, .ndim = 1, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 1, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 1, .shape = sh};
    sonnx_add_f64(&a, &b, &c);
    check_f64("f64_float_example2", exp, cd, 4);
  }

  /* float example 3, replayed exactly */
  {
    const size_t sh[1] = {3};
    double ad[3] = {0.1, 1.0, 1.0};
    double bd[3] = {0.2, 2.0, -1.0};
    double cd[3] = {0.0, 0.0, 0.0};
    const double exp[3] = {0.30000000000000004, 3.0, 0.0};
    sonnx_tensor a = {.data = ad, .ndim = 1, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 1, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 1, .shape = sh};
    sonnx_add_f64(&a, &b, &c);
    check_f64("f64_float_example3", exp, cd, 3);
  }

  /* negative zeros, overflow and a cancelling pair at the top of the range */
  {
    const size_t sh[1] = {4};
    double ad[4] = {-0.0, 0x1.fffffffffffffp1023, 0x1.fffffffffffffp1023,
                    -0x1.fffffffffffffp1023};
    double bd[4] = {-0.0, 0x1.fffffffffffffp1023, -0x1.fffffffffffffp1023,
                    -0x1.fffffffffffffp1023};
    double cd[4] = {1.0, 0.0, 1.0, 0.0};
    const double exp[4] = {-0.0, INFINITY, 0.0, -INFINITY};
    sonnx_tensor a = {.data = ad, .ndim = 1, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 1, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 1, .shape = sh};
    sonnx_add_f64(&a, &b, &c);
    check_f64("f64_zeros_overflow_cancel", exp, cd, 4);
  }

  /* rank-0 tensors: one element, no shape */
  {
    double ad[1] = {1.5};
    double bd[1] = {2.25};
    double cd[1] = {0.0};
    const double exp[1] = {3.75};
    sonnx_tensor a = {.data = ad, .ndim = 0, .shape = NULL};
    sonnx_tensor b = {.data = bd, .ndim = 0, .shape = NULL};
    sonnx_tensor c = {.data = cd, .ndim = 0, .shape = NULL};
    sonnx_add_f64(&a, &b, &c);
    check_f64("f64_rank0", exp, cd, 1);
  }

  /* ---------------- signed integers ---------------- */

  /* int example 1, replayed exactly */
  {
    const size_t sh[1] = {3};
    int8_t ad[3] = {-6, 100, -100};
    int8_t bd[3] = {-3, 100, -100};
    int8_t cd[3] = {0, 0, 0};
    const int64_t exp[3] = {-9, -56, 56};
    int64_t got[3];
    sonnx_tensor a = {.data = ad, .ndim = 1, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 1, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 1, .shape = sh};
    sonnx_add_i8(&a, &b, &c);
    for (size_t i = 0; i < 3; i++) got[i] = (int64_t)cd[i];
    check_sint("i8_example1", exp, got, 3);
  }

  /* i8 range boundaries, one step outside wraps */
  {
    const size_t sh[1] = {4};
    int8_t ad[4] = {INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN};
    int8_t bd[4] = {-1, 1, INT8_MAX, INT8_MIN};
    int8_t cd[4] = {0, 0, 0, 0};
    const int64_t exp[4] = {127, -128, -2, 0};
    int64_t got[4];
    sonnx_tensor a = {.data = ad, .ndim = 1, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 1, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 1, .shape = sh};
    sonnx_add_i8(&a, &b, &c);
    for (size_t i = 0; i < 4; i++) got[i] = (int64_t)cd[i];
    check_sint("i8_bounds", exp, got, 4);
  }

  /* i16 boundaries */
  {
    const size_t sh[1] = {3};
    int16_t ad[3] = {INT16_MIN, INT16_MAX, 1000};
    int16_t bd[3] = {-1, 1, 32000};
    int16_t cd[3] = {0, 0, 0};
    const int64_t exp[3] = {32767, -32768, -32536};
    int64_t got[3];
    sonnx_tensor a = {.data = ad, .ndim = 1, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 1, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 1, .shape = sh};
    sonnx_add_i16(&a, &b, &c);
    for (size_t i = 0; i < 3; i++) got[i] = (int64_t)cd[i];
    check_sint("i16_bounds", exp, got, 3);
  }

  /* i32 boundaries */
  {
    const size_t sh[1] = {3};
    int32_t ad[3] = {INT32_MIN, INT32_MAX, 1};
    int32_t bd[3] = {-1, 1, INT32_MAX};
    int32_t cd[3] = {0, 0, 0};
    const int64_t exp[3] = {INT32_MAX, INT32_MIN, INT32_MIN};
    int64_t got[3];
    sonnx_tensor a = {.data = ad, .ndim = 1, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 1, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 1, .shape = sh};
    sonnx_add_i32(&a, &b, &c);
    for (size_t i = 0; i < 3; i++) got[i] = (int64_t)cd[i];
    check_sint("i32_bounds", exp, got, 3);
  }

  /* i32 over a multi-element 2-D shape */
  {
    const size_t sh[2] = {2, 2};
    int32_t ad[4] = {1, -1, 2000000000, 0};
    int32_t bd[4] = {2, 1, 2000000000, -5};
    int32_t cd[4] = {0, 0, 0, 0};
    const int64_t exp[4] = {3, 0, -294967296, -5};
    int64_t got[4];
    sonnx_tensor a = {.data = ad, .ndim = 2, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 2, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 2, .shape = sh};
    sonnx_add_i32(&a, &b, &c);
    for (size_t i = 0; i < 4; i++) got[i] = (int64_t)cd[i];
    check_sint("i32_2x2", exp, got, 4);
  }

  /* empty i32 result */
  {
    const size_t sh[2] = {2, 0};
    int32_t ad[1] = {5};
    int32_t bd[1] = {7};
    int32_t cd[1] = {123};
    const int64_t exp[1] = {123};
    int64_t got[1];
    sonnx_tensor a = {.data = ad, .ndim = 2, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 2, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 2, .shape = sh};
    sonnx_add_i32(&a, &b, &c);
    got[0] = (int64_t)cd[0];
    check_sint("i32_zero_sized_dim", exp, got, 1);
  }

  /* i64 boundaries */
  {
    const size_t sh[1] = {3};
    int64_t ad[3] = {INT64_MIN, INT64_MAX, 1};
    int64_t bd[3] = {-1, 1, INT64_MAX};
    int64_t cd[3] = {0, 0, 0};
    const int64_t exp[3] = {INT64_MAX, INT64_MIN, INT64_MIN};
    int64_t got[3];
    sonnx_tensor a = {.data = ad, .ndim = 1, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 1, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 1, .shape = sh};
    sonnx_add_i64(&a, &b, &c);
    for (size_t i = 0; i < 3; i++) got[i] = cd[i];
    check_sint("i64_bounds", exp, got, 3);
  }

  /* ---------------- unsigned integers ---------------- */

  /* uint example 1, replayed exactly */
  {
    const size_t sh[1] = {3};
    uint8_t ad[3] = {6, 200, 35};
    uint8_t bd[3] = {3, 100, 5};
    uint8_t cd[3] = {0, 0, 0};
    const uint64_t exp[3] = {9, 44, 40};
    uint64_t got[3];
    sonnx_tensor a = {.data = ad, .ndim = 1, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 1, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 1, .shape = sh};
    sonnx_add_u8(&a, &b, &c);
    for (size_t i = 0; i < 3; i++) got[i] = (uint64_t)cd[i];
    check_uint("u8_example1", exp, got, 3);
  }

  /* u8 boundaries */
  {
    const size_t sh[1] = {4};
    uint8_t ad[4] = {UINT8_MAX, UINT8_MAX, 0, 128};
    uint8_t bd[4] = {1, UINT8_MAX, 0, 128};
    uint8_t cd[4] = {0, 0, 0, 0};
    const uint64_t exp[4] = {0, 254, 0, 0};
    uint64_t got[4];
    sonnx_tensor a = {.data = ad, .ndim = 1, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 1, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 1, .shape = sh};
    sonnx_add_u8(&a, &b, &c);
    for (size_t i = 0; i < 4; i++) got[i] = (uint64_t)cd[i];
    check_uint("u8_bounds", exp, got, 4);
  }

  /* u16 boundaries */
  {
    const size_t sh[1] = {3};
    uint16_t ad[3] = {UINT16_MAX, 1000, 0};
    uint16_t bd[3] = {1, 65000, UINT16_MAX};
    uint16_t cd[3] = {0, 0, 0};
    const uint64_t exp[3] = {0, 464, UINT16_MAX};
    uint64_t got[3];
    sonnx_tensor a = {.data = ad, .ndim = 1, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 1, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 1, .shape = sh};
    sonnx_add_u16(&a, &b, &c);
    for (size_t i = 0; i < 3; i++) got[i] = (uint64_t)cd[i];
    check_uint("u16_bounds", exp, got, 3);
  }

  /* u32 boundaries */
  {
    const size_t sh[1] = {3};
    uint32_t ad[3] = {UINT32_MAX, 1, 0};
    uint32_t bd[3] = {1, UINT32_MAX, UINT32_MAX};
    uint32_t cd[3] = {0, 0, 0};
    const uint64_t exp[3] = {0, 0, UINT32_MAX};
    uint64_t got[3];
    sonnx_tensor a = {.data = ad, .ndim = 1, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 1, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 1, .shape = sh};
    sonnx_add_u32(&a, &b, &c);
    for (size_t i = 0; i < 3; i++) got[i] = (uint64_t)cd[i];
    check_uint("u32_bounds", exp, got, 3);
  }

  /* u64 boundaries */
  {
    const size_t sh[1] = {3};
    uint64_t ad[3] = {UINT64_MAX, 1, 0};
    uint64_t bd[3] = {1, UINT64_MAX, UINT64_MAX};
    uint64_t cd[3] = {0, 0, 0};
    const uint64_t exp[3] = {0, 0, UINT64_MAX};
    uint64_t got[3];
    sonnx_tensor a = {.data = ad, .ndim = 1, .shape = sh};
    sonnx_tensor b = {.data = bd, .ndim = 1, .shape = sh};
    sonnx_tensor c = {.data = cd, .ndim = 1, .shape = sh};
    sonnx_add_u64(&a, &b, &c);
    for (size_t i = 0; i < 3; i++) got[i] = cd[i];
    check_uint("u64_bounds", exp, got, 3);
  }

  if (g_fail != 0) {
    fprintf(stderr, "%d failing case(s)\n", g_fail);
    return 1;
  }
  return 0;
}
