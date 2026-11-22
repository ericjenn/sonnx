/* Standalone tests for the generated Sonnx Sub kernels.
 * Expected values are taken from the specification (examples, IEEE 754 case
 * analysis, modular integer arithmetic), never from the kernel. */

#include "sonnx_kernels.h"
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <limits.h>
#include <string.h>

static int failures = 0;

static float f32_bits(uint32_t b) {
  float f;
  memcpy(&f, &b, sizeof f);
  return f;
}

static double f64_bits(uint64_t b) {
  double d;
  memcpy(&d, &b, sizeof d);
  return d;
}

static int f32_same(float x, float e) {
  uint32_t bx, be;
  if (isnan(e)) {
    return isnan(x) != 0;
  }
  memcpy(&bx, &x, sizeof bx);
  memcpy(&be, &e, sizeof be);
  return bx == be;
}

static int f64_same(double x, double e) {
  uint64_t bx, be;
  if (isnan(e)) {
    return isnan(x) != 0;
  }
  memcpy(&bx, &x, sizeof bx);
  memcpy(&be, &e, sizeof be);
  return bx == be;
}

static void fmt_f32(char *buf, size_t cap, float v) {
  if (isnan(v)) {
    snprintf(buf, cap, "nan");
  } else if (isinf(v)) {
    snprintf(buf, cap, v > 0.0f ? "+inf" : "-inf");
  } else if (v == 0.0f) {
    snprintf(buf, cap, signbit(v) ? "-0" : "+0");
  } else {
    snprintf(buf, cap, "%.9g", (double)v);
  }
}

static void fmt_f64(char *buf, size_t cap, double v) {
  if (isnan(v)) {
    snprintf(buf, cap, "nan");
  } else if (isinf(v)) {
    snprintf(buf, cap, v > 0.0 ? "+inf" : "-inf");
  } else if (v == 0.0) {
    snprintf(buf, cap, signbit(v) ? "-0" : "+0");
  } else {
    snprintf(buf, cap, "%.17g", v);
  }
}

static void check_f32(const char *name, const float *got, const float *exp, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (!f32_same(got[i], exp[i])) {
      char vb[64], eb[96], gb[96];
      fmt_f32(vb, sizeof vb, exp[i]);
      snprintf(eb, sizeof eb, "%s (element %zu)", vb, i);
      fmt_f32(vb, sizeof vb, got[i]);
      snprintf(gb, sizeof gb, "%s (element %zu)", vb, i);
      printf("FAIL %s: expected %s, got %s\n", name, eb, gb);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_f64(const char *name, const double *got, const double *exp, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (!f64_same(got[i], exp[i])) {
      char vb[64], eb[96], gb[96];
      fmt_f64(vb, sizeof vb, exp[i]);
      snprintf(eb, sizeof eb, "%s (element %zu)", vb, i);
      fmt_f64(vb, sizeof vb, got[i]);
      snprintf(gb, sizeof gb, "%s (element %zu)", vb, i);
      printf("FAIL %s: expected %s, got %s\n", name, eb, gb);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_i8(const char *name, const int8_t *got, const int8_t *exp, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (got[i] != exp[i]) {
      printf("FAIL %s: expected %lld, got %lld\n", name,
             (long long)exp[i], (long long)got[i]);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_i16(const char *name, const int16_t *got, const int16_t *exp, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (got[i] != exp[i]) {
      printf("FAIL %s: expected %lld, got %lld\n", name,
             (long long)exp[i], (long long)got[i]);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_i32(const char *name, const int32_t *got, const int32_t *exp, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (got[i] != exp[i]) {
      printf("FAIL %s: expected %lld, got %lld\n", name,
             (long long)exp[i], (long long)got[i]);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_i64(const char *name, const int64_t *got, const int64_t *exp, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (got[i] != exp[i]) {
      printf("FAIL %s: expected %lld, got %lld\n", name,
             (long long)exp[i], (long long)got[i]);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_u8(const char *name, const uint8_t *got, const uint8_t *exp, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (got[i] != exp[i]) {
      printf("FAIL %s: expected %llu, got %llu\n", name,
             (unsigned long long)exp[i], (unsigned long long)got[i]);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_u16(const char *name, const uint16_t *got, const uint16_t *exp, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (got[i] != exp[i]) {
      printf("FAIL %s: expected %llu, got %llu\n", name,
             (unsigned long long)exp[i], (unsigned long long)got[i]);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_u32(const char *name, const uint32_t *got, const uint32_t *exp, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (got[i] != exp[i]) {
      printf("FAIL %s: expected %llu, got %llu\n", name,
             (unsigned long long)exp[i], (unsigned long long)got[i]);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_u64(const char *name, const uint64_t *got, const uint64_t *exp, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (got[i] != exp[i]) {
      printf("FAIL %s: expected %llu, got %llu\n", name,
             (unsigned long long)exp[i], (unsigned long long)got[i]);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

/* ---- f32 cases ------------------------------------------------ */

/* Spec, float example 1: every difference exactly representable. */
static void case_f32_example1(void) {
  const size_t shape[2] = {3, 2};
  float ad[6] = {3.0f, 4.5f, 16.0f, 1.0f, 25.5f, 24.25f};
  float bd[6] = {3.0f, 2.0f, 4.0f, 0.0f, 5.0f, 4.0f};
  float cd[6] = {9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f};
  const float ed[6] = {0.0f, 2.5f, 12.0f, 1.0f, 20.5f, 20.25f};
  sonnx_tensor a = { .data = ad, .ndim = 2, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 2, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 2, .shape = shape };
  sonnx_sub_f32(&a, &b, &c);
  check_f32("f32_example1", cd, ed, 6);
}

/* Spec, float example 2: infinities of the same sign and signed zeros. */
static void case_f32_example2(void) {
  const size_t shape[1] = {4};
  float ad[4] = {1.0f, INFINITY, -0.0f, 0.0f};
  float bd[4] = {1.0f, INFINITY, 0.0f, -0.0f};
  float cd[4] = {9.0f, 9.0f, 9.0f, 9.0f};
  const float ed[4] = {0.0f, NAN, -0.0f, 0.0f};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_f32(&a, &b, &c);
  check_f32("f32_example2", cd, ed, 4);
}

/* Infinite operands of opposite signs give a signed infinity. */
static void case_f32_infinities(void) {
  const size_t shape[1] = {6};
  float ad[6] = {INFINITY, -INFINITY, 5.0f, -5.0f, INFINITY, -INFINITY};
  float bd[6] = {3.0f, 3.0f, INFINITY, -INFINITY, -INFINITY, INFINITY};
  float cd[6] = {9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f};
  const float ed[6] = {INFINITY, -INFINITY, -INFINITY, INFINITY, INFINITY, -INFINITY};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_f32(&a, &b, &c);
  check_f32("f32_infinities", cd, ed, 6);
}

/* NaN operands propagate. */
static void case_f32_nan(void) {
  const size_t shape[1] = {3};
  float ad[3] = {NAN, 1.0f, NAN};
  float bd[3] = {1.0f, NAN, NAN};
  float cd[3] = {9.0f, 9.0f, 9.0f};
  const float ed[3] = {NAN, NAN, NAN};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_f32(&a, &b, &c);
  check_f32("f32_nan", cd, ed, 3);
}

/* Only (-0) - (+0) yields -0; the other null differences yield +0. */
static void case_f32_signed_zero(void) {
  const size_t shape[1] = {5};
  float ad[5] = {-0.0f, 0.0f, 0.0f, -0.0f, 2.5f};
  float bd[5] = {0.0f, -0.0f, 0.0f, -0.0f, 2.5f};
  float cd[5] = {9.0f, 9.0f, 9.0f, 9.0f, 9.0f};
  const float ed[5] = {-0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_f32(&a, &b, &c);
  check_f32("f32_signed_zero", cd, ed, 5);
}

/* Overflow: FLT_MAX - (-FLT_MAX) is +inf with roundTiesToEven. */
static void case_f32_overflow(void) {
  const size_t shape[1] = {2};
  const float fmax = f32_bits(0x7F7FFFFFu);
  float ad[2] = {fmax, -fmax};
  float bd[2] = {-fmax, fmax};
  float cd[2] = {9.0f, 9.0f};
  const float ed[2] = {INFINITY, -INFINITY};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_f32(&a, &b, &c);
  check_f32("f32_overflow", cd, ed, 2);
}

/* Subnormal results are exact multiples of the smallest subnormal. */
static void case_f32_subnormal(void) {
  const size_t shape[1] = {3};
  float ad[3] = {f32_bits(0x00000001u), f32_bits(0x00800000u), f32_bits(0x00000003u)};
  float bd[3] = {f32_bits(0x00000001u), f32_bits(0x00000001u), f32_bits(0x00000001u)};
  float cd[3] = {9.0f, 9.0f, 9.0f};
  const float ed[3] = {f32_bits(0x00000000u), f32_bits(0x007FFFFFu), f32_bits(0x00000002u)};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_f32(&a, &b, &c);
  check_f32("f32_subnormal", cd, ed, 3);
}

/* Non-representable exact differences round to nearest, ties to even:
 * 1 - 2^-30 rounds to 1, and 1 - 2^-25 (an exact tie) rounds to 1. */
static void case_f32_rounding(void) {
  const size_t shape[1] = {2};
  float ad[2] = {1.0f, 1.0f};
  float bd[2] = {f32_bits(0x30800000u), f32_bits(0x33000000u)};
  float cd[2] = {9.0f, 9.0f};
  const float ed[2] = {1.0f, 1.0f};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_f32(&a, &b, &c);
  check_f32("f32_rounding", cd, ed, 2);
}

/* Rank-0 operands: one element, no shape. */
static void case_f32_rank0(void) {
  float ad[1] = {1.5f};
  float bd[1] = {0.25f};
  float cd[1] = {9.0f};
  const float ed[1] = {1.25f};
  sonnx_tensor a = { .data = ad, .ndim = 0, .shape = NULL };
  sonnx_tensor b = { .data = bd, .ndim = 0, .shape = NULL };
  sonnx_tensor c = { .data = cd, .ndim = 0, .shape = NULL };
  sonnx_sub_f32(&a, &b, &c);
  check_f32("f32_rank0", cd, ed, 1);
}

/* A zero-sized dimension makes C empty: nothing is written. */
static void case_f32_zero_size(void) {
  const size_t shape[2] = {2, 0};
  float ad[1] = {1.5f};
  float bd[1] = {0.5f};
  float cd[1] = {42.0f};
  sonnx_tensor a = { .data = ad, .ndim = 2, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 2, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 2, .shape = shape };
  sonnx_sub_f32(&a, &b, &c);
  if (cd[0] == 42.0f) {
    printf("ok f32_zero_size\n");
  } else {
    printf("FAIL f32_zero_size: expected untouched sentinel 42, got %g\n", (double)cd[0]);
    failures++;
  }
}

/* ---- f64 cases ------------------------------------------------ */

/* Spec, float example 3: 0.3 - 0.1 rounds to the nearest double
 * 0x1.9999999999999p-3; 1.0 - 1.0 is +0. */
static void case_f64_example3(void) {
  const size_t shape[1] = {2};
  double ad[2] = {0.3, 1.0};
  double bd[2] = {0.1, 1.0};
  double cd[2] = {9.0, 9.0};
  const double ed[2] = {0x1.9999999999999p-3, 0.0};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_f64(&a, &b, &c);
  check_f64("f64_example3", cd, ed, 2);
}

/* -0 - +0 = -0; +inf - +inf = NaN; exact cancellations give +0. */
static void case_f64_special(void) {
  const size_t shape[1] = {4};
  double ad[4] = {-0.0, f64_bits(0x7FF0000000000000ull), 1.5, -2.25};
  double bd[4] = {0.0, f64_bits(0x7FF0000000000000ull), 1.5, -1.0};
  double cd[4] = {9.0, 9.0, 9.0, 9.0};
  const double ed[4] = {-0.0, NAN, 0.0, -1.25};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_f64(&a, &b, &c);
  check_f64("f64_special", cd, ed, 4);
}

/* Overflow to infinity at the double boundary. */
static void case_f64_overflow(void) {
  const size_t shape[1] = {2};
  const double dmax = f64_bits(0x7FEFFFFFFFFFFFFFull);
  double ad[2] = {dmax, -dmax};
  double bd[2] = {-dmax, dmax};
  double cd[2] = {9.0, 9.0};
  const double ed[2] = {f64_bits(0x7FF0000000000000ull), f64_bits(0xFFF0000000000000ull)};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_f64(&a, &b, &c);
  check_f64("f64_overflow", cd, ed, 2);
}

/* ---- signed integer cases ------------------------------------- */

/* Spec, int example 1. */
static void case_i8_example1(void) {
  const size_t shape[1] = {3};
  int8_t ad[3] = {6, 5, -35};
  int8_t bd[3] = {3, 3, 3};
  int8_t cd[3] = {0, 0, 0};
  const int8_t ed[3] = {3, 2, -38};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_i8(&a, &b, &c);
  check_i8("i8_example1", cd, ed, 3);
}

/* Spec, int example 2: 127 - (-1) = -128 and -128 - 1 = 127. */
static void case_i8_example2(void) {
  const size_t shape[1] = {3};
  int8_t ad[3] = {127, -128, 0};
  int8_t bd[3] = {-1, 1, 1};
  int8_t cd[3] = {0, 0, 0};
  const int8_t ed[3] = {-128, 127, -1};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_i8(&a, &b, &c);
  check_i8("i8_example2", cd, ed, 3);
}

/* Modular wrap at both ends of int8, including a sum of 400 at i8. */
static void case_i8_wrap(void) {
  const size_t shape[1] = {5};
  int8_t ad[5] = {-128, 127, 0, -1, 100};
  int8_t bd[5] = {1, -1, 1, -128, -44};
  int8_t cd[5] = {0, 0, 0, 0, 0};
  const int8_t ed[5] = {127, -128, -1, 127, -112};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_i8(&a, &b, &c);
  check_i8("i8_wrap", cd, ed, 5);
}

/* int16 bounds: INT16_MAX - (-1) wraps to INT16_MIN, INT16_MIN - 1 to INT16_MAX. */
static void case_i16_bounds(void) {
  const size_t shape[1] = {4};
  int16_t ad[4] = {INT16_MAX, INT16_MIN, 0, -1};
  int16_t bd[4] = {-1, 1, 0, INT16_MIN};
  int16_t cd[4] = {0, 0, 0, 0};
  const int16_t ed[4] = {INT16_MIN, INT16_MAX, 0, INT16_MAX};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_i16(&a, &b, &c);
  check_i16("i16_bounds", cd, ed, 4);
}

/* int32 wrap: INT32_MAX - INT32_MIN = 4294967295 mod 2^32 = -1. */
static void case_i32_wrap(void) {
  const size_t shape[1] = {4};
  int32_t ad[4] = {INT32_MIN, INT32_MAX, INT32_MIN, 7};
  int32_t bd[4] = {INT32_MIN, INT32_MIN, 1, 7};
  int32_t cd[4] = {0, 0, 0, 0};
  const int32_t ed[4] = {0, -1, INT32_MAX, 0};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_i32(&a, &b, &c);
  check_i32("i32_wrap", cd, ed, 4);
}

/* A two-dimensional shape: the element count is the product of the dims. */
static void case_i32_2d(void) {
  const size_t shape[2] = {2, 3};
  int32_t ad[6] = {1, 2, 3, 4, 5, 6};
  int32_t bd[6] = {6, 5, 4, 3, 2, 1};
  int32_t cd[6] = {0, 0, 0, 0, 0, 0};
  const int32_t ed[6] = {-5, -3, -1, 1, 3, 5};
  sonnx_tensor a = { .data = ad, .ndim = 2, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 2, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 2, .shape = shape };
  sonnx_sub_i32(&a, &b, &c);
  check_i32("i32_2d", cd, ed, 6);
}

/* int64 wrap at the extremes. */
static void case_i64_wrap(void) {
  const size_t shape[1] = {3};
  int64_t ad[3] = {INT64_MAX, INT64_MIN, INT64_MIN};
  int64_t bd[3] = {INT64_MIN, 1, INT64_MIN};
  int64_t cd[3] = {0, 0, 0};
  const int64_t ed[3] = {-1, INT64_MAX, 0};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_i64(&a, &b, &c);
  check_i64("i64_wrap", cd, ed, 3);
}

/* ---- unsigned integer cases ----------------------------------- */

/* Spec, uint example 1. */
static void case_u8_example1(void) {
  const size_t shape[1] = {3};
  uint8_t ad[3] = {0, 255, 4};
  uint8_t bd[3] = {1, 1, 7};
  uint8_t cd[3] = {9, 9, 9};
  const uint8_t ed[3] = {255, 254, 253};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_u8(&a, &b, &c);
  check_u8("u8_example1", cd, ed, 3);
}

/* Underflow wraps modulo 2^8: 0 - 1 = 255. */
static void case_u8_wrap(void) {
  const size_t shape[1] = {4};
  uint8_t ad[4] = {0, 1, 255, 200};
  uint8_t bd[4] = {1, 0, 255, 100};
  uint8_t cd[4] = {9, 9, 9, 9};
  const uint8_t ed[4] = {255, 1, 0, 100};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_u8(&a, &b, &c);
  check_u8("u8_wrap", cd, ed, 4);
}

/* uint16 underflow: 4 - 7 = -3 + 2^16 = 65533. */
static void case_u16_wrap(void) {
  const size_t shape[1] = {3};
  uint16_t ad[3] = {0, UINT16_MAX, 4};
  uint16_t bd[3] = {1, 1, 7};
  uint16_t cd[3] = {9, 9, 9};
  const uint16_t ed[3] = {UINT16_MAX, UINT16_MAX - 1, 65533u};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_u16(&a, &b, &c);
  check_u16("u16_wrap", cd, ed, 3);
}

/* uint32: 0u - 1u = UINT32_MAX. */
static void case_u32_wrap(void) {
  const size_t shape[1] = {4};
  uint32_t ad[4] = {0u, 1u, UINT32_MAX, UINT32_MAX};
  uint32_t bd[4] = {1u, 0u, UINT32_MAX, 0u};
  uint32_t cd[4] = {9u, 9u, 9u, 9u};
  const uint32_t ed[4] = {UINT32_MAX, 1u, 0u, UINT32_MAX};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_u32(&a, &b, &c);
  check_u32("u32_wrap", cd, ed, 4);
}

/* A two-dimensional unsigned shape. */
static void case_u32_2d(void) {
  const size_t shape[2] = {2, 2};
  uint32_t ad[4] = {0u, 1u, 2u, 3u};
  uint32_t bd[4] = {3u, 2u, 1u, 0u};
  uint32_t cd[4] = {9u, 9u, 9u, 9u};
  const uint32_t ed[4] = {UINT32_MAX, UINT32_MAX, 1u, 3u};
  sonnx_tensor a = { .data = ad, .ndim = 2, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 2, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 2, .shape = shape };
  sonnx_sub_u32(&a, &b, &c);
  check_u32("u32_2d", cd, ed, 4);
}

/* uint64: 0 - 1 = UINT64_MAX, 5 - 7 = UINT64_MAX - 1. */
static void case_u64_wrap(void) {
  const size_t shape[1] = {3};
  uint64_t ad[3] = {0u, UINT64_MAX, 5u};
  uint64_t bd[3] = {1u, UINT64_MAX, 7u};
  uint64_t cd[3] = {9u, 9u, 9u};
  const uint64_t ed[3] = {UINT64_MAX, 0u, UINT64_MAX - 1u};
  sonnx_tensor a = { .data = ad, .ndim = 1, .shape = shape };
  sonnx_tensor b = { .data = bd, .ndim = 1, .shape = shape };
  sonnx_tensor c = { .data = cd, .ndim = 1, .shape = shape };
  sonnx_sub_u64(&a, &b, &c);
  check_u64("u64_wrap", cd, ed, 3);
}

int main(void) {
  case_f32_example1();
  case_f32_example2();
  case_f32_infinities();
  case_f32_nan();
  case_f32_signed_zero();
  case_f32_overflow();
  case_f32_subnormal();
  case_f32_rounding();
  case_f32_rank0();
  case_f32_zero_size();

  case_f64_example3();
  case_f64_special();
  case_f64_overflow();

  case_i8_example1();
  case_i8_example2();
  case_i8_wrap();
  case_i16_bounds();
  case_i32_wrap();
  case_i32_2d();
  case_i64_wrap();

  case_u8_example1();
  case_u8_wrap();
  case_u16_wrap();
  case_u32_wrap();
  case_u32_2d();
  case_u64_wrap();

  if (failures == 0) {
    return 0;
  }
  fprintf(stderr, "%d failure(s)\n", failures);
  return 1;
}
