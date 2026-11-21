#include "sonnx_kernels.h"

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <limits.h>
#include <string.h>

static int fails = 0;

/* Exact comparison for floats: bit equality, except that an expected NaN
 * accepts any NaN (the specification leaves the NaN sign and payload open). */
static int f32_match(float got, float want) {
  uint32_t bg = 0;
  uint32_t bw = 0;
  if (isnan(want)) {
    return isnan(got) != 0;
  }
  memcpy(&bg, &got, sizeof bg);
  memcpy(&bw, &want, sizeof bw);
  return bg == bw;
}

static int f64_match(double got, double want) {
  uint64_t bg = 0;
  uint64_t bw = 0;
  if (isnan(want)) {
    return isnan(got) != 0;
  }
  memcpy(&bg, &got, sizeof bg);
  memcpy(&bw, &want, sizeof bw);
  return bg == bw;
}

#define DEF_SIGNED_CHECK(NAME, T, FN)                                         \
  static void NAME(const char *name, const T *a, const T *b, const T *e,      \
                   size_t n) {                                                \
    T g[16];                                                                  \
    FN(a, b, g, n);                                                           \
    int pass = 1;                                                             \
    for (size_t i = 0; i < n; i++) {                                          \
      if (g[i] != e[i]) {                                                     \
        pass = 0;                                                             \
      }                                                                       \
    }                                                                         \
    if (pass) {                                                               \
      printf("ok %s\n", name);                                                \
      return;                                                                 \
    }                                                                         \
    printf("FAIL %s: expected [", name);                                      \
    for (size_t i = 0; i < n; i++) {                                          \
      printf("%lld%s", (long long)e[i], (i + 1 < n) ? ", " : "");             \
    }                                                                         \
    printf("], got [");                                                       \
    for (size_t i = 0; i < n; i++) {                                          \
      printf("%lld%s", (long long)g[i], (i + 1 < n) ? ", " : "");             \
    }                                                                         \
    printf("]\n");                                                            \
    fails++;                                                                  \
  }

#define DEF_UNSIGNED_CHECK(NAME, T, FN)                                       \
  static void NAME(const char *name, const T *a, const T *b, const T *e,      \
                   size_t n) {                                                \
    T g[16];                                                                  \
    FN(a, b, g, n);                                                           \
    int pass = 1;                                                             \
    for (size_t i = 0; i < n; i++) {                                          \
      if (g[i] != e[i]) {                                                     \
        pass = 0;                                                             \
      }                                                                       \
    }                                                                         \
    if (pass) {                                                               \
      printf("ok %s\n", name);                                                \
      return;                                                                 \
    }                                                                         \
    printf("FAIL %s: expected [", name);                                      \
    for (size_t i = 0; i < n; i++) {                                          \
      printf("%llu%s", (unsigned long long)e[i], (i + 1 < n) ? ", " : "");    \
    }                                                                         \
    printf("], got [");                                                       \
    for (size_t i = 0; i < n; i++) {                                          \
      printf("%llu%s", (unsigned long long)g[i], (i + 1 < n) ? ", " : "");    \
    }                                                                         \
    printf("]\n");                                                            \
    fails++;                                                                  \
  }

#define DEF_FLOAT_CHECK(NAME, T, FN, MATCH)                                   \
  static void NAME(const char *name, const T *a, const T *b, const T *e,      \
                   size_t n) {                                                \
    T g[16];                                                                  \
    FN(a, b, g, n);                                                           \
    int pass = 1;                                                             \
    for (size_t i = 0; i < n; i++) {                                          \
      if (!MATCH(g[i], e[i])) {                                               \
        pass = 0;                                                             \
      }                                                                       \
    }                                                                         \
    if (pass) {                                                               \
      printf("ok %s\n", name);                                                \
      return;                                                                 \
    }                                                                         \
    printf("FAIL %s: expected [", name);                                      \
    for (size_t i = 0; i < n; i++) {                                          \
      printf("%a%s", (double)e[i], (i + 1 < n) ? ", " : "");                  \
    }                                                                         \
    printf("], got [");                                                       \
    for (size_t i = 0; i < n; i++) {                                          \
      printf("%a%s", (double)g[i], (i + 1 < n) ? ", " : "");                  \
    }                                                                         \
    printf("]\n");                                                            \
    fails++;                                                                  \
  }

DEF_SIGNED_CHECK(check_i8, int8_t, sonnx_sub_i8)
DEF_SIGNED_CHECK(check_i16, int16_t, sonnx_sub_i16)
DEF_SIGNED_CHECK(check_i32, int32_t, sonnx_sub_i32)
DEF_SIGNED_CHECK(check_i64, int64_t, sonnx_sub_i64)
DEF_UNSIGNED_CHECK(check_u8, uint8_t, sonnx_sub_u8)
DEF_UNSIGNED_CHECK(check_u16, uint16_t, sonnx_sub_u16)
DEF_UNSIGNED_CHECK(check_u32, uint32_t, sonnx_sub_u32)
DEF_UNSIGNED_CHECK(check_u64, uint64_t, sonnx_sub_u64)
DEF_FLOAT_CHECK(check_f32, float, sonnx_sub_f32, f32_match)
DEF_FLOAT_CHECK(check_f64, double, sonnx_sub_f64, f64_match)

int main(void) {
  /* ---------------------------------------------------------------- f32 */
  {
    const float a[] = {3.0f, 4.5f, 16.0f, 1.0f, 25.5f, 24.25f};
    const float b[] = {3.0f, 2.0f, 4.0f, 0.0f, 5.0f, 4.0f};
    const float e[] = {0.0f, 2.5f, 12.0f, 1.0f, 20.5f, 20.25f};
    check_f32("f32 float example 1", a, b, e, 6);
  }
  {
    const float a[] = {1.0f, INFINITY, -0.0f, 0.0f};
    const float b[] = {1.0f, INFINITY, 0.0f, -0.0f};
    const float e[] = {0.0f, NAN, -0.0f, 0.0f};
    check_f32("f32 float example 2 (inf and signed zeros)", a, b, e, 4);
  }
  {
    const float a[] = {INFINITY, -INFINITY, INFINITY, -INFINITY, INFINITY,
                       -INFINITY};
    const float b[] = {1.0f, 1.0f, -INFINITY, INFINITY, INFINITY, -INFINITY};
    const float e[] = {INFINITY, -INFINITY, INFINITY, -INFINITY, NAN, NAN};
    check_f32("f32 infinite operands", a, b, e, 6);
  }
  {
    const float a[] = {NAN, 1.0f, -INFINITY};
    const float b[] = {1.0f, NAN, NAN};
    const float e[] = {NAN, NAN, NAN};
    check_f32("f32 NaN propagation", a, b, e, 3);
  }
  {
    const float a[] = {1.5f, 0.25f, -0.0f, -0.0f, 7.5f, -2.5f, 0.5f};
    const float b[] = {0.25f, 0.5f, 0.0f, -0.0f, 2.5f, -2.5f, 0.5f};
    const float e[] = {1.25f, -0.25f, -0.0f, 0.0f, 5.0f, 0.0f, 0.0f};
    check_f32("f32 exact finite differences and signed zeros", a, b, e, 7);
  }
  {
    const float a[] = {0x1.fffffep+127f, -0x1.fffffep+127f};
    const float b[] = {-0x1.fffffep+127f, 0x1.fffffep+127f};
    const float e[] = {INFINITY, -INFINITY};
    check_f32("f32 overflow to infinity", a, b, e, 2);
  }
  {
    const float a[] = {6.5f};
    const float b[] = {2.25f};
    const float e[] = {4.25f};
    check_f32("f32 single element", a, b, e, 1);
  }

  /* ---------------------------------------------------------------- f64 */
  {
    const double a[] = {0.3, 1.0};
    const double b[] = {0.1, 1.0};
    const double e[] = {0.19999999999999998, 0.0};
    check_f64("f64 float example 3 (rounding and exact cancellation)", a, b, e,
              2);
  }
  {
    const double a[] = {INFINITY, -INFINITY, INFINITY, -INFINITY, NAN};
    const double b[] = {INFINITY, -INFINITY, -INFINITY, 1.0, 1.0};
    const double e[] = {NAN, NAN, INFINITY, -INFINITY, NAN};
    check_f64("f64 infinite and NaN operands", a, b, e, 5);
  }
  {
    const double a[] = {0x1.fffffffffffffp+1023, -0x1.fffffffffffffp+1023,
                        0x1.fffffffffffffp+1023};
    const double b[] = {-0x1.fffffffffffffp+1023, 0x1.fffffffffffffp+1023,
                        0x1.fffffffffffffp+1023};
    const double e[] = {INFINITY, -INFINITY, 0.0};
    check_f64("f64 overflow and max minus max", a, b, e, 3);
  }
  {
    const double a[] = {-0.0, 0.0, -0.0, 0.0};
    const double b[] = {0.0, -0.0, -0.0, 0.0};
    const double e[] = {-0.0, 0.0, 0.0, 0.0};
    check_f64("f64 signed zero differences", a, b, e, 4);
  }
  {
    /* Subnormal operands: 3*2^-1074 - 1*2^-1074 = 2*2^-1074, exactly. */
    uint64_t ua = 3;
    uint64_t ub = 1;
    uint64_t ue = 2;
    double a = 0.0;
    double b = 0.0;
    double e = 0.0;
    memcpy(&a, &ua, sizeof a);
    memcpy(&b, &ub, sizeof b);
    memcpy(&e, &ue, sizeof e);
    check_f64("f64 subnormal difference is exact", &a, &b, &e, 1);
  }
  {
    const double a[] = {1.25};
    const double b[] = {0.5};
    const double e[] = {0.75};
    check_f64("f64 single element", a, b, e, 1);
  }

  /* ---------------------------------------------------------------- i8 */
  {
    const int8_t a[] = {6, 5, -35};
    const int8_t b[] = {3, 3, 3};
    const int8_t e[] = {3, 2, -38};
    check_i8("i8 int example 1", a, b, e, 3);
  }
  {
    const int8_t a[] = {INT8_MAX, INT8_MIN, 0};
    const int8_t b[] = {-1, 1, 1};
    const int8_t e[] = {INT8_MIN, INT8_MAX, -1};
    check_i8("i8 int example 2 (wraparound)", a, b, e, 3);
  }
  {
    const int8_t a[] = {INT8_MIN, INT8_MAX, 0, -1, 1};
    const int8_t b[] = {1, -1, 0, -1, 1};
    const int8_t e[] = {INT8_MAX, INT8_MIN, 0, 0, 0};
    check_i8("i8 boundaries and equal operands", a, b, e, 5);
  }

  /* --------------------------------------------------------------- i16 */
  {
    const int16_t a[] = {INT16_MIN, INT16_MAX, INT16_MIN, 0, 1000};
    const int16_t b[] = {1, -1, INT16_MAX, INT16_MIN, 1000};
    const int16_t e[] = {INT16_MAX, INT16_MIN, 1, INT16_MIN, 0};
    check_i16("i16 boundaries", a, b, e, 5);
  }

  /* --------------------------------------------------------------- i32 */
  {
    const int32_t a[] = {INT32_MAX, INT32_MIN, INT32_MIN, 0, -5};
    const int32_t b[] = {-1, 1, INT32_MAX, INT32_MIN, -5};
    const int32_t e[] = {INT32_MIN, INT32_MAX, 1, INT32_MIN, 0};
    check_i32("i32 boundaries", a, b, e, 5);
  }

  /* --------------------------------------------------------------- i64 */
  {
    const int64_t a[] = {INT64_MAX, INT64_MIN, INT64_MIN, 0};
    const int64_t b[] = {-1, 1, INT64_MAX, INT64_MIN};
    const int64_t e[] = {INT64_MIN, INT64_MAX, 1, INT64_MIN};
    check_i64("i64 boundaries", a, b, e, 4);
  }

  /* ---------------------------------------------------------------- u8 */
  {
    const uint8_t a[] = {0, 255, 4};
    const uint8_t b[] = {1, 1, 7};
    const uint8_t e[] = {255, 254, 253};
    check_u8("u8 uint example 1", a, b, e, 3);
  }
  {
    const uint8_t a[] = {0, 0, UINT8_MAX, 128, 7};
    const uint8_t b[] = {1, UINT8_MAX, UINT8_MAX, 129, 7};
    const uint8_t e[] = {255, 1, 0, 255, 0};
    check_u8("u8 wraparound and equal operands", a, b, e, 5);
  }

  /* --------------------------------------------------------------- u16 */
  {
    const uint16_t a[] = {0, UINT16_MAX, 100, 1000};
    const uint16_t b[] = {1, 0, 250, 1000};
    const uint16_t e[] = {65535, 65535, 65386, 0};
    check_u16("u16 boundaries", a, b, e, 4);
  }

  /* --------------------------------------------------------------- u32 */
  {
    const uint32_t a[] = {0, UINT32_MAX, 5};
    const uint32_t b[] = {1, UINT32_MAX, 7};
    const uint32_t e[] = {UINT32_MAX, 0, UINT32_MAX - 1};
    check_u32("u32 boundaries", a, b, e, 3);
  }

  /* --------------------------------------------------------------- u64 */
  {
    const uint64_t a[] = {0, UINT64_MAX, 5, 12345};
    const uint64_t b[] = {1, 1, 7, 12345};
    const uint64_t e[] = {UINT64_MAX, UINT64_MAX - 1, UINT64_MAX - 1, 0};
    check_u64("u64 boundaries", a, b, e, 4);
  }

  /* ------------------------------------------------------- empty output */
  {
    float a32[1] = {1.0f};
    float b32[1] = {2.0f};
    float c32[1] = {42.0f};
    double a64[1] = {1.0};
    double b64[1] = {2.0};
    double c64[1] = {42.0};
    int32_t ai[1] = {1};
    int32_t bi[1] = {2};
    int32_t ci[1] = {42};
    uint8_t au[1] = {1};
    uint8_t bu[1] = {2};
    uint8_t cu[1] = {42};
    sonnx_sub_f32(a32, b32, c32, 0);
    sonnx_sub_f64(a64, b64, c64, 0);
    sonnx_sub_i32(ai, bi, ci, 0);
    sonnx_sub_u8(au, bu, cu, 0);
    if (c32[0] == 42.0f && c64[0] == 42.0 && ci[0] == 42 && cu[0] == 42) {
      printf("ok n=0 empty output untouched\n");
    } else {
      printf("FAIL n=0 empty output untouched: expected [42 42 42 42], "
             "got [%a %a %d %u]\n",
             (double)c32[0], c64[0], (int)ci[0], (unsigned)cu[0]);
      fails++;
    }
  }

  return fails == 0 ? 0 : 1;
}
