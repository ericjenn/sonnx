#include "sonnx_kernels.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

/* Exact bit-level views, used so that signed zeros are distinguished and
   NaN payloads are not compared. */
static uint32_t f32_bits(float x) {
  uint32_t u;
  memcpy(&u, &x, sizeof u);
  return u;
}

static uint64_t f64_bits(double x) {
  uint64_t u;
  memcpy(&u, &x, sizeof u);
  return u;
}

static void check_f32(const char *name, const float *got, const float *exp,
                      size_t n) {
  size_t i;
  int bad = 0;
  for (i = 0; i < n; i++) {
    if (isnan(exp[i])) {
      if (!isnan(got[i])) {
        bad = 1;
      }
    } else if (f32_bits(got[i]) != f32_bits(exp[i])) {
      bad = 1;
    }
  }
  if (!bad) {
    printf("ok %s\n", name);
    return;
  }
  {
    char eb[2048];
    char gb[2048];
    size_t eo = 0;
    size_t go = 0;
    eb[0] = '\0';
    gb[0] = '\0';
    for (i = 0; i < n; i++) {
      int w;
      if (eo + 64 < sizeof eb) {
        if (i == 0) {
          w = snprintf(eb + eo, sizeof eb - eo, "%.9g", (double)exp[i]);
        } else {
          w = snprintf(eb + eo, sizeof eb - eo, " %.9g", (double)exp[i]);
        }
        if (w > 0) {
          eo += (size_t)w;
        }
      }
      if (go + 64 < sizeof gb) {
        if (i == 0) {
          w = snprintf(gb + go, sizeof gb - go, "%.9g", (double)got[i]);
        } else {
          w = snprintf(gb + go, sizeof gb - go, " %.9g", (double)got[i]);
        }
        if (w > 0) {
          go += (size_t)w;
        }
      }
    }
    printf("FAIL %s: expected [%s], got [%s]\n", name, eb, gb);
    failures += 1;
  }
}

static void check_f64(const char *name, const double *got, const double *exp,
                      size_t n) {
  size_t i;
  int bad = 0;
  for (i = 0; i < n; i++) {
    if (isnan(exp[i])) {
      if (!isnan(got[i])) {
        bad = 1;
      }
    } else if (f64_bits(got[i]) != f64_bits(exp[i])) {
      bad = 1;
    }
  }
  if (!bad) {
    printf("ok %s\n", name);
    return;
  }
  {
    char eb[2048];
    char gb[2048];
    size_t eo = 0;
    size_t go = 0;
    eb[0] = '\0';
    gb[0] = '\0';
    for (i = 0; i < n; i++) {
      int w;
      if (eo + 64 < sizeof eb) {
        if (i == 0) {
          w = snprintf(eb + eo, sizeof eb - eo, "%.17g", exp[i]);
        } else {
          w = snprintf(eb + eo, sizeof eb - eo, " %.17g", exp[i]);
        }
        if (w > 0) {
          eo += (size_t)w;
        }
      }
      if (go + 64 < sizeof gb) {
        if (i == 0) {
          w = snprintf(gb + go, sizeof gb - go, "%.17g", got[i]);
        } else {
          w = snprintf(gb + go, sizeof gb - go, " %.17g", got[i]);
        }
        if (w > 0) {
          go += (size_t)w;
        }
      }
    }
    printf("FAIL %s: expected [%s], got [%s]\n", name, eb, gb);
    failures += 1;
  }
}

#define DEF_INT_CHECK(TAG, T, FMT, CASTT)                                      \
  static void check_##TAG(const char *name, const T *got, const T *exp,        \
                          size_t n) {                                          \
    size_t i;                                                                  \
    int bad = 0;                                                               \
    for (i = 0; i < n; i++) {                                                  \
      if (got[i] != exp[i]) {                                                  \
        bad = 1;                                                               \
      }                                                                        \
    }                                                                          \
    if (!bad) {                                                                \
      printf("ok %s\n", name);                                                 \
      return;                                                                  \
    }                                                                          \
    {                                                                          \
      char eb[2048];                                                           \
      char gb[2048];                                                           \
      size_t eo = 0;                                                           \
      size_t go = 0;                                                           \
      eb[0] = '\0';                                                            \
      gb[0] = '\0';                                                            \
      for (i = 0; i < n; i++) {                                                \
        int w;                                                                 \
        if (eo + 64 < sizeof eb) {                                             \
          if (i == 0) {                                                        \
            w = snprintf(eb + eo, sizeof eb - eo, FMT, (CASTT)exp[i]);         \
          } else {                                                             \
            w = snprintf(eb + eo, sizeof eb - eo, " " FMT, (CASTT)exp[i]);     \
          }                                                                    \
          if (w > 0) {                                                         \
            eo += (size_t)w;                                                   \
          }                                                                    \
        }                                                                      \
        if (go + 64 < sizeof gb) {                                             \
          if (i == 0) {                                                        \
            w = snprintf(gb + go, sizeof gb - go, FMT, (CASTT)got[i]);         \
          } else {                                                             \
            w = snprintf(gb + go, sizeof gb - go, " " FMT, (CASTT)got[i]);     \
          }                                                                    \
          if (w > 0) {                                                         \
            go += (size_t)w;                                                   \
          }                                                                    \
        }                                                                      \
      }                                                                        \
      printf("FAIL %s: expected [%s], got [%s]\n", name, eb, gb);              \
      failures += 1;                                                           \
    }                                                                          \
  }

DEF_INT_CHECK(i8, int8_t, "%d", int)
DEF_INT_CHECK(i16, int16_t, "%d", int)
DEF_INT_CHECK(i32, int32_t, "%ld", long)
DEF_INT_CHECK(i64, int64_t, "%lld", long long)
DEF_INT_CHECK(u8, uint8_t, "%u", unsigned)
DEF_INT_CHECK(u16, uint16_t, "%u", unsigned)
DEF_INT_CHECK(u32, uint32_t, "%lu", unsigned long)
DEF_INT_CHECK(u64, uint64_t, "%llu", unsigned long long)

int main(void) {
  /* ------------------------------------------------------------------ f32 */

  /* Float Example 1: every sum is exactly representable. */
  {
    const float a[6] = {3.0f, 4.5f, 16.0f, 1.0f, 25.5f, 24.25f};
    const float b[6] = {3.0f, 2.0f, 4.0f, 0.0f, 5.0f, 4.0f};
    const float e[6] = {6.0f, 6.5f, 20.0f, 1.0f, 30.5f, 28.25f};
    float c[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    sonnx_add_f32(a, b, c, 6);
    check_f32("f32_example1", c, e, 6);
  }

  /* Float Example 2: infinities, NaN from (+inf) + (-inf), signed zeros. */
  {
    const float a[4] = {1.0f, INFINITY, 0.0f, -INFINITY};
    const float b[4] = {1.0f, -INFINITY, -0.0f, -INFINITY};
    const float e[4] = {2.0f, NAN, 0.0f, -INFINITY};
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    sonnx_add_f32(a, b, c, 4);
    check_f32("f32_example2_special", c, e, 4);
  }

  /* Null sums and their signs, checked through signbit. */
  {
    const float a[4] = {-0.0f, -0.0f, 0.0f, 0.0f};
    const float b[4] = {-0.0f, 0.0f, -0.0f, 0.0f};
    const int eneg[4] = {1, 0, 0, 0};
    float c[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    int got_neg[4];
    size_t i;
    int ok = 1;
    sonnx_add_f32(a, b, c, 4);
    for (i = 0; i < 4; i++) {
      got_neg[i] = signbit(c[i]) ? 1 : 0;
      if (c[i] != 0.0f) {
        ok = 0;
      }
      if (got_neg[i] != eneg[i]) {
        ok = 0;
      }
    }
    if (ok) {
      printf("ok f32_zero_signs\n");
    } else {
      char eb[64];
      char gb[64];
      snprintf(eb, sizeof eb, "%s %s %s %s", eneg[0] ? "-0" : "+0",
               eneg[1] ? "-0" : "+0", eneg[2] ? "-0" : "+0",
               eneg[3] ? "-0" : "+0");
      snprintf(gb, sizeof gb, "%s %s %s %s", got_neg[0] ? "-0" : "+0",
               got_neg[1] ? "-0" : "+0", got_neg[2] ? "-0" : "+0",
               got_neg[3] ? "-0" : "+0");
      printf("FAIL f32_zero_signs: expected [%s], got [%s]\n", eb, gb);
      failures += 1;
    }
  }

  /* Exact binary fractions, negative operands, cancelling operands. */
  {
    const float a[4] = {1.5f, -1.5f, 0.25f, -0.25f};
    const float b[4] = {0.25f, 0.25f, -0.25f, -0.25f};
    const float e[4] = {1.75f, -1.25f, 0.0f, -0.5f};
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    sonnx_add_f32(a, b, c, 4);
    check_f32("f32_binary_fractions", c, e, 4);
  }

  /* Infinities of equal sign, and infinity plus a finite value. */
  {
    const float a[3] = {INFINITY, -INFINITY, INFINITY};
    const float b[3] = {INFINITY, -INFINITY, 1.5f};
    const float e[3] = {INFINITY, -INFINITY, INFINITY};
    float c[3] = {0.0f, 0.0f, 0.0f};
    sonnx_add_f32(a, b, c, 3);
    check_f32("f32_infinity_same_sign", c, e, 3);
  }

  /* Overflow: the exact sum rounds beyond the largest finite f32. */
  {
    const float a[2] = {0x1.fffffep127f, -0x1.fffffep127f};
    const float b[2] = {0x1.fffffep127f, -0x1.fffffep127f};
    const float e[2] = {INFINITY, -INFINITY};
    float c[2] = {0.0f, 0.0f};
    sonnx_add_f32(a, b, c, 2);
    check_f32("f32_overflow_to_infinity", c, e, 2);
  }

  /* A result of length zero must leave the output untouched. */
  {
    float a = 1.0f;
    float b = 2.0f;
    float c = -7.0f;
    sonnx_add_f32(&a, &b, &c, 0);
    if (c == -7.0f) {
      printf("ok f32_zero_length\n");
    } else {
      printf("FAIL f32_zero_length: expected [-7], got [%.9g]\n", (double)c);
      failures += 1;
    }
  }

  /* ------------------------------------------------------------------ f64 */

  /* Float Example 3: 0.1 + 0.2 rounds to 0x1.3333333333334p-2; 1.0 + -1.0
     is the null sum of opposite signs, i.e. +0. */
  {
    const double a[3] = {0.1, 1.0, 1.0};
    const double b[3] = {0.2, 2.0, -1.0};
    const double e[3] = {0x1.3333333333334p-2, 3.0, 0.0};
    double c[3] = {0.0, 0.0, 0.0};
    sonnx_add_f64(a, b, c, 3);
    check_f64("f64_example3", c, e, 3);
  }

  /* Zeros, infinities and the invalid operation in f64. */
  {
    const double a[4] = {-0.0, -0.0, 0.0, INFINITY};
    const double b[4] = {-0.0, 0.0, -0.0, -INFINITY};
    const double e[4] = {-0.0, 0.0, 0.0, NAN};
    double c[4] = {0.0, 0.0, 0.0, 0.0};
    sonnx_add_f64(a, b, c, 4);
    check_f64("f64_zero_infinity_nan", c, e, 4);
  }

  /* Overflow beyond the largest f64, and a null sum of opposite values. */
  {
    const double a[3] = {0x1.fffffffffffffp1023, -0x1.fffffffffffffp1023,
                         0x1p-1022};
    const double b[3] = {0x1.fffffffffffffp1023, -0x1.fffffffffffffp1023,
                         -0x1p-1022};
    const double e[3] = {INFINITY, -INFINITY, 0.0};
    double c[3] = {0.0, 0.0, 0.0};
    sonnx_add_f64(a, b, c, 3);
    check_f64("f64_overflow_and_cancel", c, e, 3);
  }

  {
    double a = 0.5;
    double b = 0.25;
    double c = -7.0;
    sonnx_add_f64(&a, &b, &c, 0);
    if (c == -7.0) {
      printf("ok f64_zero_length\n");
    } else {
      printf("FAIL f64_zero_length: expected [-7], got [%.17g]\n", c);
      failures += 1;
    }
  }

  /* --------------------------------------------------------- signed ints */

  /* Signed Example 1, replayed exactly. */
  {
    const int8_t a[3] = {-6, 100, -100};
    const int8_t b[3] = {-3, 100, -100};
    const int8_t e[3] = {-9, -56, 56};
    int8_t c[3] = {0, 0, 0};
    sonnx_add_i8(a, b, c, 3);
    check_i8("i8_example1", c, e, 3);
  }

  /* i8 boundaries: wrap modulo 2^8, one step outside the range. */
  {
    const int8_t a[7] = {INT8_MIN, INT8_MAX, INT8_MIN, INT8_MAX, 0, -1, 1};
    const int8_t b[7] = {-1, 1, INT8_MIN, INT8_MAX, 0, 1, -1};
    const int8_t e[7] = {127, -128, 0, -2, 0, 0, 0};
    int8_t c[7] = {0, 0, 0, 0, 0, 0, 0};
    sonnx_add_i8(a, b, c, 7);
    check_i8("i8_wrap_boundaries", c, e, 7);
  }

  {
    const int16_t a[5] = {INT16_MIN, INT16_MAX, INT16_MAX, -1, 12345};
    const int16_t b[5] = {-1, 1, 1, INT16_MIN, -12345};
    const int16_t e[5] = {32767, -32768, -32768, 32767, 0};
    int16_t c[5] = {0, 0, 0, 0, 0};
    sonnx_add_i16(a, b, c, 5);
    check_i16("i16_wrap_boundaries", c, e, 5);
  }

  {
    const int32_t a[5] = {INT32_MIN, INT32_MAX, INT32_MIN, INT32_MAX, 0};
    const int32_t b[5] = {-1, 1, INT32_MIN, INT32_MIN, 0};
    const int32_t e[5] = {INT32_MAX, INT32_MIN, 0, -1, 0};
    int32_t c[5] = {0, 0, 0, 0, 0};
    sonnx_add_i32(a, b, c, 5);
    check_i32("i32_wrap_boundaries", c, e, 5);
  }

  {
    const int64_t a[5] = {INT64_MIN, INT64_MAX, INT64_MIN, INT64_MAX, 0};
    const int64_t b[5] = {-1, 1, INT64_MIN, INT64_MAX, 0};
    const int64_t e[5] = {INT64_MAX, INT64_MIN, 0, -2, 0};
    int64_t c[5] = {0, 0, 0, 0, 0};
    sonnx_add_i64(a, b, c, 5);
    check_i64("i64_wrap_boundaries", c, e, 5);
  }

  {
    int8_t a = 1;
    int8_t b = 2;
    int8_t c = -7;
    sonnx_add_i8(&a, &b, &c, 0);
    if (c == -7) {
      printf("ok i8_zero_length\n");
    } else {
      printf("FAIL i8_zero_length: expected [-7], got [%d]\n", (int)c);
      failures += 1;
    }
  }

  /* ------------------------------------------------------- unsigned ints */

  /* Unsigned Example 1, replayed exactly. */
  {
    const uint8_t a[3] = {6, 200, 35};
    const uint8_t b[3] = {3, 100, 5};
    const uint8_t e[3] = {9, 44, 40};
    uint8_t c[3] = {0, 0, 0};
    sonnx_add_u8(a, b, c, 3);
    check_u8("u8_example1", c, e, 3);
  }

  /* u8 boundaries: wrap modulo 2^8, the result may be smaller than both. */
  {
    const uint8_t a[6] = {0, 255, 255, 200, 1, 128};
    const uint8_t b[6] = {0, 1, 255, 100, 254, 128};
    const uint8_t e[6] = {0, 0, 254, 44, 255, 0};
    uint8_t c[6] = {0, 0, 0, 0, 0, 0};
    sonnx_add_u8(a, b, c, 6);
    check_u8("u8_wrap_boundaries", c, e, 6);
  }

  {
    const uint16_t a[4] = {0, 65535, 65535, 32768};
    const uint16_t b[4] = {0, 1, 65535, 32768};
    const uint16_t e[4] = {0, 0, 65534, 0};
    uint16_t c[4] = {0, 0, 0, 0};
    sonnx_add_u16(a, b, c, 4);
    check_u16("u16_wrap_boundaries", c, e, 4);
  }

  {
    const uint32_t a[4] = {0u, UINT32_MAX, UINT32_MAX, 2147483648u};
    const uint32_t b[4] = {0u, 1u, UINT32_MAX, 2147483648u};
    const uint32_t e[4] = {0u, 0u, 4294967294u, 0u};
    uint32_t c[4] = {0u, 0u, 0u, 0u};
    sonnx_add_u32(a, b, c, 4);
    check_u32("u32_wrap_boundaries", c, e, 4);
  }

  {
    const uint64_t a[4] = {UINT64_C(0), UINT64_MAX, UINT64_MAX, UINT64_C(0)};
    const uint64_t b[4] = {UINT64_C(0), UINT64_C(1), UINT64_MAX, UINT64_MAX};
    const uint64_t e[4] = {UINT64_C(0), UINT64_C(0),
                           UINT64_C(18446744073709551614), UINT64_MAX};
    uint64_t c[4] = {UINT64_C(0), UINT64_C(0), UINT64_C(0), UINT64_C(0)};
    sonnx_add_u64(a, b, c, 4);
    check_u64("u64_wrap_boundaries", c, e, 4);
  }

  {
    uint8_t a = 1;
    uint8_t b = 2;
    uint8_t c = 255;
    sonnx_add_u8(&a, &b, &c, 0);
    if (c == 255) {
      printf("ok u8_zero_length\n");
    } else {
      printf("FAIL u8_zero_length: expected [255], got [%u]\n", (unsigned)c);
      failures += 1;
    }
  }

  return failures == 0 ? 0 : 1;
}
