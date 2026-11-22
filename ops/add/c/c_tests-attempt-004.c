/*
 * Tests for the generated SONNX Add kernels.
 *
 * Expected values are taken from the specification (examples, relations and
 * the arithmetic model: exact real addition, IEEE 754 roundTiesToEven with an
 * unbounded exponent range, signed wrap modulo 2^n, unsigned wrap modulo 2^n)
 * and computed here independently of the kernel.
 */

#include "sonnx_kernels.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_failures = 0;

static void report(const char *name, int pass, const char *detail) {
  if (pass) {
    printf("ok %s\n", name);
  } else {
    printf("FAIL %s: %s\n", name, detail);
    g_failures++;
  }
}

/* Floating-point comparison: NaN is compared with isnan, a null result with
 * signbit, everything else exactly. */
static void check_f32(const char *name, const float *got, const float *want,
                      const int *want_nan, size_t n) {
  char det[256];
  size_t i;
  for (i = 0; i < n; i++) {
    int bad;
    if (want_nan[i]) {
      bad = !isnan(got[i]);
    } else if (want[i] == 0.0f) {
      bad = !(got[i] == 0.0f &&
              ((signbit(got[i]) != 0) == (signbit(want[i]) != 0)));
    } else {
      bad = !(got[i] == want[i]);
    }
    if (bad) {
      snprintf(det, sizeof det, "element %zu expected %g, got %g", i,
               (double)want[i], (double)got[i]);
      report(name, 0, det);
      return;
    }
  }
  report(name, 1, "");
}

static void check_f64(const char *name, const double *got, const double *want,
                      const int *want_nan, size_t n) {
  char det[256];
  size_t i;
  for (i = 0; i < n; i++) {
    int bad;
    if (want_nan[i]) {
      bad = !isnan(got[i]);
    } else if (want[i] == 0.0) {
      bad = !(got[i] == 0.0 &&
              ((signbit(got[i]) != 0) == (signbit(want[i]) != 0)));
    } else {
      bad = !(got[i] == want[i]);
    }
    if (bad) {
      snprintf(det, sizeof det, "element %zu expected %.17g, got %.17g", i,
               want[i], got[i]);
      report(name, 0, det);
      return;
    }
  }
  report(name, 1, "");
}

static void check_i8(const char *name, const int8_t *got, const int8_t *want,
                     size_t n) {
  char det[256];
  size_t i;
  for (i = 0; i < n; i++) {
    if (got[i] != want[i]) {
      snprintf(det, sizeof det, "element %zu expected %d, got %d", i,
               (int)want[i], (int)got[i]);
      report(name, 0, det);
      return;
    }
  }
  report(name, 1, "");
}

static void check_i16(const char *name, const int16_t *got, const int16_t *want,
                      size_t n) {
  char det[256];
  size_t i;
  for (i = 0; i < n; i++) {
    if (got[i] != want[i]) {
      snprintf(det, sizeof det, "element %zu expected %d, got %d", i,
               (int)want[i], (int)got[i]);
      report(name, 0, det);
      return;
    }
  }
  report(name, 1, "");
}

static void check_i32(const char *name, const int32_t *got, const int32_t *want,
                      size_t n) {
  char det[256];
  size_t i;
  for (i = 0; i < n; i++) {
    if (got[i] != want[i]) {
      snprintf(det, sizeof det, "element %zu expected %ld, got %ld", i,
               (long)want[i], (long)got[i]);
      report(name, 0, det);
      return;
    }
  }
  report(name, 1, "");
}

static void check_i64(const char *name, const int64_t *got, const int64_t *want,
                      size_t n) {
  char det[256];
  size_t i;
  for (i = 0; i < n; i++) {
    if (got[i] != want[i]) {
      snprintf(det, sizeof det, "element %zu expected %lld, got %lld", i,
               (long long)want[i], (long long)got[i]);
      report(name, 0, det);
      return;
    }
  }
  report(name, 1, "");
}

static void check_u8(const char *name, const uint8_t *got, const uint8_t *want,
                     size_t n) {
  char det[256];
  size_t i;
  for (i = 0; i < n; i++) {
    if (got[i] != want[i]) {
      snprintf(det, sizeof det, "element %zu expected %u, got %u", i,
               (unsigned)want[i], (unsigned)got[i]);
      report(name, 0, det);
      return;
    }
  }
  report(name, 1, "");
}

static void check_u16(const char *name, const uint16_t *got,
                      const uint16_t *want, size_t n) {
  char det[256];
  size_t i;
  for (i = 0; i < n; i++) {
    if (got[i] != want[i]) {
      snprintf(det, sizeof det, "element %zu expected %u, got %u", i,
               (unsigned)want[i], (unsigned)got[i]);
      report(name, 0, det);
      return;
    }
  }
  report(name, 1, "");
}

static void check_u32(const char *name, const uint32_t *got,
                      const uint32_t *want, size_t n) {
  char det[256];
  size_t i;
  for (i = 0; i < n; i++) {
    if (got[i] != want[i]) {
      snprintf(det, sizeof det, "element %zu expected %lu, got %lu", i,
               (unsigned long)want[i], (unsigned long)got[i]);
      report(name, 0, det);
      return;
    }
  }
  report(name, 1, "");
}

static void check_u64(const char *name, const uint64_t *got,
                      const uint64_t *want, size_t n) {
  char det[256];
  size_t i;
  for (i = 0; i < n; i++) {
    if (got[i] != want[i]) {
      snprintf(det, sizeof det, "element %zu expected %llu, got %llu", i,
               (unsigned long long)want[i], (unsigned long long)got[i]);
      report(name, 0, det);
      return;
    }
  }
  report(name, 1, "");
}

int main(void) {
  /* 1. f32, specification float Example 1 (all sums exactly representable). */
  {
    const size_t shape[2] = {3, 2};
    float a_data[6] = {3.0f, 4.5f, 16.0f, 1.0f, 25.5f, 24.25f};
    float b_data[6] = {3.0f, 2.0f, 4.0f, 0.0f, 5.0f, 4.0f};
    float c_data[6] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
    const float want[6] = {6.0f, 6.5f, 20.0f, 1.0f, 30.5f, 28.25f};
    const int want_nan[6] = {0, 0, 0, 0, 0, 0};
    sonnx_tensor a = {.data = a_data, .ndim = 2, .shape = shape};
    sonnx_tensor b = {.data = b_data, .ndim = 2, .shape = shape};
    sonnx_tensor c = {.data = c_data, .ndim = 2, .shape = shape};
    sonnx_add_f32(&a, &b, &c);
    check_f32("f32_example1", c_data, want, want_nan, 6);
  }

  /* 2. f32, specification float Example 2: infinities, NaN, signed zeros. */
  {
    const size_t shape[1] = {4};
    float a_data[4] = {1.0f, INFINITY, 0.0f, -INFINITY};
    float b_data[4] = {1.0f, -INFINITY, -0.0f, -INFINITY};
    float c_data[4] = {0.0f, 0.0f, 1.0f, 0.0f};
    const float want[4] = {2.0f, 0.0f, 0.0f, -INFINITY};
    const int want_nan[4] = {0, 1, 0, 0};
    sonnx_tensor a = {.data = a_data, .ndim = 1, .shape = shape};
    sonnx_tensor b = {.data = b_data, .ndim = 1, .shape = shape};
    sonnx_tensor c = {.data = c_data, .ndim = 1, .shape = shape};
    sonnx_add_f32(&a, &b, &c);
    check_f32("f32_example2_specials", c_data, want, want_nan, 4);
  }

  /* 3. f32, sign of a null sum: -0 + -0 = -0, +0 + -0 = +0. */
  {
    const size_t shape[1] = {2};
    float a_data[2] = {-0.0f, 0.0f};
    float b_data[2] = {-0.0f, -0.0f};
    float c_data[2] = {1.0f, 1.0f};
    const float want[2] = {-0.0f, 0.0f};
    const int want_nan[2] = {0, 0};
    sonnx_tensor a = {.data = a_data, .ndim = 1, .shape = shape};
    sonnx_tensor b = {.data = b_data, .ndim = 1, .shape = shape};
    sonnx_tensor c = {.data = c_data, .ndim = 1, .shape = shape};
    sonnx_add_f32(&a, &b, &c);
    check_f32("f32_signed_zero_sum", c_data, want, want_nan, 2);
  }

  /* 4. f32, overflow: an exact sum beyond the largest finite value gives an
   *    infinity of the sign of the exact sum. */
  {
    const size_t shape[1] = {2};
    float a_data[2] = {FLT_MAX, -FLT_MAX};
    float b_data[2] = {FLT_MAX, -FLT_MAX};
    float c_data[2] = {0.0f, 0.0f};
    const float want[2] = {INFINITY, -INFINITY};
    const int want_nan[2] = {0, 0};
    sonnx_tensor a = {.data = a_data, .ndim = 1, .shape = shape};
    sonnx_tensor b = {.data = b_data, .ndim = 1, .shape = shape};
    sonnx_tensor c = {.data = c_data, .ndim = 1, .shape = shape};
    sonnx_add_f32(&a, &b, &c);
    check_f32("f32_overflow_to_infinity", c_data, want, want_nan, 2);
  }

  /* 5. f32, subnormals, compared on the exact bit pattern:
   *    2^-149 + 2^-149 = 2^-148           (0x00000001 + 0x00000001)
   *    (2^23-1)*2^-149 + 2^-149 = 2^-126  (0x007FFFFF + 0x00000001) */
  {
    const size_t shape[1] = {2};
    const uint32_t a_bits[2] = {0x00000001u, 0x007FFFFFu};
    const uint32_t b_bits[2] = {0x00000001u, 0x00000001u};
    const uint32_t want_bits[2] = {0x00000002u, 0x00800000u};
    float a_data[2] = {0.0f, 0.0f};
    float b_data[2] = {0.0f, 0.0f};
    float c_data[2] = {0.0f, 0.0f};
    uint32_t got_bits[2] = {0u, 0u};
    char det[256];
    sonnx_tensor a = {.data = a_data, .ndim = 1, .shape = shape};
    sonnx_tensor b = {.data = b_data, .ndim = 1, .shape = shape};
    sonnx_tensor c = {.data = c_data, .ndim = 1, .shape = shape};
    memcpy(&a_data[0], &a_bits[0], sizeof(float));
    memcpy(&a_data[1], &a_bits[1], sizeof(float));
    memcpy(&b_data[0], &b_bits[0], sizeof(float));
    memcpy(&b_data[1], &b_bits[1], sizeof(float));
    sonnx_add_f32(&a, &b, &c);
    memcpy(&got_bits[0], &c_data[0], sizeof(float));
    memcpy(&got_bits[1], &c_data[1], sizeof(float));
    if (got_bits[0] == want_bits[0] && got_bits[1] == want_bits[1]) {
      report("f32_subnormal_bit_exact", 1, "");
    } else {
      snprintf(det, sizeof det,
               "expected bits %08x %08x, got %08x %08x",
               (unsigned)want_bits[0], (unsigned)want_bits[1],
               (unsigned)got_bits[0], (unsigned)got_bits[1]);
      report("f32_subnormal_bit_exact", 0, det);
    }
  }

  /* 6. f32, rank-0 operands: one element, shape is empty. */
  {
    float a_data[1] = {1.5f};
    float b_data[1] = {2.25f};
    float c_data[1] = {0.0f};
    const float want[1] = {3.75f};
    const int want_nan[1] = {0};
    sonnx_tensor a = {.data = a_data, .ndim = 0, .shape = NULL};
    sonnx_tensor b = {.data = b_data, .ndim = 0, .shape = NULL};
    sonnx_tensor c = {.data = c_data, .ndim = 0, .shape = NULL};
    sonnx_add_f32(&a, &b, &c);
    check_f32("f32_rank0", c_data, want, want_nan, 1);
  }

  /* 7. f32, a zero-sized dimension: C is empty, nothing is written. */
  {
    const size_t shape[2] = {0, 4};
    float a_data[1] = {1.0f};
    float b_data[1] = {1.0f};
    float c_data[2] = {123.0f, -7.0f};
    char det[256];
    sonnx_tensor a = {.data = a_data, .ndim = 2, .shape = shape};
    sonnx_tensor b = {.data = b_data, .ndim = 2, .shape = shape};
    sonnx_tensor c = {.data = c_data, .ndim = 2, .shape = shape};
    sonnx_add_f32(&a, &b, &c);
    if (c_data[0] == 123.0f && c_data[1] == -7.0f) {
      report("f32_zero_sized_dim_untouched", 1, "");
    } else {
      snprintf(det, sizeof det, "expected sentinels 123 and -7, got %g and %g",
               (double)c_data[0], (double)c_data[1]);
      report("f32_zero_sized_dim_untouched", 0, det);
    }
