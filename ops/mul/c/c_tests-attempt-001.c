#include "sonnx_kernels.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void report(const char *name, int ok, const char *detail) {
  if (ok) {
    printf("ok %s\n", name);
  } else {
    printf("FAIL %s: %s\n", name, detail);
    failures += 1;
  }
}

/* ---------------------------------------------------------------- float32 */

static int f32_same(float got, float exp) {
  uint32_t g, e;
  if (isnan(exp)) {
    return isnan(got) ? 1 : 0;
  }
  if (exp == 0.0f) {
    return (got == 0.0f) && ((signbit(got) != 0) == (signbit(exp) != 0));
  }
  memcpy(&g, &got, sizeof g);
  memcpy(&e, &exp, sizeof e);
  return g == e;
}

static void check_f32(const char *name, const float *got, const float *exp, size_t n) {
  char msg[256];
  for (size_t i = 0; i < n; i++) {
    if (!f32_same(got[i], exp[i])) {
      snprintf(msg, sizeof msg, "expected %a at index %zu, got %a",
               (double)exp[i], i, (double)got[i]);
      report(name, 0, msg);
      return;
    }
  }
  report(name, 1, "");
}

/* ---------------------------------------------------------------- float64 */

static int f64_same(double got, double exp) {
  uint64_t g, e;
  if (isnan(exp)) {
    return isnan(got) ? 1 : 0;
  }
  if (exp == 0.0) {
    return (got == 0.0) && ((signbit(got) != 0) == (signbit(exp) != 0));
  }
  memcpy(&g, &got, sizeof g);
  memcpy(&e, &exp, sizeof e);
  return g == e;
}

static void check_f64(const char *name, const double *got, const double *exp, size_t n) {
  char msg[256];
  for (size_t i = 0; i < n; i++) {
    if (!f64_same(got[i], exp[i])) {
      snprintf(msg, sizeof msg, "expected %a at index %zu, got %a", exp[i], i, got[i]);
      report(name, 0, msg);
      return;
    }
  }
  report(name, 1, "");
}

/* --------------------------------------------------------------- integers */

#define DEF_CHECK_INT(FN, T, CAST, FMT)                                        \
  static void FN(const char *name, const T *got, const T *exp, size_t n) {     \
    char msg[256];                                                             \
    for (size_t i = 0; i < n; i++) {                                           \
      if (got[i] != exp[i]) {                                                  \
        snprintf(msg, sizeof msg, "expected " FMT " at index %zu, got " FMT,   \
                 CAST exp[i], i, CAST got[i]);                                 \
        report(name, 0, msg);                                                  \
        return;                                                                \
      }                                                                        \
    }                                                                          \
    report(name, 1, "");                                                       \
  }

DEF_CHECK_INT(check_i8, int8_t, (int), "%d")
DEF_CHECK_INT(check_i16, int16_t, (int), "%d")
DEF_CHECK_INT(check_i32, int32_t, (long long), "%lld")
DEF_CHECK_INT(check_i64, int64_t, (long long), "%lld")
DEF_CHECK_INT(check_u8, uint8_t, (unsigned), "%u")
DEF_CHECK_INT(check_u16, uint16_t, (unsigned), "%u")
DEF_CHECK_INT(check_u32, uint32_t, (unsigned long long), "%llu")
DEF_CHECK_INT(check_u64, uint64_t, (unsigned long long), "%llu")

/* ------------------------------------------------------------- f32 cases */

static void test_f32_example1(void) {
  const size_t shape[1] = {4};
  float a_data[4] = {3.0f, 4.5f, 16.0f, 0.5f};
  float b_data[4] = {3.0f, -2.0f, 0.25f, 4.0f};
  float c_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const float expected[4] = {9.0f, -9.0f, 4.0f, 2.0f};
  sonnx_tensor a = {.data = a_data, .ndim = 1, .shape = shape};
  sonnx_tensor b = {.data = b_data, .ndim = 1, .shape = shape};
  sonnx_tensor c = {.data = c_data, .ndim = 1, .shape = shape};

  sonnx_mul_f32(&a, &b, &c);
  check_f32("f32 example 1 (exact products)", c_data, expected, 4);
}

static void test_f32_example2(void) {
  const size_t shape[1] = {6};
  float a_data[6] = {0.0f, INFINITY, 0.0f, -0.0f, -INFINITY, NAN};
  float b_data[6] = {INFINITY, -INFINITY, -3.0f, -2.0f, 2.0f, 1.0f};
  float c_data[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const float expected[6] = {NAN, -INFINITY, -0.0f, 0.0f, -INFINITY, NAN};
  sonnx_tensor a = {.data = a_data, .ndim = 1, .shape = shape};
  sonnx_tensor b = {.data = b_data, .ndim = 1, .shape = shape};
  sonnx_tensor c = {.data = c_data, .ndim = 1, .shape = shape};

  sonnx_mul_f32(&a, &b, &c);
  check_f32("f32 example 2 (zero, inf, nan, signed zeros)", c_data, expected, 6);
}

static void test_f32_multi_element(void) {
  const size_t shape[2] = {2, 3};
  float a_data[6] = {1.5f, -2.0f, 0.0f, 8.0f, -1.0f, 0.5f};
  float b_data[6] = {-2.0f, -0.5f, 5.0f, 0.25f, 0.0f, 4.0f};
  float c_data[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const float expected[6] = {-3.0f, 1.0f, 0.0f, 2.0f, -0.0f, 2.0f};
  sonnx_tensor a = {.data = a_data, .ndim = 2, .shape = shape};
  sonnx_tensor b = {.data = b_data, .ndim = 2, .shape = shape};
  sonnx_tensor c = {.data = c_data, .ndim = 2, .shape = shape};

  sonnx_mul_f32(&a, &b, &c);
  check_f32("f32 shape 2x3 (negative, cancelling, signed zero)", c_data, expected, 6);
}

static void test_f32_zero_sized(void) {
  const size_t shape[2] = {0, 3};
  float a_data[2] = {1.0f, 2.0f};
  float b_data[2] = {3.0f, 4.0f};
  float c_data[2] = {42.0f, 42.0f};
  const float expected[2] = {42.0f, 42.0f};
  sonnx_tensor a = {.data = a_data, .ndim = 2, .shape = shape};
  sonnx_tensor b = {.data = b_data, .ndim = 2, .shape = shape};
  sonnx_tensor c = {.data = c_data, .ndim = 2, .shape = shape};

  sonnx_mul_f32(&a, &b, &c);
  check_f32("f32 zero-sized dimension writes nothing", c_data, expected, 2);
}

static void test_f32_underflow(void) {
  const size_t shape[1] = {5};
  float a_data[5] = {0x1p-149f, -0x1p-149f, 0x1p-149f, 0x1p-126f, -0x1p-126f};
  float b_data[5] = {0.5f, 0.5f, 1.0f, 0.25f, 0.25f};
  float c_data[5] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  const float expected[5] = {0.0f, -0.0f, 0x1p-149f, 0x1p-128f, -0x1p-128f};
  sonnx_tensor a = {.data = a_data, .ndim = 1, .shape = shape};
  sonnx_tensor b = {.data = b_data, .ndim = 1, .shape = shape};
  sonnx_tensor c = {.data = c_data, .ndim = 1, .shape = shape};

  sonnx_mul_f32(&a, &b, &c);
  check_f32("f32 underflow to signed zeros and subnormals", c_data, expected, 5);
}

static void test_f32_overflow(void) {
  const size_t shape[1] = {3};
  float a_data[3] = {0x1p127f, -0x1p127f, 0x1.fffffep127f};
  float b_data[3] = {2.0f, 2.0f, 2.0f};
  float c_data[3] = {0.0f, 0.0f, 0.0f};
  const float expected[3] = {INFINITY, -INFINITY, INFINITY};
  sonnx_tensor a = {.data = a_data, .ndim = 1, .shape = shape};
  sonnx_tensor b = {.data = b_data, .ndim = 1, .shape = shape};
  sonnx_tensor c = {.data = c_data, .ndim = 1, .shape = shape};

  sonnx_mul_f32(&a, &b, &c);
  check_f32("f32 overflow to signed infinities", c_data, expected, 3);
}

static void test_f32_nan_propagation(void) {
  const size_t shape[1] = {5};
  float a_data[5] = {NAN, NAN, INFINITY, -INFINITY, NAN};
  float b_data[5] = {-2.0f, INFINITY, -INFINITY, -INFINITY, 0.0f};
  float c_data[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const float expected[5] = {NAN, NAN, -INFINITY, INFINITY, NAN};
  sonnx_tensor a = {.data = a_data, .ndim = 1, .shape = shape};
  sonnx_tensor b = {.data = b_data, .ndim = 1, .shape = shape};
  sonnx_tensor c = {.data = c_data, .ndim = 1, .shape = shape};

  sonnx_mul_f32(&a, &b, &c);
  check_f32("f32 NaN propagation and inf*inf rule of signs", c_data, expected, 5);
}

static void test_f32_rank0(void) {
  float a_data[1] = {1.5f};
  float b_data[1] = {-2.0f};
  float c_data[1] = {0.0f};
  const float expected[1] = {-3.0f};
  sonnx_tensor a = {.data = a_data, .ndim = 0, .shape = NULL};
  sonnx_tensor b = {.data = b_data, .ndim = 0, .shape = NULL};
  sonnx_tensor c = {.data = c_data, .ndim = 0, .shape = NULL};

  sonnx_mul_f32(&a, &b, &c);
  check_f32("f32 rank-0 tensor is one element", c_data, expected, 1);
}

/* ------------------------------------------------------------- f64 cases */

static void test_f64_example3(void) {
  const size_t shape[1] = {3};
  double a_data[3] = {0.1, 1e308, -1e-308};
  double b_data[3] = {0.2, 10.0, 1e-308};
  double c_data[3] = {0.0, 0.0, 0.0};
  const double expected[3] = {0.020000000000000004, INFINITY, -0.0};
  sonnx_tensor a = {.data = a_data, .ndim = 1, .shape = shape};
  sonnx_tensor b = {.data = b_data, .ndim = 1, .shape = shape};
  sonnx_tensor c = {.data = c_data, .ndim = 1, .shape = shape};

  sonnx_mul_f64(&a, &b, &c);
  check_f64("f64 example 3 (rounded, overflow, underflow)", c_data, expected, 3);
}

static void test_f64_special(void) {
  const size_t shape[1] = {6};
  double a_data[6] = {-INFINITY, INFINITY, 0.0, -0.0, -0.0, 1.5};
  double b_data[6] = {-INFINITY, -2.0, -INFINITY, -0.0, 0.0, 0.25};
  double c_data[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const double expected[6] = {INFINITY, -INFINITY, NAN, 0.0, -0.0, 0.375};
  sonnx_tensor a = {.data = a_data, .ndim = 1, .shape = shape};
  sonnx_tensor b = {.data = b_data, .ndim = 1, .shape = shape};
  sonnx_tensor c = {.data = c_data, .ndim = 1, .shape = shape};

  sonnx_mul_f64(&a, &b, &c);
  check_f64("f64 special values and signed zeros", c_data, expected, 6);
}

static void test_f64_exact(void) {
  const size_t shape[2] = {2, 2};
  double a_data[4] = {0.5, -2.5, 1e-320, -1e-320};
  double b_data[4] = {-3.0, 4.0, 1e-320, 1e-320};
  double c_data[4] = {0.0, 0.0, 0.0, 0.0};
  const double expected[4] = {-1.5, -10.0, 0.0, -0.0};
  sonnx_tensor a = {.data = a_data, .ndim = 2, .shape = shape};
  sonnx_tensor b = {.data = b_data, .ndim = 2, .shape = shape};
  sonnx_tensor c = {.data = c_data, .ndim = 2, .shape = shape};

  sonnx_mul_f64(&a, &b, &c);
  check_f64("f64 exact products and subnormal underflow", c_data, expected, 4);
}

/* --------------------------------------------------------- signed integers */

static void test_i8_example1(void) {
  const size_t shape[1] = {4};
  int8_t a_data[4] = {-6, 100, -100, 100};
  int8_t b_data[4] = {-3, 100, 100, 3};
  int8_t c_data[4] = {0, 0, 0, 0};
  const int8_t expected[4] = {18, 16, -16, 44};
  sonnx_tensor a = {.data = a_data, .ndim = 1, .shape = shape};
  sonnx_tensor b = {.data = b_data, .ndim = 1, .shape = shape};
  sonnx_tensor c = {.data = c_data, .ndim = 1, .shape = shape};

  sonnx_mul_i8(&a, &b, &c);
  check_i8("i8 example 1 (18, 16, -16, 44)", c_data, expected, 4);
}

static void test_i8_boundaries(void) {
  const size_t shape[1] = {6};
  int8_t a_data[6] = {(int8_t)INT8_MIN, (int8_t)INT8_MAX, (int8_t)INT8_MIN,
                      (int8_t)INT8_MAX, -1, 1};
  int8_t b_data[6] = {-1, (int8_t)INT8_MAX, (int8_t)INT8_MIN, 2, 1,
                      (int8_t)INT8_MIN};
  int8_t c_data[6] = {0, 0, 0, 0, 0, 0};
  const int8_t expected[6] = {(int8_t)INT8_MIN, 1, 0, -2, -1, (int8_t)INT8_MIN};
  sonnx_tensor a = {.data = a_data, .ndim = 1, .shape = shape};
  sonnx_tensor b = {.data = b_data, .ndim = 1, .shape = shape};
  sonnx_tensor c = {.data = c_data, .ndim = 1, .shape = shape};

  sonnx_mul_i8(&a, &b, &c);
  check_i8("i8 boundary products wrap modulo 256", c_data, expected, 6);
}

static void test_i16_boundaries(void) {
  const size_t shape[2] = {2, 3};
  int16_t a_data[6] = {(int16_t)INT16_MIN, (int16_t)INT16_MAX, 300, -300, 0, -1};
  int16_t b_data[6] = {-1, 2, 300, 300, -1, 1};
  int16_t c_data[6] = {0, 0, 0, 0, 0, 0};
  const int16_t expected[6] = {(int16_t)INT16_MIN, -2, 24464, -24464, 0, -1};
  sonnx_tensor a = {.data = a_data, .ndim = 2, .shape = shape};
  sonnx_tensor b = {.data = b_data, .ndim = 2, .shape = shape};
  sonnx_tensor c = {.data = c_data, .ndim = 2, .shape = shape};

  sonnx_mul_i16(&a, &b, &c);
  check_i16("i16 boundaries and 2x3 shape", c_data, expected, 6);
}

static void test_i32_boundaries(void) {
  const size_t shape[1] = {6};
  int32_t a_data[6] = {(int32_t)INT32_MIN, (int32_t)INT32_MAX,
                       (int32_t)INT32_MIN, 65536, 3, -7};
  int32_t b_data[6] = {-1, 2, 2, 65536, 5, 6};
  int32_t c_data[6] = {0, 0, 0, 0, 0, 0};
  const int32_t expected[6] = {(int32_t)INT32_MIN, -2, 0, 0, 15, -42};
  sonnx_tensor a = {.data = a_data, .ndim = 1, .shape = shape};
  sonnx_tensor b = {.data = b_data, .ndim = 1, .shape = shape};
  sonnx_tensor c = {.data = c_data, .ndim = 1, .shape = shape};

  sonnx_mul_i32(&a, &b, &c);
  check_i32("i32 boundary products wrap modulo 2^32", c_data, expected, 6);
}

static void test_i64_boundaries(void) {
  const size_t shape[1] = {5};
  int64_t a_data[5] = {(int64_t)INT64_MIN, (int64_t)INT64_MAX,
                       (int64_t)INT64_MIN, 1, -3};
  int64_t b_data[5] = {-1, (int64_t)INT64_MAX, 2, (int64_t)INT64_MIN, 5};
  int64_t c_data[5] = {0, 0, 0, 0, 0};
  const int64_t expected[5] = {(int64_t)INT64_MIN, 1, 0, (int64_t)INT64_MIN, -15};
  sonnx_tensor a = {.data = a_data, .ndim = 1, .shape = shape};
  sonnx_tensor b = {.data = b_data, .ndim = 1, .shape = shape};
  sonnx_tensor c = {.data = c_data, .ndim = 1, .shape = shape};

  sonnx_mul_i64(&a, &b, &c);
  check_i64("i64 boundary products wrap modulo 2^64", c_data, expected, 5);
}

/* ------------------------------------------------------- unsigned integers */

static void test_u8_example1(void) {
  const size_t shape[1] = {4};
  uint8_t a_data[4] = {6, 200, 35, 16};
  uint8_t b_data[4] = {3, 100, 5, 16};
  uint8_t c_data[4] = {0, 0, 0, 0};
  const uint8_t expected[4] = {18, 32, 175, 0};
  sonnx_tensor a = {.data = a_data, .ndim = 1, .shape = shape};
  sonnx_tensor b = {.data = b_data, .ndim = 1, .shape = shape};
  sonnx_tensor c = {.data = c_data, .ndim = 1, .shape = shape};

  sonnx_mul_u8(&a, &b, &c);
  check_u8("u8 example 1 (18, 32, 175, 0)", c_data, expected, 4);
}

static void test_u8_boundaries(void) {
  const size_t shape[1] = {5};
  uint8_t a_data[5] = {UINT8_MAX, UINT8_MAX, 0, 128, 1};
  uint8_t b_data[5] = {UINT8_MAX, 2, UINT8_MAX, 2, UINT8_MAX};
  uint8_t c_data[5] = {0, 0, 0, 0, 0};
  const uint8_t expected[5] = {1, 254, 0, 0, 255};
  sonnx_tensor a = {.data = a_data, .ndim = 1, .shape = shape};
  sonnx_tensor b = {.data = b_data, .ndim = 1, .shape = shape};
  sonnx_tensor c = {.data = c_data, .ndim = 1, .shape = shape};

  sonnx_mul_u8(&a, &b, &c);
  check_u8("u8 boundary products wrap modulo 256", c_data, expected, 5);
}

static void test_u16_boundaries(void) {
  const size_t shape[1] = {4};
  uint16_t a_data[4] = {UINT16_MAX, UINT16_MAX, 256, 1000};
  uint16_t b_data[4] = {UINT16_MAX, 2, 256, 1000};
  uint16_t c_data[4] = {0, 0, 0, 0};
  const uint16_t expected[4] = {1, 65534, 0, 16960};
  sonnx_tensor a = {.data = a_data, .ndim = 1, .shape = shape};
  sonnx_tensor b = {.data = b_data, .ndim = 1, .shape = shape};
  sonnx_tensor c = {.data = c_data, .ndim = 1, .shape = shape};

  sonnx_mul_u16(&a, &b, &c);
  check_u16("u16 boundary products wrap modulo 2^16", c_data, expected, 4);
}

static void test_u32_boundaries(void) {
  const size_t shape[1] = {4};
  uint32_t a_data[4] = {UINT32_MAX, 3u, 65536u, 1u};
  uint32_t b_data[4] = {UINT32_MAX, 1431655765u, 65536u, UINT32_MAX};
  uint32_t c_data[4] = {0u, 0u, 0u, 0u};
  const uint32_t expected[4] = {1u, UINT32_MAX, 0u, UINT32_MAX};
  sonnx_tensor a = {.data = a_data, .ndim = 1, .shape = shape};
  sonnx_tensor b = {.data = b_data, .ndim = 1, .shape = shape};
  sonnx_tensor c = {.data = c_data, .ndim = 1, .shape = shape};

  sonnx_mul_u32(&a, &b, &c);
  check_u32("u32 boundary products wrap modulo 2^32", c_data, expected, 4);
}

static void test_u64_boundaries(void) {
  const size_t shape[1] = {4};
  uint64_t a_data[4] = {UINT64_MAX, 4294967296ull, UINT64_MAX, 2ull};
  uint64_t b_data[4] = {UINT64_MAX, 4294967296ull, 2ull, 9223372036854775808ull};
  uint64_t c_data[4] = {0ull, 0ull, 0ull, 0ull};
  const uint64_t expected[4] = {1ull, 0ull, UINT64_MAX - 1ull, 0ull};
  sonnx_tensor a = {.data = a_data, .ndim = 1, .shape = shape};
  sonnx_tensor b = {.data = b_data, .ndim = 1, .shape = shape};
  sonnx_tensor c = {.data = c_data, .ndim = 1, .shape = shape};

  sonnx_mul_u64(&a, &b, &c);
  check_u64("u64 boundary products wrap modulo 2^64", c_data, expected, 4);
}

int main(void) {
  test_f32_example1();
  test_f32_example2();
  test_f32_multi_element();
  test_f32_zero_sized();
  test_f32_underflow();
  test_f32_overflow();
  test_f32_nan_propagation();
  test_f32_rank0();

  test_f64_example3();
  test_f64_special();
  test_f64_exact();

  test_i8_example1();
  test_i8_boundaries();
  test_i16_boundaries();
  test_i32_boundaries();
  test_i64_boundaries();

  test_u8_example1();
  test_u8_boundaries();
  test_u16_boundaries();
  test_u32_boundaries();
  test_u64_boundaries();

  if (failures != 0) {
    fprintf(stderr, "%d failing case(s)\n", failures);
    return 1;
  }
  return 0;
}
