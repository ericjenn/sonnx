#include "sonnx_kernels.h"
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <limits.h>
#include <string.h>

static int failures = 0;

static void report_ok(const char *name) {
  printf("ok %s\n", name);
}

static void report_fail_f32(const char *name, float expected, float got) {
  printf("FAIL %s: expected %.9g, got %.9g\n", name, (double)expected, (double)got);
  failures++;
}

static void report_fail_f64(const char *name, double expected, double got) {
  printf("FAIL %s: expected %.17g, got %.17g\n", name, expected, got);
  failures++;
}

static void report_fail_i64(const char *name, long long expected, long long got) {
  printf("FAIL %s: expected %lld, got %lld\n", name, expected, got);
  failures++;
}

static void report_fail_u64(const char *name, unsigned long long expected, unsigned long long got) {
  printf("FAIL %s: expected %llu, got %llu\n", name, expected, got);
  failures++;
}

/* ---- f32 cases ---- */

static void test_f32_example1(void) {
  const size_t shape[2] = {3, 2};
  float a_data[6] = {3.0f, 4.5f, 16.0f, 1.0f, 25.5f, 24.25f};
  float b_data[6] = {3.0f, 2.0f, 4.0f, 0.0f, 5.0f, 4.0f};
  float c_data[6] = {0};
  float exp[6] = {0.0f, 2.5f, 12.0f, 1.0f, 20.5f, 20.25f};
  sonnx_tensor a = {a_data, 2, shape};
  sonnx_tensor b = {b_data, 2, shape};
  sonnx_tensor c = {c_data, 2, shape};
  sonnx_sub_f32(&a, &b, &c);
  for (int i = 0; i < 6; i++) {
    if (memcmp(&c_data[i], &exp[i], sizeof(float)) != 0) {
      report_fail_f32("f32_example1", exp[i], c_data[i]);
      return;
    }
  }
  report_ok("f32_example1");
}

static void test_f32_example2(void) {
  const size_t shape[1] = {4};
  float a_data[4] = {1.0f, INFINITY, -0.0f, 0.0f};
  float b_data[4] = {1.0f, INFINITY, 0.0f, -0.0f};
  float c_data[4] = {0};
  sonnx_tensor a = {a_data, 1, shape};
  sonnx_tensor b = {b_data, 1, shape};
  sonnx_tensor c = {c_data, 1, shape};
  sonnx_sub_f32(&a, &b, &c);
  /* c[0] = +0 */
  if (c_data[0] != 0.0f || signbit(c_data[0])) {
    report_fail_f32("f32_example2[0]", 0.0f, c_data[0]);
    return;
  }
  /* c[1] = NaN */
  if (!isnan(c_data[1])) {
    report_fail_f32("f32_example2[1]", NAN, c_data[1]);
    return;
  }
  /* c[2] = -0 */
  if (c_data[2] != 0.0f || !signbit(c_data[2])) {
    report_fail_f32("f32_example2[2]", -0.0f, c_data[2]);
    return;
  }
  /* c[3] = +0 */
  if (c_data[3] != 0.0f || signbit(c_data[3])) {
    report_fail_f32("f32_example2[3]", 0.0f, c_data[3]);
    return;
  }
  report_ok("f32_example2");
}

static void test_f32_inf(void) {
  const size_t shape[1] = {4};
  float a_data[4] = {INFINITY, -INFINITY, INFINITY, -INFINITY};
  float b_data[4] = {1.0f, 1.0f, -INFINITY, INFINITY};
  float c_data[4] = {0};
  sonnx_tensor a = {a_data, 1, shape};
  sonnx_tensor b = {b_data, 1, shape};
  sonnx_tensor c = {c_data, 1, shape};
  sonnx_sub_f32(&a, &b, &c);
  if (!isinf(c_data[0]) || signbit(c_data[0])) { report_fail_f32("f32_inf[0]", INFINITY, c_data[0]); return; }
  if (!isinf(c_data[1]) || !signbit(c_data[1])) { report_fail_f32("f32_inf[1]", -INFINITY, c_data[1]); return; }
  if (!isinf(c_data[2]) || signbit(c_data[2])) { report_fail_f32("f32_inf[2]", INFINITY, c_data[2]); return; }
  if (!isinf(c_data[3]) || !signbit(c_data[3])) { report_fail_f32("f32_inf[3]", -INFINITY, c_data[3]); return; }
  report_ok("f32_inf");
}

static void test_f32_nan(void) {
  const size_t shape[1] = {2};
  float a_data[2] = {NAN, 1.0f};
  float b_data[2] = {1.0f, NAN};
  float c_data[2] = {0};
  sonnx_tensor a = {a_data, 1, shape};
  sonnx_tensor b = {b_data, 1, shape};
  sonnx_tensor c = {c_data, 1, shape};
  sonnx_sub_f32(&a, &b, &c);
  if (!isnan(c_data[0])) { report_fail_f32("f32_nan[0]", NAN, c_data[0]); return; }
  if (!isnan(c_data[1])) { report_fail_f32("f32_nan[1]", NAN, c_data[1]); return; }
  report_ok("f32_nan");
}

static void test_f32_subnormal(void) {
  const size_t shape[1] = {2};
  /* smallest subnormal - 0 = smallest subnormal */
  float a_data[2] = {0x1p-149f, 0x1p-126f};
  float b_data[2] = {0.0f, 0x1p-149f};
  float c_data[2] = {0};
  sonnx_tensor a = {a_data, 1, shape};
  sonnx_tensor b = {b_data, 1, shape};
  sonnx_tensor c = {c_data, 1, shape};
  sonnx_sub_f32(&a, &b, &c);
  float e0 = 0x1p-149f;
  float e1 = 0x1.fffffcp-127f; /* 0x1p-126 - 0x1p-149 */
  if (memcmp(&c_data[0], &e0, sizeof(float)) != 0) { report_fail_f32("f32_subnormal[0]", e0, c_data[0]); return; }
  if (memcmp(&c_data[1], &e1, sizeof(float)) != 0) { report_fail_f32("f32_subnormal[1]", e1, c_data[1]); return; }
  report_ok("f32_subnormal");
}

static void test_f32_overflow(void) {
  const size_t shape[1] = {2};
  float a_data[2] = {3.0e38f, -3.0e38f};
  float b_data[2] = {-3.0e38f, 3.0e38f};
  float c_data[2] = {0};
  sonnx_tensor a = {a_data, 1, shape};
  sonnx_tensor b = {b_data, 1, shape};
  sonnx_tensor c = {c_data, 1, shape};
  sonnx_sub_f32(&a, &b, &c);
  if (!isinf(c_data[0]) || signbit(c_data[0])) { report_fail_f32("f32_overflow[0]", INFINITY, c_data[0]); return; }
  if (!isinf(c_data[1]) || !signbit(c_data[1])) { report_fail_f32("f32_overflow[1]", -INFINITY, c_data[1]); return; }
  report_ok("f32_overflow");
}

static void test_f32_zero_size(void) {
  const size_t shape[2] = {0, 3};
  float a_data[1] = {0};
  float b_data[1] = {0};
  float c_data[1] = {42.0f};
  sonnx_tensor a = {a_data, 2, shape};
  sonnx_tensor b = {b_data, 2, shape};
  sonnx_tensor c = {c_data, 2, shape};
  sonnx_sub_f32(&a, &b, &c);
  if (c_data[0] != 42.0f) { report_fail_f32("f32_zero_size", 42.0f, c_data[0]); return; }
  report_ok("f32_zero_size");
}

/* ---- f64 cases ---- */

static void test_f64_example3(void) {
  const size_t shape[1] = {2};
  double a_data[2] = {0.3, 1.0};
  double b_data[2] = {0.1, 1.0};
  double c_data[2] = {0};
  sonnx_tensor a = {a_data, 1, shape};
  sonnx_tensor b = {b_data, 1, shape};
  sonnx_tensor c = {c_data, 1, shape};
  sonnx_sub_f64(&a, &b, &c);
  double e0 = 0.19999999999999998;
  if (memcmp(&c_data[0], &e0, sizeof(double)) != 0) { report_fail_f64("f64_example3[0]", e0, c_data[0]); return; }
  if (c_data[1] != 0.0 || signbit(c_data[1])) { report_fail_f64("f64_example3[1]", 0.0, c_data[1]); return; }
  report_ok("f64_example3");
}

static void test_f64_special(void) {
  const size_t shape[1] = {4};
  double a_data[4] = {-0.0, 0.0, -0.0, 0.0};
  double b_data[4] = {0.0, -0.0, -0.0, 0.0};
  double c_data[4] = {0};
  sonnx_tensor a = {a_data, 1, shape};
  sonnx_tensor b = {b_data, 1, shape};
  sonnx_tensor c = {c_data, 1, shape};
  sonnx_sub_f64(&a, &b, &c);
  if (c_data[0] != 0.0 || !signbit(c_data[0])) { report_fail_f64("f64_special[0]", -0.0, c_data[0]); return; }
  if (c_data[1] != 0.0 || signbit(c_data[1])) { report_fail_f64("f64_special[1]", 0.0, c_data[1]); return; }
  if (c_data[2] != 0.0 || signbit(c_data[2])) { report_fail_f64("f64_special[2]", 0.0, c_data[2]); return; }
  if (c_data[3] != 0.0 || signbit(c_data[3])) { report_fail_f64("f64_special[3]", 0.0, c_data[3]); return; }
  report_ok("f64_special");
}

static void test_f64_rank0(void) {
  double a_data[1] = {5.0};
  double b_data[1] = {2.5};
  double c_data[1] = {0};
  sonnx_tensor a = {a_data, 0, NULL};
  sonnx_tensor b = {b_data, 0, NULL};
  sonnx_tensor c = {c_data, 0, NULL};
  sonnx_sub_f64(&a, &b, &c);
  if (c_data[0] != 2.5) { report_fail_f64("f64_rank0", 2.5, c_data[0]); return; }
  report_ok("f64_rank0");
}

/* ---- i8 cases ---- */

static void test_i8_example1(void) {
  const size_t shape[1] = {3};
  int8_t a_data[3] = {6, 5, -35};
  int8_t b_data[3] = {3, 3, 3};
  int8_t c_data[3] = {0};
  int8_t exp[3] = {3, 2, -38};
  sonnx_tensor a = {a_data, 1, shape};
  sonnx_tensor b = {b_data, 1, shape};
  sonnx_tensor c = {c_data, 1, shape};
  sonnx_sub_i8(&a, &b, &c);
  for (int i = 0; i < 3; i++) {
    if (c_data[i] != exp[i]) { report_fail_i64("i8_example1", exp[i], c_data[i]); return; }
  }
  report_ok("i8_example1");
}

static void test_i8_example2(void) {
  const size_t shape[1] = {3};
  int8_t a_data[3] = {127, -128, 0};
  int8_t b_data[3] = {-1, 1, 1};
  int8_t c_data[3] = {0};
  int8_t exp[3] = {-128, 127, -1};
  sonnx_tensor a = {a_data, 1, shape};
  sonnx_tensor b = {b_data, 1, shape};
  sonnx_tensor c = {c_data, 1, shape};
  sonnx_sub_i8(&a, &b, &c);
  for (int i = 0; i < 3; i++) {
    if (c_data[i] != exp[i]) { report_fail_i64("i8_example2", exp[i], c_data[i]); return; }
  }
  report_ok("i8_example2");
}

/* ---- i16 cases ---- */

static void test_i16_bounds(void) {
  const size_t shape[1] = {4};
  int16_t a_data[4] = {INT16_MAX, INT16_MIN, 0, -1};
  int16_t b_data[4] = {-1, 1, 0, 1};
  int16_t c_data[4] = {0};
  int16_t exp[4] = {(int16_t)-32768, (int16_t)32767, 0, -2};
  sonnx_tensor a = {a_data, 1, shape};
  sonnx_tensor b = {b_data, 1, shape};
  sonnx_tensor c = {c_data, 1, shape};
  sonnx_sub_i16(&a, &b, &c);
  for (int i = 0; i < 4; i++) {
    if (c_data[i] != exp[i]) { report_fail_i64("i16_bounds", exp[i], c_data[i]); return; }
  }
  report_ok("i16_bounds");
}

/* ---- i32 cases ---- */

static void test_i32_bounds(void) {
  const size_t shape[1] = {4};
  int32_t a_data[4] = {INT32_MAX, INT32_MIN, 0, -1};
  int32_t b_data[4] = {-1, 1, 0, 1};
  int32_t c_data[4] = {0};
  int32_t exp[4] = {INT32_MIN, INT32_MAX, 0, -2};
  sonnx_tensor a = {a_data, 1, shape};
  sonnx_tensor b = {b_data, 1, shape};
  sonnx_tensor c = {c_data, 1, shape};
  sonnx_sub_i32(&a, &b, &c);
  for (int i = 0; i < 4; i++) {
    if (c_data[i] != exp[i]) { report_fail_i64("i32_bounds", exp[i], c_data[i]); return; }
  }
  report_ok("i32_bounds");
}

/* ---- i64 cases ---- */

static void test_i64_bounds(void) {
  const size_t shape[1] = {4};
  int64_t a_data[4] = {INT64_MAX, INT64_MIN, 0, -1};
  int64_t b_data[4] = {-1, 1, 0, 1};
  int64_t c_data[4] = {0};
  int64_t exp[4] = {INT64_MIN, INT64_MAX, 0, -2};
  sonnx_tensor a = {a_data, 1, shape};
  sonnx_tensor b = {b_data, 1, shape};
  sonnx_tensor c = {c_data, 1, shape};
  sonnx_sub_i64(&a, &b, &c);
  for (int i = 0; i < 4; i++) {
    if (c_data[i] != exp[i]) { report_fail_i64("i64_bounds", (long long)exp[i], (long long)c_data[i]); return; }
  }
  report_ok("i64_bounds");
}

/* ---- u8 cases ---- */

static void test_u8_example1(void) {
  const size_t shape[1] = {3};
  uint8_t a_data[3] = {0, 255, 4};
  uint8_t b_data[3] = {1, 1, 7};
  uint8_t c_data[3] = {0};
  uint8_t exp[3] = {255, 254, 253};
  sonnx_tensor a = {a_data, 1, shape};
  sonnx_tensor b = {b_data, 1, shape};
  sonnx_tensor c = {c_data, 1, shape};
  sonnx_sub_u8(&a, &b, &c);
  for (int i = 0; i < 3; i++) {
    if (c_data[i] != exp[i]) { report_fail_u64("u8_example1", exp[i], c_data[i]); return; }
  }
  report_ok("u8_example1");
}

/* ---- u16 cases ---- */

static void test_u16_bounds(void) {
  const size_t shape[1] = {3};
  uint16_t a_data[3] = {0, UINT16_MAX, 0};
  uint16_t b_data[3] = {1, 1, 0};
  uint16_t c_data[3] = {0};
  uint16_t exp[3] = {UINT16_MAX, (uint16_t)(UINT16_MAX - 1), 0};
  sonnx_tensor a = {a_data, 1, shape};
  sonnx_tensor b = {b_data, 1, shape};
  sonnx_tensor c = {c_data, 1, shape};
  sonnx_sub_u16(&a, &b, &c);
  for (int i = 0; i < 3; i++) {
    if (c_data[i] != exp[i]) { report_fail_u64("u16_bounds", exp[i], c_data[i]); return; }
  }
  report_ok("u16_bounds");
}

/* ---- u32 cases ---- */

static void test_u32_bounds(void) {
  const size_t shape[1] = {3};
  uint32_t a_data[3] = {0, UINT32_MAX, 0};
  uint32_t b_data[3] = {1, 1, 0};
  uint32_t c_data[3] = {0};
  uint32_t exp[3] = {UINT32_MAX, UINT32_MAX - 1, 0};
  sonnx_tensor a = {a_data, 1, shape};
  sonnx_tensor b = {b_data, 1, shape};
  sonnx_tensor c = {c_data, 1, shape};
  sonnx_sub_u32(&a, &b, &c);
  for (int i = 0; i < 3; i++) {
    if (c_data[i] != exp[i]) { report_fail_u64("u32_bounds", exp[i], c_data[i]); return; }
  }
  report_ok("u32_bounds");
}

/* ---- u64 cases ---- */

static void test_u64_bounds(void) {
  const size_t shape[1] = {3};
  uint64_t a_data[3] = {0, UINT64_MAX, 0};
  uint64_t b_data[3] = {1, 1, 0};
  uint64_t c_data[3] = {0};
  uint64_t exp[3] = {UINT64_MAX, UINT64_MAX - 1, 0};
  sonnx_tensor a = {a_data, 1, shape};
  sonnx_tensor b = {b_data, 1, shape};
  sonnx_tensor c = {c_data, 1, shape};
  sonnx_sub_u64(&a, &b, &c);
  for (int i = 0; i < 3; i++) {
    if (c_data[i] != exp[i]) { report_fail_u64("u64_bounds", exp[i], c_data[i]); return; }
  }
  report_ok("u64_bounds");
}

int main(void) {
  test_f32_example1();
  test_f32_example2();
  test_f32_inf();
  test_f32_nan();
  test_f32_subnormal();
  test_f32_overflow();
  test_f32_zero_size();
  test_f64_example3();
  test_f64_special();
  test_f64_rank0();
  test_i8_example1();
  test_i8_example2();
  test_i16_bounds();
  test_i32_bounds();
  test_i64_bounds();
  test_u8_example1();
  test_u16_bounds();
  test_u32_bounds();
  test_u64_bounds();
  return failures == 0 ? 0 : 1;
}
