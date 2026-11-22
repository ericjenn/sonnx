#include "sonnx_kernels.h"
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <limits.h>
#include <string.h>

static int failures = 0;

static void report(const char *name, int ok) {
  if (ok) {
    printf("ok %s\n", name);
  } else {
    printf("FAIL %s\n", name);
    failures++;
  }
}

static void report_int(const char *name, int ok, long long expected, long long got) {
  if (ok) {
    printf("ok %s\n", name);
  } else {
    printf("FAIL %s: expected %lld, got %lld\n", name, expected, got);
    failures++;
  }
}

static void report_f32(const char *name, int ok, float expected, float got) {
  if (ok) {
    printf("ok %s\n", name);
  } else {
    printf("FAIL %s: expected %g (%a), got %g (%a)\n", name,
           (double)expected, (double)expected, (double)got, (double)got);
    failures++;
  }
}

static void report_f64(const char *name, int ok, double expected, double got) {
  if (ok) {
    printf("ok %s\n", name);
  } else {
    printf("FAIL %s: expected %g (%a), got %g (%a)\n", name,
           expected, expected, got, got);
    failures++;
  }
}

/* bit-pattern view of a float/double */
static uint32_t f32_bits(float x) { uint32_t u; memcpy(&u, &x, sizeof u); return u; }
static uint64_t f64_bits(double x) { uint64_t u; memcpy(&u, &x, sizeof u); return u; }

int main(void) {

  /* ------------------------------------------------------------------ */
  /* f32                                                                */
  /* ------------------------------------------------------------------ */

  /* Example 1, replayed exactly (every difference exact in f32). */
  {
    const size_t shape[2] = {3, 2};
    float A[6] = {3.0f, 4.5f, 16.0f, 1.0f, 25.5f, 24.25f};
    float B[6] = {3.0f, 2.0f, 4.0f, 0.0f, 5.0f, 4.0f};
    float C[6] = {0};
    float E[6] = {0.0f, 2.5f, 12.0f, 1.0f, 20.5f, 20.25f};
    sonnx_tensor a = {A, 2, shape}, b = {B, 2, shape}, c = {C, 2, shape};
    sonnx_sub_f32(&a, &b, &c);
    int ok = 1;
    for (int i = 0; i < 6; i++) {
      if (f32_bits(C[i]) != f32_bits(E[i])) ok = 0;
    }
    report("f32_example1", ok);
  }

  /* Example 2: 1-1=+0, inf-inf=NaN, (-0)-(+0)=-0, (+0)-(-0)=+0 */
  {
    const size_t shape[1] = {4};
    float A[4] = {1.0f, INFINITY, -0.0f, 0.0f};
    float B[4] = {1.0f, INFINITY, 0.0f, -0.0f};
    float C[4] = {123.0f, 123.0f, 123.0f, 123.0f};
    sonnx_tensor a = {A, 1, shape}, b = {B, 1, shape}, c = {C, 1, shape};
    sonnx_sub_f32(&a, &b, &c);
    int ok = 1;
    if (!(C[0] == 0.0f && !signbit(C[0]))) ok = 0;      /* +0 */
    if (!isnan(C[1])) ok = 0;                            /* NaN */
    if (!(C[2] == 0.0f && signbit(C[2]))) ok = 0;        /* -0 */
    if (!(C[3] == 0.0f && !signbit(C[3]))) ok = 0;       /* +0 */
    report("f32_example2_special", ok);
  }

  /* Infinite results: +inf - finite = +inf; finite - -inf = +inf;
     finite - +inf = -inf; -inf - finite = -inf. */
  {
    const size_t shape[1] = {4};
    float A[4] = {INFINITY, 3.0f, 3.0f, -INFINITY};
    float B[4] = {3.0f, -INFINITY, INFINITY, 3.0f};
    float C[4] = {0};
    float E[4] = {INFINITY, INFINITY, -INFINITY, -INFINITY};
    sonnx_tensor a = {A, 1, shape}, b = {B, 1, shape}, c = {C, 1, shape};
    sonnx_sub_f32(&a, &b, &c);
    int ok = 1;
    for (int i = 0; i < 4; i++) {
      if (!(isinf(C[i]) && ((C[i] > 0) == (E[i] > 0)))) ok = 0;
    }
    report("f32_infinities", ok);
  }

  /* NaN propagation from an operand. */
  {
    const size_t shape[1] = {2};
    float A[2] = {NAN, 1.0f};
    float B[2] = {1.0f, NAN};
    float C[2] = {0.0f, 0.0f};
    sonnx_tensor a = {A, 1, shape}, b = {B, 1, shape}, c = {C, 1, shape};
    sonnx_sub_f32(&a, &b, &c);
    report("f32_nan_propagation", isnan(C[0]) && isnan(C[1]));
  }

  /* Same-sign infinities yield NaN both ways. */
  {
    const size_t shape[1] = {2};
    float A[2] = {INFINITY, -INFINITY};
    float B[2] = {INFINITY, -INFINITY};
    float C[2] = {0.0f, 0.0f};
    sonnx_tensor a = {A, 1, shape}, b = {B, 1, shape}, c = {C, 1, shape};
    sonnx_sub_f32(&a, &b, &c);
    report("f32_invalid_inf_minus_inf", isnan(C[0]) && isnan(C[1]));
  }

  /* Subnormal exact result: smallest normal minus smallest subnormal.
     FLT_MIN - 2^-149 = largest subnormal, exactly representable. */
  {
    const size_t shape[1] = {1};
    float A[1] = {5.877471754111438e-39f};   /* 2^-127, smallest normal f32 */
    float B[1] = {1.401298464324817e-45f};   /* 2^-149, smallest subnormal */
    float C[1] = {0.0f};
    float expected = 7.0064923216240854e-39f; /* largest subnormal 0x007FFFFF */
    sonnx_tensor a = {A, 1, shape}, b = {B, 1, shape}, c = {C, 1, shape};
    sonnx_sub_f32(&a, &b, &c);
    report_f32("f32_subnormal_result", f32_bits(C[0]) == f32_bits(expected),
               expected, C[0]);
  }

  /* Cancellation / negative operand. */
  {
    const size_t shape[1] = {3};
    float A[3] = {5.0f, -3.25f, 1.0f};
    float B[3] = {5.0f, -3.25f, 2.5f};
    float C[3] = {9.0f, 9.0f, 9.0f};
    float E[3] = {0.0f, 0.0f, -1.5f};
    sonnx_tensor a = {A, 1, shape}, b = {B, 1, shape}, c = {C, 1, shape};
    sonnx_sub_f32(&a, &b, &c);
    int ok = f32_bits(C[0]) == f32_bits(E[0]) &&
             f32_bits(C[1]) == f32_bits(E[1]) &&
             f32_bits(C[2]) == f32_bits(E[2]);
    report("f32_cancel_and_negative", ok);
  }

  /* Rank-0 tensor: one element, broadcast by the caller. */
  {
    const size_t shape[2] = {2, 2};
    float A[4] = {5.0f, 3.25f, 4.0f, 1.75f};
    float Bs[1] = {2.5f};
    float C[4] = {0};
    float E[4] = {2.5f, 0.75f, 1.5f, -0.75f};
    sonnx_tensor a = {A, 2, shape};
    sonnx_tensor b = {Bs, 0, NULL};
    sonnx_tensor c = {C, 2, shape};
    sonnx_sub_f32(&a, &b, &c);
    int ok = 1;
    for (int i = 0; i < 4; i++) {
      if (f32_bits(C[i]) != f32_bits(E[i])) ok = 0;
    }
    report("f32_rank0_example3", ok);
  }

  /* Zero-sized dimension: the kernel writes nothing. */
  {
    const size_t shape[2] = {0, 3};
    float A[1] = {1.0f};
    float B[1] = {2.0f};
    float C[1] = {42.0f};
    sonnx_tensor a = {A, 2, shape}, b = {B, 2, shape}, c = {C, 2, shape};
    sonnx_sub_f32(&a, &b, &c);
    report("f32_zero_sized_dim", C[0] == 42.0f);
  }

  /* ------------------------------------------------------------------ */
  /* f64                                                                */
  /* ------------------------------------------------------------------ */

  /* Example 3: 0.3 - 0.1 in double, expect the rounded double; 1.0-1.0=+0. */
  {
    const size_t shape[1] = {2};
    double A[2] = {0.3, 1.0};
    double B[2] = {0.1, 1.0};
    double C[2] = {0.0, 0.0};
    volatile double expect0 = 0.3 - 0.1; /* computed by the C compiler on doubles */
    sonnx_tensor a = {A, 1, shape}, b = {B, 1, shape}, c = {C, 1, shape};
    sonnx_sub_f64(&a, &b, &c);
    int ok = f64_bits(C[0]) == f64_bits(expect0) &&
             f64_bits(C[1]) == 0x0000000000000000ULL;
    report("f64_example3", ok);
  }

  /* f64 specials: signed zeros and infinities. */
  {
    const size_t shape[1] = {4};
    double A[4] = {-0.0, 0.0, INFINITY, -INFINITY};
    double B[4] = {0.0, -0.0, -INFINITY, INFINITY};
    double C[4] = {1.0, 1.0, 1.0, 1.0};
    sonnx_tensor a = {A, 1, shape}, b = {B, 1, shape}, c = {C, 1, shape};
    sonnx_sub_f64(&a, &b, &c);
    int ok = 1;
    if (!(C[0] == 0.0 && signbit(C[0]))) ok = 0;    /* -0 - +0 = -0 */
    if (!(C[1] == 0.0 && !signbit(C[1]))) ok = 0;   /* +0 - -0 = +0 */
    if (!(isinf(C[2]) && C[2] > 0)) ok = 0;         /* +inf - -inf = +inf */
    if (!(isinf(C[3]) && C[3] < 0)) ok = 0;         /* -inf - +inf = -inf */
    report("f64_specials", ok);
  }

  /* f64 NaN propagation and same-sign infinity invalid op. */
  {
    const size_t shape[1] = {3};
    double A[3] = {NAN, INFINITY, -INFINITY};
    double B[3] = {1.0, INFINITY, -INFINITY};
    double C[3] = {0.0, 0.0, 0.0};
    sonnx_tensor a = {A, 1, shape}, b = {B, 1, shape}, c = {C, 1, shape};
    sonnx_sub_f64(&a, &b, &c);
    report("f64_nan_and_invalid", isnan(C[0]) && isnan(C[1]) && isnan(C[2]));
  }

  /* f64 exact binary fractions, negatives, cancellation. */
  {
    const size_t shape[2] = {2, 2};
    double A[4] = {1.5, -0.25, 0.125, -8.0};
    double B[4] = {0.25, -0.25, 0.125, 8.0};
    double C[4] = {0};
    double E[4] = {1.25, 0.0, 0.0, -16.0};
    sonnx_tensor a = {A, 2, shape}, b = {B, 2, shape}, c = {C, 2, shape};
    sonnx_sub_f64(&a, &b, &c);
    int ok = 1;
    for (int i = 0; i < 4; i++) {
      if (f64_bits(C[i]) != f64_bits(E[i])) ok = 0;
    }
    report("f64_exact_fractions", ok);
  }

  /* f64 zero-sized dimension. */
  {
    const size_t shape[1] = {0};
    double A[1] = {1.0};
    double B[1] = {2.0};
    double C[1] = {7.0};
    sonnx_tensor a = {A, 1, shape}, b = {B, 1, shape}, c = {C, 1, shape};
    sonnx_sub_f64(&a, &b, &c);
    report("f64_zero_sized_dim", C[0] == 7.0);
  }

  /* ------------------------------------------------------------------ */
  /* Signed integers                                                    */
  /* ------------------------------------------------------------------ */

  /* int8 Example 1: {6,5,-35} - {3,3,3} = {3,2,-38} */
  {
    const size_t shape[1] = {3};
    int8_t A[3] = {6, 5, -35};
    int8_t B[3] = {3, 3, 3};
    int8_t C[3] = {0, 0, 0};
    int8_t E[3] = {3, 2, -38};
    sonnx_tensor a = {A, 1, shape}, b = {B, 1, shape}, c = {C, 1, shape};
    sonnx_sub_i8(&a, &b, &c);
    int ok = C[0] == E[0] && C[1] == E[1] && C[2] == E[2];
    report("i8_example1", ok);
  }

  /* int8 Example 2: {127,-128,0} - {-1,1,1} = {-128,127,-1} (wraps) */
  {
    const size_t shape[1] = {3};
    int8_t A[3] = {INT8_MAX, INT8_MIN, 0};
    int8_t B[3] = {-1, 1, 1};
    int8_t C[3] = {0, 0, 0};
    int8_t E[3] = {(int8_t)-128, (int8_t)127, (int8_t)-1};
    sonnx_tensor a = {A, 1, shape}, b = {B, 1, shape}, c = {C, 1, shape};
    sonnx_sub_i8(&a, &b, &c);
    int ok = C[0] == E[0] && C[1] == E[1] && C[2] == E[2];
    report("i8_example2_wrap", ok);
  }

  /* int8 limits: MIN - 1 wraps to MAX; MAX - MIN wraps to -1. */
  {
    const size_t shape[1] = {3};
    int8_t A[3] = {INT8_MIN, INT8_MAX, INT8_MIN};
    int8_t B[3] = {1, INT8_MIN, INT8_MAX};
    int8_t C[3] = {0, 0, 0};
    int8_t E[3] = {(int8_t)127, (int8_t)-1, (int8_t)1};
    sonnx_tensor a = {A, 1, shape}, b = {B, 1, shape}, c = {C, 1, shape};
    sonnx_sub_i8(&a, &b, &c);
    int ok = C[0] == E[0] && C[1] == E[1] && C[2] == E[2];
    report("i8_boundaries_wrap", ok);
  }

  /* int16 boundaries. */
  {
    const size_t shape[2] = {2, 2};
    int16_t A[4] = {INT16_MIN, INT16_MAX, 0, -1};
    int16_t B[4] = {1, -1, 1, INT16_MIN};
    int16_t C[4] = {0, 0, 0, 0};
    int16_t E[4] = {(int16_t)32767, (int16_t)-32768, (int16_t)-1,
                    (int16_t)32767};
    sonnx_tensor a = {A, 2, shape}, b = {B, 2, shape}, c = {C, 2, shape};
    sonnx_sub_i16(&a, &b, &c);
    int ok = C[0] == E[0] && C[1] == E[1] && C[2] == E[2] && C[3] == E[3];
    report("i16_boundaries_wrap", ok);
  }

  /* int32 boundaries. */
  {
    const size_t shape[1] = {3};
    int32_t A[3] = {INT32_MIN, INT32_MAX, 100};
    int32_t B[3] = {1, -1, 250};
    int32_t C[3] = {0, 0, 0};
    int32_t E[3] = {(int32_t)2147483647, (int32_t)-2147483648, (int32_t)-150};
    sonnx_tensor a = {A, 1, shape}, b = {B, 1, shape}, c = {C, 1, shape};
    sonnx_sub_i32(&a, &b, &c);
    int ok = C[0] == E[0] && C[1] == E[1] && C[2] == E[2];
    report("i32_boundaries_wrap", ok);
  }

  /* int64 boundaries. */
  {
    const size_t shape[1] = {2};
    int64_t A[2] = {INT64_MIN, INT64_MAX};
    int64_t B[2] = {1, -1};
    int64_t C[2] = {0, 0};
    int64_t E[2] = {(int64_t)INT64_MAX, (int64_t)INT64_MIN};
    sonnx_tensor a = {A, 1, shape}, b = {B, 1, shape}, c = {C, 1, shape};
    sonnx_sub_i64(&a, &b, &c);
    int ok = C[0] == E[0] && C[1] == E[1];
    report("i64_boundaries_wrap", ok);
  }

  /* int32 zero-sized dimension writes nothing. */
  {
    const size_t shape[2] = {2, 0};
    int32_t A[1] = {5};
    int32_t B[1] = {7};
    int32_t C[1] = {99};
    sonnx_tensor a = {A, 2, shape}, b = {B, 2, shape}, c = {C, 2, shape};
    sonnx_sub_i32(&a, &b, &c);
    report("i32_zero_sized_dim", C[0] == 99);
  }

  /* ------------------------------------------------------------------ */
  /* Unsigned integers                                                  */
  /* ------------------------------------------------------------------ */

  /* uint8 Example 1: {0,255,4} - {1,1,7} = {255,254,253} */
  {
    const size_t shape[1] = {3};
    uint8_t A[3] = {0, 255, 4};
    uint8_t B[3] = {1, 1, 7};
    uint8_t C[3] = {0, 0, 0};
    uint8_t E[3] = {255, 254, 253};
    sonnx_tensor a = {A, 1, shape}, b = {B, 1, shape}, c = {C, 1, shape};
    sonnx_sub_u8(&a, &b, &c);
    int ok = C[0] == E[0] && C[1] == E[1] && C[2] == E[2];
    report("u8_example1_wrap", ok);
  }

  /* uint8 limits: 0-1 wraps to UINT8_MAX, 0-UINT8_MAX = 1, equal => 0. */
  {
    const size_t shape[2] = {2, 2};
    uint8_t A[4] = {0, 0, 200, 13};
    uint8_t B[4] = {1, UINT8_MAX, 200, 13};
    uint8_t C[4] = {0, 0, 0, 0};
    uint8_t E[4] = {UINT8_MAX, 1, 0, 0};
    sonnx_tensor a = {A, 2, shape}, b = {B, 2, shape}, c = {C, 2, shape};
    sonnx_sub_u8(&a, &b, &c);
    int ok = C[0] == E[0] && C[1] == E[1] && C[2] == E[2] && C[3] == E[3];
    report("u8_boundaries_wrap", ok);
  }

  /* uint16 boundaries. */
  {
    const size_t shape[1] = {3};
    uint16_t A[3] = {0, 0, UINT16_MAX};
    uint16_t B[3] = {1, UINT16_MAX, 1};
    uint16_t C[3] = {0, 0, 0};
    uint16_t E[3] = {UINT16_MAX, 1, UINT16_MAX - 1};
    sonnx_tensor a = {A, 1, shape}, b = {B, 1, shape}, c = {C, 1, shape};
    sonnx_sub_u16(&a, &b, &c);
    int ok = C[0] == E[0] && C[1] == E[1] && C[2] == E[2];
    report("u16_boundaries_wrap", ok);
  }

  /* uint32 boundaries. */
  {
    const size_t shape[1] = {2};
    uint32_t A[2] = {0, 7};
    uint32_t B[2] = {1, 10};
    uint32_t C[2] = {0, 0};
    uint32_t E[2] = {UINT32_MAX, UINT32_MAX - 2};
    sonnx_tensor a = {A, 1, shape}, b = {B, 1, shape}, c = {C, 1, shape};
    sonnx_sub_u32(&a, &b, &c);
    int ok = C[0] == E[0] && C[1] == E[1];
    report("u32_boundaries_wrap", ok);
  }

  /* uint64 boundaries. */
  {
    const size_t shape[1] = {2};
    uint64_t A[2] = {0, UINT64_MAX};
    uint64_t B[2] = {1, UINT64_MAX};
    uint64_t C[2] = {0, 0};
    uint64_t E[2] = {UINT64_MAX, 0};
    sonnx_tensor a = {A, 1, shape}, b = {B, 1, shape}, c = {C, 1, shape};
    sonnx_sub_u64(&a, &b, &c);
    int ok = C[0] == E[0] && C[1] == E[1];
    report("u64_boundaries_wrap", ok);
  }

  /* uint16 zero-sized dimension. */
  {
    const size_t shape[1] = {0};
    uint16_t A[1] = {10};
    uint16_t B[1] = {20};
    uint16_t C[1] = {77};
    sonnx_tensor a = {A, 1, shape}, b = {B, 1, shape}, c = {C, 1, shape};
    sonnx_sub_u16(&a, &b, &c);
    report("u16_zero_sized_dim", C[0] == 77);
  }

  /* uint32 multi-element shape, several elements so loop bounds are checked. */
  {
    const size_t shape[2] = {2, 3};
    uint32_t A[6] = {10, 20, 30, 40, 50, 60};
    uint32_t B[6] = {1, 2, 3, 4, 5, 6};
    uint32_t C[6] = {0, 0, 0, 0, 0, 0};
    uint32_t E[6] = {9, 18, 27, 36, 45, 54};
    sonnx_tensor a = {A, 2, shape}, b = {B, 2, shape}, c = {C, 2, shape};
    sonnx_sub_u32(&a, &b, &c);
    int ok = 1;
    for (int i = 0; i < 6; i++) {
      if (C[i] != E[i]) ok = 0;
    }
    report("u32_multi_element", ok);
  }

  (void)report_int; /* silence unused helper if unused below */
  (void)report_f64;

  return failures == 0 ? 0 : 1;
}
