#include "sonnx_kernels.h"

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <limits.h>

static int failures = 0;

static int f32_same(float got, float want) {
  if (isnan(want)) {
    return isnan(got) != 0;
  }
  if (want == 0.0f) {
    return got == 0.0f && ((signbit(got) != 0) == (signbit(want) != 0));
  }
  return got == want;
}

static int f64_same(double got, double want) {
  if (isnan(want)) {
    return isnan(got) != 0;
  }
  if (want == 0.0) {
    return got == 0.0 && ((signbit(got) != 0) == (signbit(want) != 0));
  }
  return got == want;
}

static void report_f32(const char *name, const float *got, const float *want, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    if (!f32_same(got[i], want[i])) {
      printf("FAIL %s: expected %a, got %a (index %zu)\n",
             name, (double)want[i], (double)got[i], i);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void report_f64(const char *name, const double *got, const double *want, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    if (!f64_same(got[i], want[i])) {
      printf("FAIL %s: expected %a, got %a (index %zu)\n",
             name, want[i], got[i], i);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void report_i8(const char *name, const int8_t *got, const int8_t *want, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    if (got[i] != want[i]) {
      printf("FAIL %s: expected %d, got %d (index %zu)\n",
             name, (int)want[i], (int)got[i], i);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void report_i16(const char *name, const int16_t *got, const int16_t *want, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    if (got[i] != want[i]) {
      printf("FAIL %s: expected %d, got %d (index %zu)\n",
             name, (int)want[i], (int)got[i], i);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void report_i32(const char *name, const int32_t *got, const int32_t *want, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    if (got[i] != want[i]) {
      printf("FAIL %s: expected %ld, got %ld (index %zu)\n",
             name, (long)want[i], (long)got[i], i);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void report_i64(const char *name, const int64_t *got, const int64_t *want, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    if (got[i] != want[i]) {
      printf("FAIL %s: expected %lld, got %lld (index %zu)\n",
             name, (long long)want[i], (long long)got[i], i);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void report_u8(const char *name, const uint8_t *got, const uint8_t *want, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    if (got[i] != want[i]) {
      printf("FAIL %s: expected %u, got %u (index %zu)\n",
             name, (unsigned)want[i], (unsigned)got[i], i);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void report_u16(const char *name, const uint16_t *got, const uint16_t *want, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    if (got[i] != want[i]) {
      printf("FAIL %s: expected %u, got %u (index %zu)\n",
             name, (unsigned)want[i], (unsigned)got[i], i);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void report_u32(const char *name, const uint32_t *got, const uint32_t *want, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    if (got[i] != want[i]) {
      printf("FAIL %s: expected %lu, got %lu (index %zu)\n",
             name, (unsigned long)want[i], (unsigned long)got[i], i);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void report_u64(const char *name, const uint64_t *got, const uint64_t *want, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    if (got[i] != want[i]) {
      printf("FAIL %s: expected %llu, got %llu (index %zu)\n",
             name, (unsigned long long)want[i], (unsigned long long)got[i], i);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

int main(void) {
  /* f32, float example 1: every sum is exactly representable. */
  {
    const size_t shape[2] = {3, 2};
    float a_data[6] = {3.0f, 4.5f, 16.0f, 1.0f, 25.5f, 24.25f};
    float b_data[6] = {3.0f, 2.
