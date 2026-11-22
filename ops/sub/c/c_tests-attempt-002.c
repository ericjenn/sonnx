#include "sonnx_kernels.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

/* ---- comparators: exact, bit-pattern based for floating point ---- */

static void check_f32(const char *name, const float *exp, const float *got, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    uint32_t eb = 0, gb = 0;
    memcpy(&eb, &exp[i], sizeof eb);
    memcpy(&gb, &got[i], sizeof gb);
    if (isnan(exp[i])) {
      if (!isnan(got[i])) {
        printf("FAIL %s: expected NaN at index %zu, got 0x%08x\n",
               name, i, (unsigned)gb);
        failures++;
        return;
      }
    } else if (isnan(got[i]) || eb != gb) {
      printf("FAIL %s: expected bits 0x%08x at index %zu, got bits 0x%08x\n",
             name, (unsigned)eb, i, (unsigned)gb);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_f64(const char *name, const double *exp, const double *got, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    uint64_t eb = 0, gb = 0;
    memcpy(&eb, &exp[i], sizeof eb);
    memcpy(&gb, &got[i], sizeof gb);
    if (isnan(exp[i])) {
      if (!isnan(got[i])) {
        printf("FAIL %s: expected NaN at index %zu, got 0x%016llx\n",
               name, i, (unsigned long long)gb);
        failures++;
        return;
      }
    } else if (isnan(got[i]) || eb != gb) {
      printf("FAIL %s: expected bits 0x%016llx at index %zu, got bits 0x%016llx\n",
             name, (unsigned long long)eb, i, (unsigned long long)gb);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_i8(const char *name, const int8_t *exp, const int8_t *got, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    if (exp[i] != got[i]) {
      printf("FAIL %s: expected %d at index %zu, got %d\n",
             name, (int)exp[i], i, (int)got[i]);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_i16(const char *name, const int16_t *exp, const int16_t *got, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    if (exp[i] != got[i]) {
      printf("FAIL %s: expected %d at index %zu, got %d\n",
             name, (int)exp[i], i, (int)got[i]);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_i32(const char *name, const int32_t *exp, const int32_t *got, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    if (exp[i] != got[i]) {
      printf("FAIL %s: expected %lld at index %zu, got %lld\n",
             name, (long long)exp[i], i, (long long)got[i]);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_i64(const char *name, const int64_t *exp, const int64_t *got, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    if (exp[i] != got[i]) {
      printf("FAIL %s: expected %lld at index %zu, got %lld\n",
             name, (long long)exp[i], i, (long long)got[i]);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_u8(const char *name, const uint8_t *exp, const uint8_t *got, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    if (exp[i] != got[i]) {
      printf("FAIL %s: expected %u at index %zu, got %u\n",
             name, (unsigned)exp[i], i, (unsigned)got[i]);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_u16(const char *name, const uint16_t *exp, const uint16_t *got, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    if (exp[i] != got[i]) {
      printf("FAIL %s: expected %u at index %zu, got %u\n",
             name, (unsigned)exp[i], i, (unsigned)got[i]);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_u32(const char *name, const uint32_t *exp, const uint32_t *got, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    if (exp[i] != got[i]) {
      printf("FAIL %s: expected %llu at index %zu, got %llu\n",
             name, (unsigned long long)exp[i], i, (unsigned long long)got[i]);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

static void check_u64(const char *name, const uint64_t *exp, const uint64_t *got, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    if (exp[i] != got[i]) {
      printf("FAIL %s: expected %llu at index %zu, got %llu\n",
             name, (unsigned long long)exp[i], i, (unsigned long long)got[i]);
      failures++;
      return;
    }
  }
  printf("ok %s\n", name);
}

int main(void) {
  /* ------------------------------------------------------------------ */
  /* f32                                                                */
  /* ------------------------------------------------------------------ */

  /* Float example 1: every difference is exactly representable. */
  {
    const size_t sh[2] = {3, 2};
    float A[6] = {3.0f, 4.5f, 16.0f, 1.0f, 25.5f, 24.25f};
    float B[6] = {3.0f, 2.0f, 4.0f, 0.0f, 5.0f, 4.0f};
    float C[6] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
    float E[6] = {0.0f, 2.5f, 12.0f, 1.0f, 20.5f, 20.25f};
    sonnx_tensor a = { .data = A, .ndim = 2, .shape = sh };
    sonnx_tensor b = { .data = B, .ndim = 2, .shape = sh };
    sonnx_tensor c = { .data = C, .ndim = 2, .shape = sh };
    sonnx_sub_f32(&a, &b, &c);
    check_f32("f32 float example 1", E, C, 6);
  }

  /* Float example 2: infinities, signed zeros and NaN. */
  {
    const size_t sh[1] = {4};
    float A[4] = {1.0f, INFINITY, -0.0f, 0.0f};
    float B[4] = {1.0f, INFINITY, 0.0f, -0.0f};
    float C[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
    float E[4] = {0.0f, NAN, -0.0f, 0.0f};
    sonnx_tensor a = { .data = A, .ndim = 1, .shape = sh };
    sonnx_tensor b = { .data = B, .ndim = 1, .shape = sh };
    sonnx_tensor c = { .data = C, .ndim = 1, .shape = sh };
    sonnx_sub_f32(&a, &b, &c);
    check_f32("f32 float example 2 (inf, signed zeros, NaN)", E, C, 4);
  }

  /* Infinite operand combinations, the two not being infinities of the same sign. */
  {
    const size_t sh[1] = {6};
    float A[6] = {INFINITY, 5.0f, -INFINITY, INFINITY, -INFINITY, 5.0f};
    float B[6] = {5.0f, -INFINITY, 5.0f, -INFINITY, INFINITY, INFINITY};
    float C[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float E[6] = {INFINITY, INFINITY,
