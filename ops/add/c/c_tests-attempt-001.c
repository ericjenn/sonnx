/*
 * c_tests.c -- conformance tests for the generated SONNX Add kernels.
 *
 * Every expectation is computed here from the specification (ONNX Add 14,
 * sections "real", "float", "int", "uint") and never from the kernel.
 * Floating-point expectations are compared by bit pattern (so +/-0 are
 * distinguished) and NaN expectations with isnan(); integer expectations
 * are stated literally modulo 2^bits.
 */
#include "sonnx_kernels.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void cat(char *dst, size_t cap, const char *src) {
  size_t d = strlen(dst);
  size_t s = strlen(src);
  if (d + s + 1 <= cap) {
    memcpy(dst + d, src, s + 1);
  }
}

/* Exact comparison: NaN matches NaN, everything else (including +0 vs -0)
 * must match bit for bit. */
static int same_f32(float e, float g) {
  uint32_t eb;
  uint32_t gb;
  if (isnan(e)) {
    return isnan(g) != 0;
  }
  memcpy(&eb, &e, sizeof eb);
  memcpy(&gb, &g, sizeof gb);
  return eb == gb;
}

static int same_f64(double e, double g) {
  uint64_t eb;
  uint64_t gb;
  if (isnan(e)) {
    return isnan(g) != 0;
  }
  memcpy(&eb, &e, sizeof eb);
  memcpy(&gb, &g, sizeof gb);
  return eb == gb;
}

static void report_f32(const char *name, const float *exp, const float *got, size_t n) {
  char estr[1024] = "";
  char gstr[1024] = "";
  char tmp[64];
  int ok = 1;
  for (size_t i = 0; i < n; i++) {
    if (i > 0) {
      cat(estr, sizeof estr, ", ");
      cat(gstr, sizeof gstr, ", ");
    }
    if (isnan(exp[i])) {
      snprintf(tmp, sizeof tmp, "NaN");
    } else {
      snprintf(tmp, sizeof tmp, "%.9g", (double)exp[i]);
    }
    cat(estr, sizeof estr, tmp);
    if (isnan(got[i])) {
      snprintf(tmp, sizeof tmp, "NaN");
    } else {
      snprintf(tmp, sizeof tmp, "%.9g", (double)got[i]);
    }
    cat(gstr, sizeof gstr, tmp);
    if (!same_f32(exp[i], got[i])) {
      ok = 0;
    }
  }
  if (ok) {
    printf("ok %s\n", name);
  } else {
    printf("FAIL %s: expected [%s], got [%s]\n", name, estr, gstr);
    failures++;
  }
}

static void report_f64(const char *name, const double *exp, const double *got, size_t n) {
  char estr[1024] = "";
  char gstr[1024] = "";
  char tmp[64];
  int ok = 1;
  for (size_t i = 0; i < n; i++) {
    if (i > 0) {
      cat(estr, sizeof estr, ", ");
      cat(gstr, sizeof gstr, ", ");
    }
    if (isnan(exp[i])) {
      snprintf(tmp, sizeof tmp, "NaN");
    } else {
      snprintf(tmp, sizeof tmp, "%.17g", exp[i]);
    }
    cat(estr, sizeof estr, tmp);
    if (isnan(got[i])) {
      snprintf(tmp, sizeof tmp, "NaN");
    } else {
      snprintf(tmp, sizeof tmp, "%.17g", got[i]);
    }
    cat(gstr, sizeof gstr, tmp);
    if (!same_f64(exp[i], got[i])) {
      ok = 0;
    }
  }
  if (ok) {
    printf("ok %s\n", name);
  } else {
    printf("FAIL %s: expected [%s], got [%s]\n", name, estr, gstr);
    failures++;
  }
}

static void report_i(const char *name, const int64_t *exp, const int64_t *got, size_t n) {
  char estr[512] = "";
  char gstr[512] = "";
  char tmp[
