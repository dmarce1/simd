# simd

This is a library that provides fast mathematical functions, 
such as `sqrt, sin, cos, erf` etc., for the SIMD (Single Instruction,
Multiple Data) processor capabilities and SIMD data types.
SIMD instructions were introduced for `x86_64` architecture (AVX,
AVX2, AVX512, some AMD versions).

These library introduces the capability to use the math functions
similar to what is provided in `<math.h>` for the `double` and `float`
types in C.


## Installation

Prerequisites:
 - gmp, gmp-devel library;
 - mpfr, mpfr-devel library.

To install, use the standard cmake procedure:

```bash
   mkdir build; cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make
```

This will create an executable `simd_test`. If you run it, it should
display the results like this:
```
$ ./simd_test
5.000000e+00 nan
Testing SIMD Functions

Single Precision
name   speed        avg err      max err
  asin 6.382254e+00 4.199801e-01 3.000000e+00
  acos 8.652846e+00 2.972159e-01 2.000000e+00
  atan 3.004277e+00 4.158778e-01 6.000000e+00
 acosh 1.524083e+00 2.971838e-01 5.000000e+00

...
```

Here, the first column is the name of the transcendental function tested,
the second one is the speedup compared to the standard scalar types, and
the third and fourth columns are the average and max errors with respect to
the scalar type, in the units of ULP (=Unit in the Last Place for the FP
representation).
