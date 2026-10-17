# INT8 KV scale range contract

Finite components are saturated individually to `[-8319008,8319008]` (`65504*127`) before either codes or the scale are computed. This prevents finite extreme inputs from producing infinite FP16 scales and `0*infinity` NaNs during FP32 dequantization. Clamping only the stored scale would define a different, inconsistent quantizer; this implementation clips the values used by both calculations.

In-range finite inputs retain the previous code bytes and scale bits. Each 32-value group uses the existing maximum floor `2^-100`, reciprocal implementation, FP32 multiplication by 127, nearest-even integer conversion, and nearest-even FP16 conversion of `fl(g*float(1/127))`. SASS performs the two component clamps before its existing maximum/reduction. The unchecked C twin requires finite inputs and FE_TONEAREST. The checked C entry point rejects null pointers, nonfinite inputs and unsupported rounding modes before modifying either output.

The stored scale is finite and nonnegative, but need not be positive. Very small groups can round to zero. The actual zero/subnormal boundary is governed by the rounded FP32 scale expression before half conversion; do not replace it with an exact-real division threshold. When the scale is zero, reconstruction is zero and absolute error is bounded by the clipped group's maximum. There is no uniform relative-error guarantee near zero.

Half conversion alone contributes at most `127*2^-25 = 0.0000037848949432373046875` absolute reconstruction error in the half-subnormal range. This is not the total quantization-error bound: integer quantization, reciprocal approximation and FP32 scale computation must be accounted for separately. INT8 times a finite FP16 scale is representable within FP32's finite range. Saturation deliberately loses magnitudes above the cap; it does not imply that dequantization into FP16 would remain finite.

## Validation

`make -C kern test-quant-range` compares the bounded quantizer with the original operation sequence applied to independently clipped inputs. Cases cover signed zeros, ordinary values, both original underflow/overflow witnesses, clipping neighbors, half zero/subnormal/normal/overflow neighborhoods, all nonfinite positions, unchanged rejection outputs and explicit scale classes.

`QSA_KV_EXTREMES=1 ./qsa_test kv 16 2` exercises the changed GPU V quantizer, current code/scale bits, gather and attention. The fixture rotates through 18 magnitudes, including guaranteed half-subnormal and normal scales. Independent finite-scale/reconstruction/output checks fail even if the scalar twin shared a numerical error. Repeat with `QSA_HOSTKV=1` and the normal finite fixture.

Synthetic extremes establish the range policy, not observed checkpoint impact. Per-layer group maxima and clipping/underflow counts on representative real long-context requests, plus masking/clamp overhead, remain separate measurement gates. This PR must not claim those results until an exact-build model run supplies them.
