"""Regression tests for fused sigmoid-gate top-k and logsigmoid renorm.

Normalizing sigmoid(x) directly produces 0/0 when every active sigmoid
underflows. The implementation must normalize in the log domain and exclude
padding lanes before the max shift. See sgl-project/sglang#51102.
"""

import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.moe.inkling_gate_topk_renorm import (
    inkling_gate_topk_renorm_v2,
)
from sglang.kernels.ops.moe.sigmoid_gate_topk_renorm import sigmoid_gate_topk_renorm
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")
# backend-specific: ROCm exercises the Triton fallback and active-lane masking.
register_amd_ci(est_time=60, suite="base-a-jit-kernel-unit-test-amd")

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _reference(logits, k, n_shared, bias, route_scale, global_scale):
    """FP32 selection followed by an independent FP64 logsigmoid renorm."""
    n_routed = logits.shape[1] - n_shared
    routed = logits[:, :n_routed].float()
    ids = torch.topk(torch.sigmoid(routed) + bias, k, dim=-1).indices
    active = routed.gather(-1, ids).double()
    if n_shared:
        active = torch.cat([active, logits[:, n_routed:].double()], dim=-1)
    weights = F.logsigmoid(active).softmax(dim=-1)
    return ids, weights * (route_scale * global_scale.double())


def _unpack_packed_topk(packed):
    ids = (packed >> 16).to(torch.int32)
    weights = (packed & 0xFFFF).to(torch.int16).view(torch.bfloat16).float()
    return ids, weights


@unittest.skipUnless(torch.cuda.is_available(), "CUDA or ROCm is required")
class TestSigmoidGateTopKRenorm(CustomTestCase):
    def _check(self, logits, k, n_shared, dtype=torch.float32, return_packed=False):
        logits = logits.to(_DEVICE, dtype)
        n_routed = logits.shape[1] - n_shared
        bias = (torch.randn(n_routed, dtype=torch.float32) * 0.1).to(_DEVICE)
        global_scale = torch.tensor([0.7], device=_DEVICE)
        route_scale = 1.3

        with envs.SGLANG_OPT_USE_GATE_TOPK_JIT.override(False):
            routed_w, ids, shared_w, packed = sigmoid_gate_topk_renorm(
                logits,
                k,
                n_shared,
                route_scale,
                global_scale,
                bias,
                return_packed_topk=return_packed,
            )
        ref_ids, ref_w = _reference(
            logits, k, n_shared, bias, route_scale, global_scale
        )

        if return_packed:
            self.assertIsNone(routed_w)
            self.assertIsNone(ids)
            ids, routed_w = _unpack_packed_topk(packed)
        else:
            self.assertIsNone(packed)

        self.assertEqual(ids.cpu().tolist(), ref_ids.cpu().tolist())
        observed_w = torch.cat([routed_w, shared_w], dim=-1)
        approximate = dtype == torch.bfloat16 or return_packed
        atol, rtol = (2e-2, 2e-2) if approximate else (1e-6, 1e-5)
        torch.testing.assert_close(
            observed_w.float(), ref_w.float(), atol=atol, rtol=rtol
        )
        self.assertTrue(torch.isfinite(observed_w).all())
        torch.testing.assert_close(
            observed_w.float().sum(-1),
            torch.full_like(observed_w.float().sum(-1), route_scale * 0.7),
            atol=atol,
            rtol=rtol,
        )

    def test_matches_reference(self):
        cases = [
            (1, 258, 6, 2),
            (7, 258, 6, 2),
            (64, 258, 6, 2),
            (3, 64, 4, 0),
            (3, 64, 6, 0),
            (3, 33, 6, 1),
            (33, 40, 6, 2),
        ]
        for tokens, experts, k, n_shared in cases:
            for scale in (1.0, 30.0):
                for return_packed in (False, True):
                    with self.subTest(
                        tokens=tokens,
                        experts=experts,
                        k=k,
                        n_shared=n_shared,
                        scale=scale,
                        return_packed=return_packed,
                    ):
                        self._check(
                            torch.randn(tokens, experts) * scale,
                            k,
                            n_shared,
                            return_packed=return_packed,
                        )

    def test_matches_reference_bf16_logits(self):
        for return_packed in (False, True):
            with self.subTest(return_packed=return_packed):
                self._check(
                    torch.randn(5, 258) * 10.0,
                    6,
                    2,
                    dtype=torch.bfloat16,
                    return_packed=return_packed,
                )

    def test_no_nan_when_every_active_logit_underflows(self):
        for value in (-100.0, -300.0, -1e30):
            for n_shared in (2, 1, 0):
                for return_packed in (False, True):
                    with self.subTest(
                        value=value, n_shared=n_shared, return_packed=return_packed
                    ):
                        self._check(
                            torch.full((2, 258), value),
                            6,
                            n_shared,
                            return_packed=return_packed,
                        )

    def test_underflow_in_one_row_does_not_leak_into_another(self):
        logits = torch.randn(2, 258)
        logits[0] = -300.0
        for return_packed in (False, True):
            with self.subTest(return_packed=return_packed):
                self._check(logits, 6, 2, return_packed=return_packed)

    def test_large_positive_logits_remain_stable(self):
        for value in (100.0, 300.0, 1e30):
            with self.subTest(value=value):
                self._check(torch.full((2, 258), value), 6, 2)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None,
    "the fused gate JIT kernel is CUDA-only",
)
class TestInklingGateTopKRenormV2(CustomTestCase):
    def test_jit_renorm_underflow(self):
        for value in (-100.0, -300.0, -1e30):
            for return_packed in (False, True):
                with self.subTest(value=value, return_packed=return_packed):
                    # Match the production 32-byte-aligned padded row layout.
                    padded = torch.full((2, 264), value, device="cuda")
                    logits = padded[:, :258]
                    bias = torch.zeros(256, device="cuda")
                    global_scale = torch.ones(1, device="cuda")
                    routed_w, ids, shared_w, packed = inkling_gate_topk_renorm_v2(
                        logits,
                        bias,
                        global_scale,
                        1.0,
                        return_packed=return_packed,
                    )
                    if return_packed:
                        ids, routed_w = _unpack_packed_topk(packed)
                    weights = torch.cat([routed_w, shared_w], dim=-1)
                    self.assertTrue(torch.isfinite(weights).all(), msg=str(weights))
                    torch.testing.assert_close(
                        weights,
                        torch.full_like(weights, 1.0 / 8.0),
                        atol=1e-5,
                        rtol=1e-5,
                    )
                    for row in ids.cpu().tolist():
                        self.assertEqual(len(set(row)), 6)


if __name__ == "__main__":
    unittest.main()
