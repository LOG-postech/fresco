"""The optimizer wrapper supports the stock base optimizer and its projections."""

import unittest

import torch

from lib.correctors import AsyncPipelineOptimizer
from lib.correctors.fresco import Projection


class CorrectorTests(unittest.TestCase):
    def test_stock_nadam_supports_wrapper_mode_calls(self):
        layer = torch.nn.Linear(4, 3)
        base = torch.optim.NAdam(layer.parameters(), decoupled_weight_decay=True)
        optimizer = AsyncPipelineOptimizer(base, [layer], num_versions=1)
        optimizer.train()
        optimizer.eval()

    def test_projection_preserves_protected_features_including_bias(self):
        torch.manual_seed(0)
        layer = torch.nn.Linear(4, 3)
        q = torch.linalg.qr(torch.randn(5, 2))[0]
        before = torch.cat([layer.weight.detach(), layer.bias.detach()[:, None]], 1)
        proposal = torch.randn_like(before) * 0.01
        projection = Projection(layer.weight, layer.bias, k2=2)
        with torch.no_grad():
            projection.before(q)
            layer.weight.add_(proposal[:, :4])
            layer.bias.add_(proposal[:, 4])
            projection.after(q)
        after = torch.cat([layer.weight.detach(), layer.bias.detach()[:, None]], 1)
        torch.testing.assert_close(after @ q, before @ q)
        torch.testing.assert_close(
            after - before, proposal @ (torch.eye(5) - q @ q.T)
        )


if __name__ == '__main__':
    unittest.main()
