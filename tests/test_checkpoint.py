"""Checkpoints saved by the trainer must be loadable for resuming."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

from lib import checkpoint


class CheckpointTests(unittest.TestCase):
    def test_resumed_recovery_keeps_update_ema_when_training_starts(self):
        import configs
        import main
        from lib import correctors

        def make_optimizer():
            layer = torch.nn.Linear(2, 2)
            base = torch.optim.NAdam(
                layer.parameters(), lr=0.0, decoupled_weight_decay=True
            )
            return correctors.build(base, [layer], 2, backward=configs.Recover())

        source = make_optimizer()
        for delta in source.backward.realization.delta.values():
            delta.fill_(1.0)
        restored = make_optimizer()
        restored.load_state_dict(source.state_dict())

        runtime = Mock(stage=0, num_stages=2, num_warmup_minibatches=1)
        runtime.get_adjusted_learning_rate.return_value = 0.0
        cfg = configs.Config(
            model='ngpt_95m_32layer', method=configs.MethodConfig(name='recovery')
        )
        cfg.lr.schedule.warmup_init = 0.0
        # One resumed step with a zero update must decay the restored EMA,
        # rather than resetting it before the first forward.
        main.train(cfg, runtime, restored, Mock(), start_step=1, total_steps=2)
        for delta in restored.backward.realization.delta.values():
            torch.testing.assert_close(
                delta, torch.full_like(delta, restored.backward.realization.gamma)
            )

    def test_trainer_state_round_trip_without_legacy_epoch_fields(self):
        state = {
            "step": 20000,
            "arch": "gptn",
            "eval_loss": 4.9,
            "state_dict": {"weight": torch.tensor([1.0, 2.0])},
            "optimizer": {"state": {}, "param_groups": []},
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint.save_checkpoint(state, directory, stage=3)
            restored = checkpoint.load_checkpoint(
                str(Path(directory) / "checkpoint"), stage=3
            )
        self.assertEqual(restored["step"], state["step"])
        self.assertEqual(restored["arch"], state["arch"])
        self.assertEqual(restored["eval_loss"], state["eval_loss"])
        self.assertEqual(restored["optimizer"], state["optimizer"])
        torch.testing.assert_close(
            restored["state_dict"]["weight"], state["state_dict"]["weight"]
        )


if __name__ == "__main__":
    unittest.main()
