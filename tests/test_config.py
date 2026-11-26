"""Fractions are accepted through both command-line entry points."""

import contextlib
import io
import unittest

import train
from lib.config import parse_config


class FractionArgumentTests(unittest.TestCase):
    def test_launcher_and_trainer_accept_fractional_hyperparameters(self):
        args, extra = train.parse_args([
            '--lr', '3/10000', '--method.proj_rank_ratio', '96/384',
            '--optimizer.momentum=99/100', '--optimizer.clip_grad', 'None',
        ])
        cfg = parse_config(train.stage_commands(args, extra, 12345)[0][3:])
        self.assertEqual(cfg.lr.schedule.peak_value, 3e-4)
        self.assertEqual(cfg.method.proj_rank_ratio, 0.25)
        self.assertEqual(cfg.optimizer.momentum, 0.99)
        self.assertIsNone(cfg.optimizer.clip_grad)

    def test_direct_trainer_accepts_fractional_recovery_coefficient(self):
        cfg = parse_config([
            '--model', 'ngpt_95m_32layer', '--method.name', 'recovery',
            '--method.discrepancy_correction=1/50',
        ])
        self.assertEqual(cfg.method.discrepancy_correction, 0.02)

    def test_invalid_fractions_report_cli_errors(self):
        for value in ('1/0', 'one/half'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    train.parse_args(['--lr', value])
                self.assertEqual(error.exception.code, 2)
                with self.assertRaises(SystemExit) as error:
                    parse_config([
                        '--model', 'ngpt_95m_32layer', '--method.name', 'fresco',
                        '--method.proj_rank_ratio', value,
                    ])
                self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
