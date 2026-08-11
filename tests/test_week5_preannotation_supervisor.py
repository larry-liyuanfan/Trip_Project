from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Week5PreannotationSupervisorTests(unittest.TestCase):
    def test_supervisor_keeps_tunnel_alive_and_resumes_without_repeating_successes(self) -> None:
        script = (
            ROOT / "scripts" / "supervise_week5_preannotation.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("ServerAliveInterval=30", script)
        self.assertIn("ServerAliveCountMax=3", script)
        self.assertIn("ExitOnForwardFailure=yes", script)
        self.assertIn('"--resume"', script)
        self.assertIn("another Week 5 preannotation supervisor", script)
        self.assertIn("partial_after_cleanup", script)

    def test_main_pass_does_not_retry_terminal_bad_cases(self) -> None:
        script = (
            ROOT / "scripts" / "supervise_week5_preannotation.ps1"
        ).read_text(encoding="utf-8")
        retry_flag = script.index('$arguments += "--retry-failures"')
        cleanup_guard = script.rindex("if ($RetryFailures)", 0, retry_flag)
        self.assertLess(cleanup_guard, retry_flag)


if __name__ == "__main__":
    unittest.main()
