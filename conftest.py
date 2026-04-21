"""Shared pytest configuration and fixtures for SABLE tests."""


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Regenerate golden files instead of comparing against them.",
    )


def pytest_report_teststatus(report, config):
    """Pad outcome words so PASSED/FAILED/SKIPPED/ERROR stay column-aligned."""
    if report.when == "call" or (report.when == "setup" and report.skipped):
        mapping = {
            "passed":  ("passed",  ".", "PASSED "),
            "failed":  ("failed",  "F", "FAILED "),
            "skipped": ("skipped", "s", "SKIPPED"),
            "error":   ("error",   "E", "ERROR  "),
        }
        return mapping.get(report.outcome)
