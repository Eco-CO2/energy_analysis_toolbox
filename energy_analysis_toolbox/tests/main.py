# -*- coding: utf-8 -*-
"""Run tests and coverages.

This module implements a ``run`` function which can be used to run the library's
test suite from a python interpreter.

Using this function rather than launching using ``pytest`` enables you to be sure
that the tested code is the one from your ``PYTHONPATH`` i.e., the code which
you use when you type ``import energy_analysis_toolbox`` in the calling interpreter.
"""

import pytest
import os


def run():
    """Use pytest to run the test.

    Preferred way to test the library's installation used by the calling python process
    """
    TEST_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

    test_report_filename = os.getcwd() + "/pytest_report.xml"
    coverage_config_filename = (
        os.path.dirname(os.path.dirname(TEST_DIRECTORY)) + "/.coveragerc"
    )
    if not os.path.isfile(coverage_config_filename):
        coverage_config_filename = os.getcwd() + "/.coveragerc"
    if not os.path.isfile(coverage_config_filename):
        coverage_config_filename = ""
    code = pytest.main(
        [
            TEST_DIRECTORY,
            "--junitxml",
            test_report_filename,
            "--cov",
            "energy_analysis_toolbox",
            "--cov-config",
            coverage_config_filename,
            "--cov-report",
            "xml",
            "--cov-report",
            "term",
        ]
    )

    if code != pytest.ExitCode(0):
        raise ValueError("Failed test suite")


if __name__ == "__main__":
    run()
