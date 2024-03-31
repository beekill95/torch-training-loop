from __future__ import annotations

import glob

DOIT_CONFIG = {
    "action_string_formatting": "new",
}

SRC_FILES = glob.glob("src/**/*.py")
TEST_FILES = glob.glob("tests/**/*.py")
EXAMPLE_FILES = glob.glob("examples/**/*.py")


def task_type_check():
    return {
        "file_dep": (
            *SRC_FILES,
            *TEST_FILES,
            *EXAMPLE_FILES,
        ),
        "actions": ["mypy --check-untyped-defs {dependencies}"],
    }


def task_test():
    return {
        "file_dep": (*SRC_FILES, *TEST_FILES),
        "actions": ["pytest --lf"],
        "verbosity": 2,
    }
