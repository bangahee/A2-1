"""Validate repository-wide commit subjects and local branch names."""

import re
import subprocess

SUBJECT = re.compile(
    r"^(feat|fix|docs|test|refactor|chore|ci|perf|style|build|revert)"
    r"(\([a-z0-9/-]+\))?!?: .+"
)
BRANCH = re.compile(r"^(main|codex/[a-z0-9]+(?:-[a-z0-9]+)*)$")


def main():
    """Reject any commit or local branch that violates CONTRIBUTING.md."""
    subjects = subprocess.check_output(
        ["git", "log", "--all", "--format=%s"], text=True
    ).splitlines()
    branches = subprocess.check_output(
        ["git", "for-each-ref", "--format=%(refname:short)", "refs/heads"], text=True
    ).splitlines()
    errors = [s for s in subjects if not SUBJECT.fullmatch(s)]
    errors += [b for b in branches if not BRANCH.fullmatch(b)]
    if errors:
        raise SystemExit("Convention violations: " + repr(errors))
    print(f"PASS: {len(subjects)} commits and {len(branches)} branches.")


if __name__ == "__main__":
    main()
