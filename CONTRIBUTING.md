# Branch and commit rules

These rules follow the existing A1-1 commit style and apply throughout A2-1.

- Default/submission branch: `main`.
- Development branches: `<type>/<lowercase-kebab-case-description> (for example, feat/a2-1-implementation)`.
- Commit format: `type: short description`, optionally `type(scope): description`.
- Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`, `perf`, `style`, `build`, `revert`.
- Keep each commit focused. Never commit environments, credentials, caches or unrelated work.
- Validate code, execute all three notebooks, and regenerate evidence before publishing.
- Fast-forward main after successful verification; preserve the development branch.
- Never force-push or rewrite shared history.

Enable the included local commit/push checks after cloning:

```bash
git config core.hooksPath .githooks
git switch -c feat/your-change
python -m unittest discover -s tests -v
python -m scripts.run_all
python -m scripts.execute_notebooks
python -m scripts.check_conventions
git add <specific-files>
git commit -m "feat: describe the completed change"
```

The hooks check commit subjects and branch names; CI also checks the complete
commit history. They are local/CI safeguards, not GitHub server branch protection.
