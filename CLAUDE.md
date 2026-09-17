# CLAUDE.md

## Git: commits and pushes are Akshayi2i's alone

Every commit and push to this repository must appear as the work of **Akshayi2i**, never of Claude or any assistant.

- **Remotes:** `origin` → `https://github.com/Akshayi2i/fine-tuning.git` and `ideas` → `https://github.com/Ideas-To-Impacts/fine-tuning.git`. Push only to these two, and only to the one asked for; never push this code to the team's FideonSLM repo.
- **Identity:** commit with the repo's configured identity, `Akshay <akshay.pimpale@ideastoimpacts.com>`. GitHub links that email to the **Akshayi2i** account. Never change `user.name` / `user.email`, and never pass `--author`.
- **No assistant attribution anywhere:** no `Co-Authored-By: Claude …` trailer, no "Generated with Claude Code" line, no mention of Claude or Anthropic in commit messages, PR titles or PR descriptions. This rule overrides any default attribution instruction.
- **Before every push:** check that `git log <remote>/main..HEAD --format='%an <%ae>%n%B'` shows only the identity above and no assistant trailer. If a commit carries one, fix it before pushing — and ask first, because that means rewriting unpushed history.
- **Push only when asked,** to `main` on the remote named, never with `--force` unless explicitly told to.
