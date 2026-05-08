---
title: ai-langgraph-server Validation Guide
docType: guide
scope: repo
status: active
authoritative: false
owner: ai-langgraph-server
language: en
whenToUse:
  - when selecting local validation commands
  - when recording proof for documentation-governance changes
whenToUpdate:
  - when validation commands or local gate behavior changes
checkPaths:
  - docs/agents/repo-validation.md
  - .docpact/config.yaml
  - .githooks/pre-push
  - scripts/docpact-gate.sh
  - scripts/install-git-hooks.sh
lastReviewedAt: 2026-05-08
lastReviewedCommit: 171453290f1c1c319e5d9c47af6622d76e6aa9df
related:
  - ../../AGENTS.md
  - ../../.docpact/config.yaml
---

# ai-langgraph-server Validation Guide

## Local Docpact Push Gate

Install the versioned local hook once per checkout:

```bash
./scripts/install-git-hooks.sh
```

The `pre-push` hook runs `scripts/docpact-gate.sh`, which performs strict config validation and `docpact lint --mode enforce` before the push leaves the machine. The default comparison base is `origin/main`. Override it for unusual stacks with `DOCPACT_BASE_REF=<ref>` or `scripts/docpact-gate.sh --base <ref>`. The gate writes its detailed report to a temporary file so normal pushes do not create `.docpact/runs/` artifacts.
