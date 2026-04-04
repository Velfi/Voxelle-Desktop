# Changelog Fragments

Drop a markdown file here for each user-visible change.

## Format

```markdown
---
category: added
---

Short description of the change.
```

**Categories:** `added`, `changed`, `fixed`, `removed`

## Workflow

1. Create a `.md` file with a descriptive slug (e.g., `add-walk-mode.md`)
2. Run `npm run changelog:preview` to see what the next release notes will look like
3. At release time, `bump-version.sh` compiles fragments into `CHANGELOG.md` and deletes them
