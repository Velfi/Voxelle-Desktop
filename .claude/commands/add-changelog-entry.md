Add a changelog entry to Voxelle Desktop. The change to document is: $ARGUMENTS

## How it works

Changelog entries are markdown fragment files in `.changes/`. Each file has YAML frontmatter with a `category` and a body describing the change. At release time, `bump-version.sh` compiles them into `CHANGELOG.md`.

## Steps

1. Determine the appropriate category from $ARGUMENTS:
   - `added` — new feature or capability
   - `changed` — modification to existing behavior
   - `fixed` — bug fix
   - `removed` — removed feature or deprecated item

2. Create a slug from $ARGUMENTS (e.g., "fix camera jitter" → `fix-camera-jitter`). Use lowercase, hyphens, no special characters, max ~40 chars.

3. Write the fragment file at `.changes/<slug>.md`:
   ```markdown
   ---
   category: <category>
   ---

   <One or two sentence description of the change from the user's perspective.>
   ```

4. Run `npm run changelog:preview` to show the user what the next release notes will look like with this entry included.

## Guidelines

- Write the description from the **user's perspective**, not the developer's. Say "Scenes load 2x faster" not "Rewrote mesh upload to use staging buffers".
- Keep it to one or two sentences.
- Don't duplicate an existing entry — check `.changes/` first.
- One fragment per logical change. If $ARGUMENTS describes multiple changes, create multiple files.
