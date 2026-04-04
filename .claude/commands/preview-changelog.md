Preview the changelog for the next release of Voxelle Desktop.

## Steps

1. List all fragment files in `.changes/` (excluding `README.md`). If there are none, tell the user there are no pending changelog entries.

2. Run `npm run changelog:preview` and show the output to the user.

3. If there are any issues with the entries (duplicates, unclear descriptions, missing categories), flag them.
