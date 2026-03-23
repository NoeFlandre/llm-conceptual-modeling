# Commands Module Notes

- Keep CLI handlers small and focused.
- Prefer adding a new command module over expanding an existing handler beyond a narrow responsibility.
- New live-run utilities should read from the run artifacts written by `post_revision_debug` rather than mutating experiment state.
- Favor simple, text-first operator tools when the user wants quick monitoring or resume control.
