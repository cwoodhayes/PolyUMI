# PolyUMI Catalog

A localhost web app for browsing the tasks, scenes, sessions, and training datasets produced by PolyUMI. It scans the recordings directory into a local SQLite cache (`polyumi-catalog sync`), and provides r/w access to the metadata for each item. See
[docs/catalog-ui-plan.md](../docs/catalog-ui-plan.md) for the full design and phase
plan.

## Running it

From the repo root:

```bash
# one-off scan of ~/recordings into the SQLite cache (also runs automatically on `serve`)
uv run polyumi-catalog sync

# start the browser at http://127.0.0.1:8420
uv run polyumi-catalog serve
```

Both commands default to `~/recordings` and `~/recordings/.catalog/catalog.db`; pass
`--recordings <dir>` / `--db <path>` to point elsewhere, and `--port` to change the
serve port. Use the "Rescan" button in the UI (or `polyumi-catalog sync`) to refresh
the cache after new sessions land on disk.
