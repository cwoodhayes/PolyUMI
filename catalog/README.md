# PolyUMI Catalog

A localhost web app for browsing the tasks, scenes, sessions, and training datasets produced by PolyUMI. It scans the recordings directory into a local SQLite cache (`polyumi-catalog sync`), and provides r/w access to the metadata for each item.

Note: Unlike the other code in this repo, this app is pretty much fully vibecoded with no manual review, as it is purely a convenient UI layer (which calls the actual postprocessing scripts, which I maintain more carefully). 
The point is, my standards for code quality etc are lower for this part of the repo, and anyone picking up this work
should likely consider continuing in the same vein (treating this code as cheap).

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
the cache after new sessions land on disk. The "Fetch from Pi" button beside it pulls any
not-yet-local scenes off the Pi (`pingest fetch`, without the GoPro SD-card step) and
re-syncs when it's done; `--pi-host` (or `POLYUMI_PI_HOST` in your shell) sets which Pi,
defaulting to the `polyumi-pi` ssh alias that `pingest` and `fr3_session.sh` also default to.
