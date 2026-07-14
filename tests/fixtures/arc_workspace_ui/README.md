# ARC Workspace UI Export Fixtures

These JSON files are verbatim copies of the prediction JSONs that
`benchmarks/arc/workspace_ui/js/workspace.js` exports when a user clicks
"Download Prediction" (see `updatePredictionPreview` at workspace.js:199).

Their purpose is to pin the UI → converter schema contract: if either side
changes the set of expected fields, the converter test in
`tests/test_arc_prep.py` will fail.

**Do not edit by hand for convenience** — if the UI export format changes,
regenerate the fixtures by running the UI against a real payload and
committing the fresh output.
