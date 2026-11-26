# Agent interface

Terasweep exposes a versioned fit packet through the CLI and a local stdio MCP
server. Both call the same Rust single-model runner and return the same receipt.

| Operation | Command |
| --- | --- |
| Discover the request and receipt schemas | `python3 scripts/model_run.py schema` |
| Fit a packet from a file | `python3 scripts/model_run.py fit docs/examples/ridge.json` |
| Fit a packet from stdin | `python3 scripts/model_run.py fit -` |
| Start MCP over stdio | `python3 scripts/model_run.py mcp` |

The MCP server exposes `terasweep_fit` with the same `inputSchema` and
`outputSchema` shown by `schema`. For a client that accepts MCP command
configuration, use the expanded path to your checkout and its working directory.

```json
{
  "mcpServers": {
    "terasweep": {
      "command": "python3",
      "args": ["/path/to/Terasweep/scripts/model_run.py", "mcp"]
    }
  }
}
```

Send exactly one `model` and explicit `train` and `test` matrix paths. Paths
resolve from the server's working directory. Use absolute paths when the agent
runs outside the checkout. LightGBM and TiDE require their local backend
environment to be present when the server starts. [Backend setup](backends.md)

## Matrix format

Both matrix files begin with `RIDGE001`, then little-endian unsigned 64-bit
row and column counts, row-major `f64` features, and one `f64` target per row.
The fit uses training targets and test features. Use finite placeholders in
the test target slots when only features are available. The receipt identifies the
actual matrix bytes by hash and lists predictions in test-row order. Sweeps
use labeled validation matrices in this same format. [Sweep interface](sweeps.md)

A fit result contains `version`, `model`, train and test fingerprints,
`predictions`, and `fit_predict_seconds`. Each request fits a new model and
makes one prediction pass. The timer covers model work after input validation
and backend loading. [Timing method](research/model-reference-parity.md#single-fit-performance)

CLI errors exit nonzero with a JSON error on stderr. MCP reports tool failures
with `isError`. For training settings and backend requirements, see the
[model cards](../README.md#model-cards).
