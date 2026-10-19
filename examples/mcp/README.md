# Shape Studio as an MCP server

`shapestudio serve` speaks the Model Context Protocol over stdio. Any MCP client can turn
pictures into meshes on this machine, offline, with the same honesty rules as the window:
quality is a setting, never a consequence; estimates are measured; nothing is uploaded.

| Client | File | Where |
| --- | --- | --- |
| Claude Desktop | `claude-desktop.json` | `%APPDATA%\Claude\claude_desktop_config.json` (Windows), `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) |
| Claude Code | `claude-code.mcp.json` | `.mcp.json` in the project, or `claude mcp add shapestudio -- shapestudio serve` |
| Cursor | `cursor.mcp.json` | `.cursor/mcp.json` in the project, or `~/.cursor/mcp.json` |
| VS Code (Copilot agent mode) | `vscode.mcp.json` | `.vscode/mcp.json` |
| Windows app instead of pip | `windows-app.json` | the installed `ShapeStudio.exe cli serve` is the same server |

The command must be on PATH: `pipx install presidium-shapestudio` puts `shapestudio` there.

## A typical agent flow

1. `capabilities()` once: hardware, PyTorch runtime (CPU or CUDA), which tier is downloaded
   and fits, expected seconds per preset (labelled "not measured yet" until a run happened).
   If TripoSR is missing, `download_model("triposr", consent=true)` after telling the user
   the size (1.56 GiB) and licence (MIT); the same for `isnet-general-use` (170 MB).
2. `reconstruct(image)` with a PNG/JPEG/WebP path returns a job id and an honest estimate at
   once. Poll `job_status` every 10-15 s; never block inside one call.
3. `job_result(job_id)` gives the GLB path (OBJ with `format="obj"`), the turntable PNG and
   the vertex/face counts; `as_base64=true` returns the bytes when the client cannot read
   local files.
4. `plan(tier, preset)` is the dry run to explain a refusal ("the best tier needs an NVIDIA
   GPU with 16 GB") or a trade-off ("fast is a 128³ grid") before committing.

## Who runs the jobs

Only one process runs jobs at a time (a lock in the data folder). The first studio to start
takes the worker role: if the desktop window is open it runs them; if a client launches
`shapestudio serve` first, that server runs them and the window shows the queue. When the
window starts later it asks an idle worker to hand over. Pause (Settings or tray) holds every
submitter, including agents.

## Timing to expect

Nothing has been measured by this project yet on real hardware; `capabilities()` reports the
calibrated numbers for the machine it runs on once a reconstruction or the benchmark has run.
Quote those to the user, not the heuristics.
