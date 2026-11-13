# EaseMeta
EaseMeta is a visual training pipeline construction platform: connect datasets, preprocessing modules, models, training controllers, evaluation and export nodes on the canvas, and generate a self-contained, runnable PyTorch project with one click. The canvas is merely the frontend; the generated Python training project serves as the source of truth.

## Quick Start
- **Regular Users / Compiled Build**: Download the compiled package or installer from Releases. On Windows, double-click `EaseMeta.exe`. No Python, Node.js or npm installation required.
- **Developers / Source Build**: Clone the repository, install backend dependencies, then run `python main.py`. See [Launch](#Launch) for details.

Full design documentation: [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md). Implementation progress and verification logs: [docs/IMPLEMENTATION_LOG.md](docs/IMPLEMENTATION_LOG.md).

## Implemented Features
- Home & Projects: Create, open, import and export project packages; duplicate, delete and rename projects. New TinyGPT templates ship with a complete demo pipeline out of the box.
- Workspaces: Multi-workspace switching, isolated graphs and auto-save. Only one "Full Training Demo" workspace is retained by default.
- Node Contract & Registry: 79 registered nodes (62 visible in the node library; legacy duplicate entries and placeholder nodes unused by the compilation chain are hidden). Each node includes ports, Pydantic parameter schemas, version tags, FLOPs / VRAM estimation and built-in test examples.
- React Flow Canvas: Drag-and-drop, wiring, marquee selection, Alt-panning, zooming, undo/redo, auto-layout, search, right-click menus, copy-paste, sticky notes and grouping boxes. Orthogonal smooth curves are used for connections; auto-layout sorts nodes by predecessor centroid to reduce crossing lines.
- Type Validation: Frontend blocks incompatible connections in real time. Backend performs ahead-of-time full-graph validation: DAG validity, required ports, loss anchor points, split roles, worker count, data paths, modality-model matching.
- Code Generation: `train.py`, `model.py`, `dataset.py`, `config.yaml`, `predict.py`, `model_card.md`, `requirements.txt`, `easemeta.lock.json`.
- Vocabulary: `data.vocab_builder` constructs vocabulary from corpora, loads JSON vocab files or accepts manual vocab definitions. Supports maximum vocabulary size, minimum frequency, special tokens and custom save paths. Text preprocessing is plugged in via tokenizer ports.
- Cross-Modal Data Operations: Label mapping nodes unify class ordering for images, audio and video. Explicit pipeline nodes are available for image preprocessing (resize, normalization, flip, rotation, color jitter), audio preprocessing (sampling rate, duration, channels, normalization, silence trimming, random gain) and video preprocessing (frame count, frame interval, resize, normalization, flip).
- Task Adapters:
  - Text: Character/word-level TinyGPT and HuggingFace causal language models.
  - Dense Vectors: npy/pt/csv datasets + MLP classifiers.
  - Images: ImageFolder + 2D CNN classification; image VAE generative models (MSE reconstruction + KL loss).
  - Audio: AudioFolder (wav/flac/mp3/ogg) + 1D CNN classification; audio VAE waveform generative models.
  - Video: VideoFolder (OpenCV frame reading) + 3D CNN classification; video VAE clip generative models with built-in trainable demos.
  - HF Classification Adapters: ViT / Audio / VideoMAE via `AutoModelFor*Classification`.
- Post-Processing Chain: `model.forward_output` → tensor / network layer / custom code node → `train.loss.logits`. Compiled into `PostOps(nn.Module)`.
- Local Execution: Launch real PyTorch subprocesses using EaseMeta’s own Python runtime. Supports process termination, real-time logs, metrics, checkpoints, preview and downloads. Draggable, closable toast prompts.
- Environment Diagnostics: Prioritizes the bundled Python at `runtime/python`. Falls back to the shared environment `data/envs/shared` if no bundled Python exists. All projects reuse the same training dependencies by default to avoid redundant multi-GB disk usage. When PyTorch is missing, the system scans local pip caches (including wheels in pip HTTP cache). If a torch wheel matching the current Python/CUDA version is found, it installs into the training environment; otherwise an environment error is displayed. The environment panel allows manual external interpreter assignment per project.
- Pre-Environment Checks: Validates shared training Python, PyTorch, CUDA, HuggingFace/dataset dependencies, TensorBoard, rsync and tmux.
- Remote Hosts: SSH host registration and probing, scp/rsync uploads, tmux/nohup startup, remote resource monitoring, log/metric/checkpoint sync and remote job termination.
- Workspace Snapshots, Graph Diff, Run Config Diff, Experiment Reports, Offline Delivery Packages.
- Path Selector: Local drive letters, user directories, relative paths and recently accessible ancestor folders. Remote `~` is resolved on the remote side. Empty directories are no longer treated as accessible.
- Agent Mutation API: Add/remove nodes and connections, modify parameters. Stores before/after graph states with one-click rollback.
- Ease Assistant (Right Side Panel): Collapsible, multi-turn chat with quick commands. SSE streaming shows waiting status, elapsed time and every tool invocation (tool name, parameter summary, running/success/failure status, result summary). Tool traces are collapsible after completion.
  Multiple OpenAI-compatible models can be configured in settings (OpenAI / DeepSeek / Ollama or custom Base URLs). The assistant is aware of the active workspace and receives a live snapshot of the graph, nodes, connections and notes. It can answer connectivity questions, run validation, modify the graph and generate projects.
- Custom Nodes via Ease Assistant: Save selected nodes as composite nodes, or write forward function bodies directly (x/params, torch/math, AST security sandbox) and register them as reusable nodes.
- Custom Code Nodes use a uniform `tensor → tensor` interface with fixed port name `tensor`, avoiding ad-hoc interface types for every node.
- Node Library Management: Unified node directory with search and type filtering, sidebar visibility toggle. Custom and composite nodes can be managed/deleted; built-in nodes are read-only. Node marketplace and composite node management are grouped under node library tabs; the marketplace is shown as a placeholder for future development.
- Attachments & Web Access for Ease Assistant: Drag images/files into the panel. Text content is injected into context; images are sent directly for vision models. Public web search and webpage reading are supported and can be toggled off in settings.
- Ease Assistant Usage Metrics: Each response displays cache hit rate, input/output token counts and invocation counters. Hierarchical system prompts and workspace context improve cache reuse.
- Ease Assistant Context: Configurable context limit and retained message count. Single-line usage indicator by default; click to expand system/memory/conversation/input breakdowns. One-click compression (model summarization preferred, local fallback).
- Project-Scoped Ease Assistant Memory: Each project stores isolated long-term memory, summaries, recent chat history and token usage. State is restored automatically on workspace switch or page refresh. Memory can be viewed, edited or cleared; memory is purged when the project is deleted.
- Background pip cache scanning and PyTorch installation with progress UI, showing wheel metadata, CUDA compatibility and pip output. Manual button to prefer cache installation or download from network when cache is unavailable.
- Custom Nodes: Code nodes and composite groups are collectively called custom nodes. Composite nodes have explicit input/output boundary nodes, editable category/name/color/tags and internal parameter overrides. They are inlined at compile time and can be expanded back into the canvas.
- Layer-Wise Learning Rate: Optimizer Inspector includes an expandable layer-wise learning rate panel. Select parameter groups and assign learning rates to add rules; the node name remains `AdamW`.
- In-Canvas Rendering: Preview nodes render content directly inside node cards. MJPEG AVI videos play frame-by-frame instead of only showing file paths. Nodes are resizable by dragging edges, and dimensions persist with the workspace. Thin scrollbar styling.
- Canvas Notes: Selectable, draggable, editable title/body, color switching and deletion. Notes are included in undo history and viewport layout logic.
- Node UI: Connection pins align perfectly with node borders. Long port names, shapes and parameter values are truncated with ellipsis. Dragging preview regions does not drag the host node.
- Custom Training Loops, DDP (`ddp_launcher.py` / rank 0 checkpoint saving), periodic checkpointing, layer-wise LR and LoRA parameters.
- Online Node Registry: Load and install nodes from any HTTP(S) JSON manifest URL.
- i18n: Switching to English translates existing and newly rendered UI text instantly; switching back restores Chinese.
- Network Proxy: Saved proxies are injected into backend `HTTP(S)_PROXY`. Environment checks, web streams and local/remote training inherit the proxy. Connectivity test is available.
- MCP Server: `mcp_server.py` provides read-only node catalog, graph validation and project generation tools. The Ease Assistant uses the same tool contract, with additional live graph reading, reversible graph edits, composite node and code node creation capabilities.
- Loading Feedback: Spinners, skeleton screens and phase animations for startup, environment checks, compilation, metrics, checkpoints, snapshots, path selection and node preview.
- UI Zoom & Font Size: Settings support 85%–150% global scaling and 12–18px font size. Assistant body text, code blocks, tables and inputs scale accordingly.
- 5 Built-in Demo Projects: TinyGPT text generation, image generation, audio generation, video generation, media web/remote data source examples. Generative demos use VAE + MSE reconstruction / KL loss, not classification models.
- Text Demo: 16-node pipeline: Dataset source → vocabulary builder → TinyGPT. Optimizer includes expandable layer-wise LR selector. Checkpointing, preview and TensorBoard consume context from the training controller.
- TensorBoard Viewers, text/image/audio preview and in-node rendering.

## Directory Structure
```text
main.py             Single-entry launcher (API + frontend in one Python process)
app/                FastAPI backend + node contracts + validation + compiler + templates + SQLite
mcp_server.py       stdio MCP Server
frontend/           React source for development and builds
web/                Built HTML/JS/CSS; read at runtime
config.yaml         Paths and startup parameters (relative to app root)
data/               Database, tasks, datasets, shared environments, logs
docs/               Project documentation (not required at runtime; can be removed before release)
node_create_boot.md External Agent prompt for node creation (not required at runtime)
runtime/python/     Optional bundled portable Python (recommended for release builds)
```

## Launch
### Option 1: Compiled Build (End Users)
Download the compiled archive or installer from Releases. Double-click `EaseMeta.exe` on Windows. No Python, Node.js or npm required.

The FastAPI/Uvicorn backend is packaged inside the EXE. `runtime/python` is only used for training jobs and environment setup. Do **not** launch compiled builds with `python.exe main.py`.

The extracted/installed directory must preserve this structure at the same level as the EXE:
```text
EaseMeta.exe
config.yaml
runtime/python/        # Training and environment setup
web/                   # Compiled frontend assets
data/                  # Database, sample projects, tasks and logs
app/templates/         # Code templates for generated projects
app/sample_data/       # Sample corpora
```
- First launch initializes the database and 5 sample projects under `data/`.
- Defaults to a pywebview desktop window; a standalone Web API page is also available.
- Ports, data directories and frontend paths are controlled by `config.yaml`.
- Closing the window prompts: “Close window only” (backend and training continue) or “Quit application (stop backend)”.
- Duplicate launch attempts restore and focus the existing window instead of spawning duplicate backends.
- Training dependencies are installed into `runtime/python`. The training runtime never touches the system Python.

### Option 2: Source Launch (Development)
Source builds require Python 3.11+ (3.12 recommended). Install backend dependencies once before first run.

Windows:
```bat
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[desktop]"
.venv\Scripts\python main.py
```

Linux / macOS:
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[desktop]"
```

You may also run `python main.py` directly. If a complete portable Python exists under `runtime/python`, the launcher automatically switches to it:
```text
runtime/python/python.exe          # Windows
runtime/python/bin/python3         # Linux / macOS
```

`main.py` behavior:
- Prioritizes bundled Python if present
- Falls back to project `.venv` (if complete) when no bundled Python exists
- Prints installation instructions instead of throwing raw tracebacks when no valid runtime is found
- Serves API and prebuilt `web/` frontend in one process
- Creates a pywebview desktop window by default (does not auto-open browser)
- Window close prompt: detach window only or fully shut down backend
- Repeated startup restores the existing window rather than starting a new backend
- Console prints startup banner, frontend/API addresses and runtime logs

The web interface is always available. You may open `[http://127.0.0.1:8000/](http://127.0.0.1:8000/)` directly in a browser. Desktop window requires `pywebview` (included in bundled Python). If missing, the service falls back to HTTP-only mode.

Common CLI arguments:
```bash
python main.py --headless          # Server mode: HTTP only, no GUI
python main.py --browser           # Open frontend in browser instead of desktop window
python main.py --verbose           # Show info + access logs
python main.py --quiet             # Error logs only
python main.py --port 8010         # Change listening port
python main.py --reload            # Dev hot-reload mode (HTTP service only)
```
All paths and startup flags are defined in root `config.yaml`. Relative paths are resolved against the application root directory.

#### Frontend Build
npm is not required for runtime. Rebuild frontend assets after modifying source code:
```bash
cd frontend
npm run build
```
Build outputs go to `web/`, served directly by `main.py`. Release packages ship only `web/`; `frontend/` source and npm are optional.

Do **not** run `python app/main.py` directly. Always launch from repo root via `main.py`.

#### Frontend Dev Mode
```bash
cd frontend
npm install
npm run dev
```
Open [http://127.0.0.1:5173](http://127.0.0.1:5173). Vite proxies `/api` to port 8000 backend.

#### Training Interpreter & Environment
All projects share one training environment by default, priority order:
1. `runtime/python` bundled portable Python
2. `data/envs/shared` shared venv (auto-created if no bundled Python)
3. External interpreter manually assigned per project

Training dependencies are installed once and reused across all projects: `torch`, `PyYAML`, `tensorboard`. Image/audio/video tasks pull in `Pillow`, `soundfile`, `opencv-python` etc. on demand into the active training environment.
- When bundled Python is active, `pip` installs into its own site-packages. Training and environment checks use this runtime.
- Compiled EXE backend is isolated; `runtime/python` is solely for training and dependency installation.
- `data/envs/shared` is only used when no bundled Python is available, as a developer fallback.
- Bundled Python must be a full distribution with pip (or support `python -m ensurepip`). Windows embeddable distributions are incomplete by default and require site-packages + pip setup.
- The environment panel displays the runtime source: Bundled Python / Shared Environment for audit.

Environment panel capabilities:
- Auto-scan pip cache and install matching PyTorch; manual “Install PyTorch from pip cache” retry button
- Advanced: bind external Python interpreter per project
- Reset to bundled/shared environment

Deleting a project does not remove the shared training environment. Generated scripts still ship `requirements.txt` for standalone deployment.

## MCP Server
Run from application root:
```bash
python mcp_server.py
```
MCP clients communicate over stdio. Exposed tools:
- `list_node_specs`: Node contracts, ports and parameter JSON schemas.
- `validate_graph`: Validate graph, return diagnostics, bindings, parameter count and FLOPs.
- `generate_project`: Return complete project files after validation, no persistent side effects.

## Testing
```bash
.venv/Scripts/python -m pytest -q
```
Frontend type check and build:
```bash
cd frontend
npm run build
```

## Verified Scope
- 86 backend tests passing; frontend build passes.
- End-to-end TinyGPT / HF causal model training, checkpointing and generation.
- Dense vector 4-step, image 2-step, audio 2-step training and inference validated.
- MCP stdio initialization, tool list and node catalog calls validated.
- Image/audio/video VAE demos run 2 training steps successfully; image generation demo API runs 6 steps, training preview PNG renders in preview nodes.
- Video demo reads AVI via OpenCV; MJPEG frames play inline inside canvas nodes. Agent tool loop validated with local OpenAI-compatible mock endpoints.
- Alice in Wonderland (Project Gutenberg, 174KB corpus) preloaded at `data/llm_alice.txt`. Character-level TinyGPT trained for 1000 steps: train loss 2.3449, val loss 2.7558, val perplexity 15.73. `predict.py` generates text correctly.

## Known Limitations
- End-to-end SSH remote training, rsync transfer and cross-platform matrix (Linux/Windows/ARM) are implemented but not validated on real hardware.
- HF vision/audio/video classification adapters generate valid `AutoModelFor*Classification` code, though real training against HF checkpoints is untested. Native CNN classifiers and VAE generators are fully validated.
- Video demo uses `opencv-python` for AVI reading. HF video adapter code generation is complete but not run against real HF models.
- Proxy forwarding has not been validated against live proxy servers.
- Dual-input tensor ops (cat/add/multiply/matmul) and unused placeholder nodes are hidden from the library for now; old graphs still load. These may be enabled later or replaced with custom code nodes.
- Online node registry ships with no preset remote URLs; users must supply trusted JSON manifest addresses.
- HF vision/audio/video checkpoint download and training are unvalidated; bundled VAE and OpenCV demos are validated.

---
