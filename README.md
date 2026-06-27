# hoi_recon — Hand-Object Interaction Reconstruction: methods workbench

A workbench for **reconstructing 4D hand-object interaction (HOI) from monocular RGB
video**, holding several reconstruction *methods* side by side so they can be run on
the same input, measured with the same metrics, and compared to find the best one.

Each method lives in its own self-contained subfolder (its own code, env, configs,
tests, and docs) and follows a shared **method contract** (below) so results line up
across methods.

## Methods

| Method | Status | What it is |
|---|---|---|
| [`render_and_compare/`](render_and_compare/) | ✅ runnable | Compositional, CHOIR-derived pipeline (arXiv:2605.20992): coarse contact-agnostic init → spatial rectification → contact-aware joint optimization, with render-and-compare object pose tracking. Runs end-to-end today in `mock` mode (no weights). See its [README](render_and_compare/README.md). |
| _(your next method)_ | — | Add a sibling folder following the contract below. |

## Method contract

Every method folder is a peer that implements the same job, so a future comparison
harness (or a human reading the numbers) can treat them interchangeably. A method
MUST be:

- **Self-contained.** Its own environment/build files, dependencies, tests, and docs
  live inside the folder. It is runnable from within its own directory without
  reaching into sibling folders. Runtime artifacts (`runs/`, `checkpoints/`,
  `third_party/`, `data/`) are created *inside* the method folder and are gitignored.

- **Input.** A monocular RGB video, plus optional camera intrinsics. Methods should
  accept the same clip so comparisons are like-for-like.

- **Output.** A per-clip result under `<method>/runs/<clip>/` describing the
  reconstructed 4D HOI: per-frame MANO hand, object mesh, object 6D-pose trajectory,
  and contact maps. (See `render_and_compare`'s on-disk bundle format as the
  reference layout.)

- **Evaluation.** Report the shared metric set so numbers are directly comparable
  across methods:
  - hand MPJPE (mm) and hand jitter/acceleration
  - object translation error (mm)
  - penetration depth
  - contact F1 and contact-frame gap (mm)

## Adding a new method

1. Create a sibling folder, e.g. `hamer_baseline/`, with the same self-contained shape
   (code + env file + tests + a method `README.md`).
2. Implement the contract: same video input, the documented `runs/<clip>/` output
   layout, and the shared eval metrics.
3. Add a row to the **Methods** table above pointing at the new folder.

A shared comparison harness (one input → side-by-side report across methods) is
intentionally deferred until a second method exists — the contract above is the
interface it will build on.
