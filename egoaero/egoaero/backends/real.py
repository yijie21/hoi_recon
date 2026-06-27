"""Real-backend registry. Each raises until its weights/repo are installed."""


def get(kind: str):
    raise NotImplementedError(
        f"real backend '{kind}' not installed; run in --mock or install via setup "
        "(HaWoR, SAM3, ORB-SLAM3, BundleSDF, SAM3D). See egoaero/README.md.")
