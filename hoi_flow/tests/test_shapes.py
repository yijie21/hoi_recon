"""CPU-only shape / behavior contracts for the hoi_flow model core. Tiny dims, no GPU,
no network, no weights. Run:
  cd /workspace/code/hoi_recon && python -m pytest hoi_flow/tests -q
"""
import torch

from hoi_flow.bridge import bridge_loss, sample, _window_bounds
from hoi_flow.conditioning import CondEncoders, MODALITIES
from hoi_flow.geometry import matrix_to_rot6d, rot6d_to_matrix, so3_geodesic
from hoi_flow.model import HOIDiT
from hoi_flow.state import D, ROT6D_DIMS, Normalizer, pack_state, unpack_state

TINY = {"d": 64, "depth": 2, "heads": 4, "mlp_ratio": 2, "t_freq": 64}


# ---------------------------------------------------------------- geometry / state
def test_rot6d_roundtrip_exact():
    d6 = torch.randn(20, 6)
    R = rot6d_to_matrix(d6)
    R2 = rot6d_to_matrix(matrix_to_rot6d(R))  # 6d->R->6d->R must be a fixed point
    assert torch.allclose(R, R2, atol=1e-5)
    eye = torch.eye(3).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(so3_geodesic(R, R), torch.zeros(20), atol=2e-3)  # arccos ill-cond. near 0


def test_pack_unpack_with_nan_hands():
    T = 10
    obj_R = rot6d_to_matrix(torch.randn(T, 6))
    obj_t = torch.randn(T, 3)
    obj_pose = torch.zeros(T, 4, 4)
    obj_pose[:, :3, :3], obj_pose[:, :3, 3], obj_pose[:, 3, 3] = obj_R, obj_t, 1.0

    wrist = torch.full((T, 2, 4, 4), float("nan"))
    lw_R = rot6d_to_matrix(torch.randn(T, 6))
    lw_t = torch.randn(T, 3)
    wrist[:, 0] = torch.eye(4)                             # proper SE3 bottom row
    wrist[:, 0, :3, :3], wrist[:, 0, :3, 3] = lw_R, lw_t   # left present
    thetas = torch.full((T, 2, 15), float("nan"))
    thetas[:, 0] = torch.randn(T, 15)                                            # right stays NaN

    x, valid = pack_state(obj_pose, wrist, thetas)
    assert not torch.isnan(x).any()                       # no NaN leaks into x
    assert valid[:, 0:9].all()                            # object present
    assert valid[:, 9:33].all()                           # left hand present
    assert not valid[:, 33:57].any()                      # right hand absent

    d = unpack_state(x)                                    # roundtrip on valid entries
    assert torch.allclose(d["obj_pose"][:, :3, :3], obj_R, atol=1e-5)
    assert torch.allclose(d["obj_pose"][:, :3, 3], obj_t, atol=1e-5)
    assert torch.allclose(d["hand_wrist"][:, 0, :3, :3], lw_R, atol=1e-5)
    assert torch.allclose(d["hand_thetas"][:, 0], thetas[:, 0], atol=1e-5)


def test_pack_unpack_batched_shapes():
    x = torch.randn(3, 10, D)
    d = unpack_state(x)
    assert d["obj_pose"].shape == (3, 10, 4, 4)
    assert d["hand_wrist"].shape == (3, 10, 2, 4, 4)
    assert d["hand_thetas"].shape == (3, 10, 2, 15)


def test_normalizer_roundtrip_and_rot_untouched(tmp_path):
    xs = [torch.randn(10, D) * 3 + 1 for _ in range(3)]
    nrm = Normalizer().fit(xs)
    x = torch.randn(5, D)
    assert torch.allclose(nrm.inverse(nrm.transform(x)), x, atol=1e-5)
    assert torch.allclose(nrm.transform(x)[:, ROT6D_DIMS], x[:, ROT6D_DIMS], atol=1e-6)
    p = tmp_path / "norm.json"
    nrm.save(p)
    assert torch.allclose(Normalizer.load(p).transform(x), nrm.transform(x), atol=1e-6)


# ---------------------------------------------------------------- conditioning
def _cond_batch(B=2, T=10, R=32):
    return {
        "rgb": torch.rand(B, T, 3, R, R),
        "depth": torch.rand(B, T, R, R),
        "seg_mask": torch.randint(0, 5, (B, T, R, R)),
        "obj_points": torch.randn(B, 200, 3),
        "K": torch.randn(B, 3, 3),
    }


def test_cond_token_counts():
    B, T = 2, 10
    enc = CondEncoders(64, {m: True for m in MODALITIES} | {"rgb_backbone": "tinycnn"})
    batch = _cond_batch(B, T)
    tok, meta = enc(batch)
    assert tok.shape == (B, 16 * T * 3 + 16 + 1, 64)     # 497 at T=10
    assert meta["n_tokens"] == tok.shape[1]

    for name, cnt in [("rgb", 16 * T), ("depth", 16 * T), ("seg_mask", 16 * T),
                      ("obj_points", 16), ("K", 1)]:
        drop = {m: True for m in MODALITIES if m != name}
        t, _ = enc(batch, drop=drop)
        assert t.shape == (B, cnt, 64), name

    # all dropped -> single learned null token
    t, meta = enc(batch, drop={m: True for m in MODALITIES})
    assert t.shape == (B, 1, 64)
    assert meta["layout"] == [("null", 1)]


# ---------------------------------------------------------------- model
def test_model_forward_shapes_and_identity_init():
    B, T = 2, 10
    model = HOIDiT(TINY)
    x_t = torch.randn(B, T, D)
    t = torch.rand(B)
    cond = torch.randn(B, 5, TINY["d"])
    pres = torch.ones(B, T, 2, dtype=torch.bool)
    out = model(x_t, t, cond, pres)
    assert out.shape == (B, T, D)
    assert torch.allclose(out, x_t, atol=1e-5)           # untrained == identity refiner


# ---------------------------------------------------------------- bridge
def test_bridge_loss_finite():
    B, T = 2, 10
    model = HOIDiT(TINY)
    x0, x1 = torch.randn(B, T, D), torch.randn(B, T, D)
    valid = torch.ones(B, T, D, dtype=torch.bool)
    cond = torch.randn(B, 5, TINY["d"])
    pres = torch.ones(B, T, 2, dtype=torch.bool)
    loss, logs = bridge_loss(model, x0, x1, valid, cond, pres)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(v)) for v in logs.values())


def test_bridge_loss_trains_down():
    torch.manual_seed(0)
    B, T = 4, 10
    model = HOIDiT(TINY)
    x1 = torch.randn(B, T, D)
    x0 = x1 + 0.5                                          # fixed constant offset
    valid = torch.ones(B, T, D, dtype=torch.bool)
    cond = torch.randn(B, 5, TINY["d"])
    pres = torch.ones(B, T, 2, dtype=torch.bool)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    def eval_loss():                                       # fixed t/eps for apples-to-apples
        g = torch.Generator().manual_seed(123)
        with torch.no_grad():
            return bridge_loss(model, x0, x1, valid, cond, pres, rng=g)[0].item()

    before = eval_loss()
    for _ in range(60):
        opt.zero_grad()
        loss, _ = bridge_loss(model, x0, x1, valid, cond, pres)
        loss.backward()
        opt.step()
    after = eval_loss()
    assert after < before / 5.0, (before, after)


# ---------------------------------------------------------------- sampler
def test_sample_stitches_and_shape():
    torch.manual_seed(0)
    model = HOIDiT(TINY)
    T = 50
    x0 = torch.randn(T, D)
    pres = torch.ones(T, 2, dtype=torch.bool)
    ctok = torch.randn(1, 5, TINY["d"])
    bounds = _window_bounds(T, 16, 8)
    assert bounds[0][0] == 0 and bounds[-1][1] == T       # full coverage
    out = sample(model, x0, lambda sl: ctok, pres, n_steps=4, window=16, overlap=8, seed=0)
    assert out.shape == (T, D)
    assert torch.isfinite(out).all()


def test_sample_guidance_pulls_toward_minimum():
    torch.manual_seed(0)
    model = HOIDiT(TINY)
    T = 40
    x0 = torch.randn(T, D) + 2.0                          # start far from the minimum at 0
    pres = torch.ones(T, 2, dtype=torch.bool)
    ctok = torch.randn(1, 5, TINY["d"])

    def energy(x):                                        # window-shape-agnostic, min at x==0
        return (x ** 2).sum()

    kw = dict(n_steps=8, window=16, overlap=8, seed=0)
    free = sample(model, x0, lambda sl: ctok, pres, guidance=(), **kw)
    guided = sample(model, x0, lambda sl: ctok, pres, guidance=[(energy, 0.5)], **kw)
    assert guided.pow(2).sum() < free.pow(2).sum()        # guidance pulled samples toward 0


def test_sample_all_modalities_dropped_runs():
    torch.manual_seed(0)
    model = HOIDiT(TINY)
    enc = CondEncoders(TINY["d"], {m: True for m in MODALITIES} | {"rgb_backbone": "tinycnn"})
    T = 20
    batch = _cond_batch(B=1, T=T)
    x0 = torch.randn(T, D)
    pres = torch.ones(T, 2, dtype=torch.bool)

    def cond(sl):  # everything dropped -> null token
        return enc({k: v for k, v in batch.items() if k in ("obj_points", "K")},
                   drop={m: True for m in MODALITIES})[0]

    out = sample(model, x0, cond, pres, n_steps=4, window=16, overlap=8, seed=0)
    assert out.shape == (T, D) and torch.isfinite(out).all()
