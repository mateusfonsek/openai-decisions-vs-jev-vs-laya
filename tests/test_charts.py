import matplotlib.pyplot as plt
import numpy as np

from bench.charts import logo_for, write_charts

M = {
    p: {s: {"accuracy": 0.8, "f1": 0.9, "exact": 0.7, "latency_p50": lat, "cost_per_1k": 0.02,
            "reliability": [{"conf": 0.9, "acc": 0.85, "n": 3}],
            "by_difficulty": {d: {"accuracy": 0.8, "f1": 0.9, "exact": 0.7}
                              for d in ("easy", "ambiguous", "adversarial")}}
        for s in ("routing", "injection", "judge")}
    for p, lat in [("openai", 200.0), ("jev", 230.0), ("laya", 24.0), ("laya-rot", 190.0)]
}
FILES = [f"{k}_{s}.png" for k in ("hero", "by_difficulty") for s in ("routing", "injection", "judge")] + ["reliability.png"]


def test_charts_without_logos_fall_back_to_text(tmp_path):
    write_charts(M, tmp_path / "out", tmp_path / "no-logos")
    for f in FILES:
        assert (tmp_path / "out" / f).stat().st_size > 0, f


def test_charts_with_logos(tmp_path):
    logos = tmp_path / "logos"
    logos.mkdir()
    for name in ("openai", "typesafe", "laya"):
        plt.imsave(logos / f"{name}.png", np.ones((32, 32, 4)))
    assert logo_for("jev", logos).name == "typesafe.png"
    assert logo_for("laya-rot", logos).name == "laya.png"
    write_charts(M, tmp_path / "out", logos)
    assert all((tmp_path / "out" / f).exists() for f in FILES)


def test_logo_for_missing_returns_none(tmp_path):
    assert logo_for("openai", tmp_path) is None


def test_spread_offsets_separates_close_points_and_keeps_far_ones():
    from bench.charts import spread_offsets

    pts = [(100.0, 100.0), (105.0, 102.0), (400.0, 300.0)]
    offs = spread_offsets(pts, min_dist=40)
    placed = [(x + dx, y + dy) for (x, y), (dx, dy) in zip(pts, offs)]
    for i in range(3):
        for j in range(i + 1, 3):
            d = ((placed[i][0] - placed[j][0]) ** 2 + (placed[i][1] - placed[j][1]) ** 2) ** 0.5
            assert d >= 40, (i, j, d)
    assert offs[0] == (0, 0) and offs[2] == (0, 0)
