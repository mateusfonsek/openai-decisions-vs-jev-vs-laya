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



def _placed(pts, offs):
    return [(x + dx, y + dy) for (x, y), (dx, dy) in zip(pts, offs)]


def _min_pair_dist(placed):
    return min(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
               for i, a in enumerate(placed) for b in placed[i + 1:])


def test_spread_offsets_separates_close_points_and_keeps_far_ones():
    from bench.charts import spread_offsets

    pts = [(100.0, 100.0), (105.0, 102.0), (400.0, 300.0)]
    offs = spread_offsets(pts, min_dist=40)
    assert _min_pair_dist(_placed(pts, offs)) >= 40 - 1e-9
    assert offs[2] == (0, 0)                               # ponto isolado não se move


def test_spread_offsets_vertical_only_and_keeps_order():
    from bench.charts import spread_offsets

    pts = [(100.0, 100.0), (104.0, 103.0)]                 # o 2º é o maior valor
    offs = spread_offsets(pts, min_dist=40)
    assert all(dx == 0 for dx, _ in offs)                  # latência lida não muda
    placed = _placed(pts, offs)
    assert placed[1][1] > placed[0][1]                     # quem é maior continua em cima
    assert max(abs(dy) for _, dy in offs) <= 40            # cada um anda no máximo um passo


def test_spread_offsets_respects_bounds():
    from bench.charts import spread_offsets

    pts = [(100.0, 100.0), (104.0, 101.0)]
    offs = spread_offsets(pts, min_dist=40, bounds=(0, 0, 500, 110))
    placed = _placed(pts, offs)
    assert all(0 <= y <= 110 for _, y in placed)
    assert _min_pair_dist(placed) >= 40 - 1e-9 and placed[1][1] > placed[0][1]
