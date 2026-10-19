"""Regression tests for the second-dataset (Rossmann) benchmark.

These check the properties the audit protocol depends on, using only the
artifacts (no 38 MB CSV needed): metric definition, zero-denominator policy,
and that every saved prediction file lines up with its truth/id vector.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data" / "alt" / "rossmann" / "panel.npz"
RES = ROOT / "audit" / "results" / "rossmann"
PRED = RES / "preds"

pytestmark = pytest.mark.skipif(not PANEL.exists(), reason="Rossmann panel not built")


def _bench():
    sys.path.insert(0, str(ROOT / "repro" / "alt"))
    import bench_rossmann
    return bench_rossmann


def test_wape_definition():
    b = _bench()
    y = np.array([10.0, 0.0, 5.0, 5.0])
    p = np.array([8.0, 1.0, 5.0, 4.0])
    assert b.series_wape(y, p, "x") == pytest.approx(4.0 / 20.0)


def test_zero_denominator_raises():
    b = _bench()
    with pytest.raises(ValueError, match="zero WAPE denominator"):
        b.series_wape(np.zeros(4), np.ones(4), "S0")


def test_negative_predictions_clipped():
    b = _bench()
    y = np.array([2.0, 2.0])
    out = b.score(y[None, :], np.array([[-5.0, 3.0]]), ["S0"], y[None, :])
    assert out["n_negative_clipped"] == 1
    assert out["wape_mean"] == pytest.approx(3.0 / 4.0)


def test_splits_do_not_touch_test_window():
    """Hyperparameter selection must stay strictly before every outer origin."""
    b = _bench()
    assert b.VAL_ORIGIN + b.H <= min(b.OUTER_ORIGINS)
    assert b.OUTER_ORIGINS[-1] + b.H <= 942


def test_saved_predictions_align():
    if not PRED.exists():
        pytest.skip("no predictions")
    truth = np.load(PRED / "truth_o900.npz", allow_pickle=False)
    ids = [str(s) for s in truth["ids"]]
    assert truth["truth"].shape == (len(ids), 42)
    files = sorted(PRED.glob("preds_*_o900.npz"))
    assert files, "no prediction files"
    for f in files:
        z = np.load(f, allow_pickle=False)
        assert [str(s) for s in z["ids"]] == ids, f.name
        assert z["preds"].shape == z["truth"].shape == (len(ids), 42), f.name
        assert np.isfinite(z["preds"]).all(), f.name
        assert (z["preds"] >= 0).all(), f.name


def test_metrics_match_saved_predictions():
    if not (RES / "metrics.json").exists():
        pytest.skip("no metrics")
    m = json.loads((RES / "metrics.json").read_text(encoding="utf-8"))
    for key in ("o816", "o858", "o900"):
        for model, rec in m["origins"][key]["models"].items():
            stem = model.replace("+", "plus")
            z = np.load(PRED / f"preds_{stem}_{key}.npz", allow_pickle=False)
            t = np.load(PRED / f"truth_{key}.npz", allow_pickle=False)["truth"]
            p = np.clip(z["preds"].astype(float), 0, None)
            agg = np.abs(t - p).sum() / t.sum()
            assert agg == pytest.approx(rec["wape_aggregate"], abs=1e-6), f"{model}/{key}"


def test_future_sales_cannot_change_training_or_forecast_features():
    import pandas as pd
    b=_bench();p=b.Panel(PANEL,n_series=4);o=140
    X,y=b.build_training(p,o,True);features=b.pd_frame(b.build_matrix(p,o,1,True))
    p.y[:,o:]=1234567
    X2,y2=b.build_training(p,o,True);features2=b.pd_frame(b.build_matrix(p,o,1,True))
    pd.testing.assert_frame_equal(X,X2);pd.testing.assert_frame_equal(features,features2)
    np.testing.assert_array_equal(y,y2)


def test_subset_run_does_not_overwrite_published_metrics(tmp_path,monkeypatch):
    b=_bench();monkeypatch.setattr(b,'OUTDIR',tmp_path)
    published=tmp_path/'metrics.json';published.write_text('preserve')
    monkeypatch.setattr(sys,'argv',['bench_rossmann.py','--quick','--n-series','3','--models','SeasonalNaive7'])
    b.main()
    assert published.read_text()=='preserve'
    assert (tmp_path/'runs/quick-3-SeasonalNaive7/smoke.json').exists()


def test_unknown_model_is_rejected(tmp_path,monkeypatch):
    b=_bench();monkeypatch.setattr(b,'OUTDIR',tmp_path)
    monkeypatch.setattr(sys,'argv',['bench_rossmann.py','--models','typo'])
    with pytest.raises(SystemExit) as e:b.main()
    assert e.value.code==2
    assert not (tmp_path/'metrics.json').exists()
