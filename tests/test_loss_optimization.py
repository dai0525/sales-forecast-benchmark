import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'repro'))
from loss_optimization import MODELS,choose,ratio


def test_calibration_uses_earlier_window_and_can_reject_correction():
    pred={m:np.ones((2,2)) for m in MODELS}
    y0=np.full((2,2),2.);y1=np.ones((2,2))
    result=choose(pred,pred,y0,y1,'vn1_score')
    for opt in result['options'].values():
        assert opt['first_origin_factor']==2
        assert opt['strength']==0
        assert opt['final_factor']==1.5
        assert opt['selection_score']==0


def test_full_correction_selected_when_transfer_is_correct():
    pred={m:np.ones((2,2)) for m in MODELS};truth=np.full((2,2),1.5)
    result=choose(pred,pred,truth,truth,'wape')
    for opt in result['options'].values():
        assert opt['strength']==1 and opt['selection_score']==0


def test_zero_predictions_and_extreme_ratios_are_bounded():
    assert ratio(np.zeros((2,2)),np.ones((2,2)))==1
    assert ratio(np.ones((2,2)),np.full((2,2),100))==2
    assert ratio(np.ones((2,2)),np.zeros((2,2)))==.5


def test_business_weight_fit_includes_errors_on_zero_demand_series():
    from ase_objective_alignment import fit_weights
    y=np.array([[0.],[10.]])
    candidates={'bad_on_zero':np.array([[100.],[10.]]),'safer':np.array([[0.],[8.]])}
    result=fit_weights(candidates,y,'wape','arithmetic')
    assert result['weights']['bad_on_zero']==0
    assert result['weights']['safer']==1
