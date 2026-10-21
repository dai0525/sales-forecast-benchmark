import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'repro'))
from retail_expansion import features,metrics


def panel():
    n,T=3,110
    return dict(y=np.arange(n*T,dtype=float).reshape(n,T),static=pd.DataFrame({'store_id':[1,2,3]}),dates=pd.date_range('2024-01-01',periods=T),cov={k:np.ones((n,T)) for k in ['discount','activity_flag','holiday_flag','precpt','avg_temperature','avg_humidity','avg_wind_level','stock_hour6_22_cnt','price']},H=7,period=7)


def test_future_realized_values_never_change_features():
    p=panel();o=76
    before={g:features(p,o,g) for g in ['history','plan','weather','stock','price']}
    p['y'][:,o:]=1e9
    for k in ['precpt','avg_temperature','avg_humidity','avg_wind_level','stock_hour6_22_cnt','price']:p['cov'][k][:,o:]=1e9
    for g,x in before.items():pd.testing.assert_frame_equal(x,features(p,o,g))
    p['cov']['discount'][:,o:o+7]=.5
    assert not features(p,o,'plan').equals(before['plan'])


def test_target_labels_not_reused_in_direct_horizon_lags():
    p=panel();p['H']=13;p['period']=52;o=78
    expected=p['y'][:,o+np.arange(13)-52].ravel()
    np.testing.assert_array_equal(features(p,o,'history')['seasonal_target_lag'],expected)


def test_zero_series_retained_in_aggregate_error():
    y=np.array([[0.,0.],[1.,1.]])
    pred=np.array([[1.,1.],[1.,1.]])
    m=metrics(pred,y)
    assert m['zero_denominator_series']==1 and m['mean_series_wape']==0
    assert m['wape']==1 and m['vn1_score']==2
