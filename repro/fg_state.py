"""Exact Gaussian forward messages for local damped trend + Fourier seasonality.

No shared calendar assumption: each series has its own Fourier coefficients.
All series share hyperparameters and normalized prior covariance, so a single
Riccati recursion suffices. Conditional on hyperparameters this is Kalman/BP,
not a new inference algorithm. Parameters are selected outside this module.
"""
import numpy as np


def forecast(y, h=13, harmonics=2, qr=.02, trend_q=0., damp=.98, period=52,
             seasonal_var=1., return_components=False, covariance_mode="full"):
    if covariance_mode not in {"full", "block", "information"}:
        raise ValueError("Unknown covariance mode")
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or y.shape[1] < 2 or not np.isfinite(y).all():
        raise ValueError('Expected finite (series, time) observations')
    if h < 1 or qr <= 0 or trend_q < 0 or not 0 <= damp <= 1:
        raise ValueError('Invalid horizon or noise/damping parameters')
    if harmonics < 0 or 2 * harmonics >= period or seasonal_var < 0:
        raise ValueError('Invalid Fourier prior')
    n, tmax = y.shape
    scale = np.maximum(np.std(y, axis=1), .02*np.abs(y).mean(axis=1)+1e-6)
    z = y/scale[:, None]
    d = 2+2*harmonics
    F = np.eye(d); F[0,1] = damp; F[1,1] = damp
    Q = np.diag([qr, trend_q]+[0.]*(d-2))
    # Data-independent proper prior. No observation is reused to initialize it.
    P = np.diag([1e6, 0. if trend_q == 0 else 1., *([seasonal_var]*(d-2))])
    state = np.zeros((n,d))
    def design(t):
        angles = 2*np.pi*np.arange(1,harmonics+1)*t/period
        return np.r_[1.,0.,np.cos(angles),np.sin(angles)]
    for t in range(tmax):
        if t:
            state = state@F.T; P = F@P@F.T+Q
        a = design(t); v = P@a; s = float(a@v+1.)
        K = v/s
        if covariance_mode == "information":
            # Independent precision-form observation update on nondegenerate states.
            active = np.flatnonzero(np.diag(P) > 0)
            prior = P[np.ix_(active, active)]
            precision = np.linalg.solve(prior, np.eye(len(active)))
            aa = a[active]
            posterior = np.linalg.solve(precision + np.outer(aa, aa), np.eye(len(active)))
            rhs = state[:,active]@precision + z[:,t,None]*aa
            state[:,active] = rhs@posterior.T
            P = np.zeros_like(P); P[np.ix_(active,active)] = posterior
        else:
            state += (z[:,t]-state@a)[:,None]*K
            # Joseph update preserves PSD and retains cross covariance.
            J = np.eye(d)-np.outer(K,a)
            P = J@P@J.T+np.outer(K,K)
        P = (P+P.T)/2
        if covariance_mode == "block":
            # Approximate posterior projection: retain within-block covariance,
            # discard level/trend <-> seasonal dependence after every observation.
            P[:2,2:] = 0.; P[2:,:2] = 0.

    out = np.empty((n,h)); level = out.copy(); season = out.copy(); variance=out.copy()
    for j in range(h):
        state = state@F.T; P = F@P@F.T+Q; a=design(tmax+j)
        level[:,j] = state[:,0]*scale
        season[:,j] = (state[:,2:]@a[2:])*scale
        out[:,j] = level[:,j]+season[:,j]
        variance[:,j] = (a@P@a+1)*scale**2
    pred=np.maximum(out,0.)
    if return_components:
        return pred, {'level':level,'season':season,'raw_mean':out,'variance':variance}
    return pred
