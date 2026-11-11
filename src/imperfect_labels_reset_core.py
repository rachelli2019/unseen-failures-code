"""Agreed Q=P algorithms: false-negative IPW calibration and PPI with r=0."""
import numpy as np


class FalseNegativeCalibrator:
    """Common fixed-kappa mixture; no restarts or threshold multiplicity split."""
    def __init__(self, zeta, delta, rho, points=100):
        if not (0 < zeta < 1 and 0 < delta < 1 and 0 < rho < 1 and points > 0):
            raise ValueError('invalid calibration parameters')
        self.zeta, self.delta, self.rho = zeta, delta, rho
        # Integer Luna scores 0..100: these include every strict-threshold region,
        # including +inf (accept score 100) and -inf (accept nothing).
        self.thresholds = np.r_[np.arange(101)/100, np.inf]
        self.kappa = (np.arange(points)+.5)/points / ((1-rho)/rho+zeta)
        self.base = np.log1p(self.kappa*zeta)
        self.correction = np.zeros((len(self.thresholds), points))
        self.n = 0
        self.threshold = -np.inf

    def propensity(self, score):
        if not 0 <= score <= 1:
            raise ValueError('uncertainty must lie in [0,1]')
        return self.rho if score < self.threshold else 1.

    def log_wealth(self, index):
        values = self.n*self.base+self.correction[index]
        return np.logaddexp.reduce(values)-np.log(len(self.kappa))

    def update(self, score, false_negative, probability):
        """None means unaudited. Only audited true-unsafe/Luna-safe pairs are losses."""
        if probability != self.propensity(score):
            raise ValueError('feedback must use the pre-update propensity')
        if false_negative is None and probability == 1:
            raise ValueError('missing forced audit')
        if false_negative not in (None, 0, 1):
            raise ValueError('false-negative feedback must be binary')
        if false_negative == 1:
            loss = (1-self.rho)/probability
            start = np.searchsorted(self.thresholds, score, side='right')
            self.correction[start:] += np.log1p(self.kappa*(self.zeta-loss))-self.base
        self.n += 1
        lo, hi = -1, len(self.thresholds)
        while hi-lo > 1:
            mid = (lo+hi)//2
            if self.log_wealth(mid) >= np.log(1/self.delta):
                lo = mid
            else:
                hi = mid
        self.threshold = self.thresholds[lo] if lo >= 0 else -np.inf


class OnlineNegativeResidual:
    """Past audited novelty among Luna negatives, shrunk toward zero.

    Constant random auditing makes these audits representative; no IPS weights
    are needed. No category/novelty feature is used to make a prediction.
    """
    def __init__(self, bins=10, prior_count=10.):
        if not isinstance(bins, int) or bins < 1 or not prior_count > 0:
            raise ValueError('invalid residual parameters')
        self.bins = bins
        self.prior_count = prior_count
        self.count = np.zeros(bins)
        self.total = np.zeros(bins)

    def bin(self, score):
        if not 0 <= score <= 1:
            raise ValueError('invalid uncertainty score')
        return min(int(score*self.bins), self.bins-1)

    def predict(self, hat, score):
        if hat not in (0, 1):
            raise ValueError('invalid prediction')
        k = self.bin(score)
        return 0. if hat else self.total[k]/(self.prior_count+self.count[k])

    def update(self, score, audited_novelty):
        if audited_novelty not in (0, 1):
            raise ValueError('residual update requires audited binary novelty')
        k = self.bin(score)
        self.count[k] += 1
        self.total[k] += audited_novelty


def label_round(hat, probability, coin, seen, audit, categorize, residual=0.):
    """Callbacks enforce the information boundary; no hidden truth/category input.

    Categorize Luna positives BEFORE the coin. Audited false negatives may be
    categorized afterward. An unaudited Luna negative invokes neither callback.
    Returns Y, emitted discovery, audit/categorizer counts, and audited FN loss.
    """
    if hat not in (0, 1) or not 0 < probability <= 1 or not 0 <= coin < 1:
        raise ValueError('invalid round inputs')
    category = categorize() if hat else None
    calls = int(hat)
    if not np.isfinite(residual) or not 0 <= residual <= 1:
        raise ValueError('invalid residual prediction')
    mu = float(np.clip(int(hat and category is not None and category not in seen)+residual, 0, 1))
    queried = coin < probability
    truth = audit() if queried else None
    if queried and truth not in (0, 1):
        raise ValueError('audit must return binary truth')
    if queried and truth and not hat:
        category = categorize()
        calls += 1
    if queried and truth and category is None:
        raise ValueError('unsafe audit requires a valid category')
    if queried and not truth and category is not None:
        raise ValueError('categorizer contradicts safe truth')
    # The true indicator is used ONLY on audited draws. Do not set latent truth
    # to the emitted label: false negatives must be corrected by inverse weighting.
    true_z = int(truth and category is not None and category not in seen) if queried else None
    y = float(mu) if not queried else mu+(true_z-mu)/probability
    emitted = truth if queried else hat
    discovery = int(emitted and category is not None and category not in seen)
    if emitted and category is not None:
        seen.add(category)
    fn = int(truth == 1 and hat == 0) if queried else None
    return y, discovery, int(queried), calls, fn
