"""Общие функции: регионы, региональные признаки, модель возраста, метрики."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LassoCV, Lasso, lasso_path
from sklearn.model_selection import KFold
from scipy import stats

MIN_COV = 5


# ----------------------------------------------------------------------------- данные
class Methylome:
    """Обёртка над матрицей counts: образцы × CpG (uint16)."""

    def __init__(self, keys, mc, cov, sample_ids, meta):
        self.keys = keys
        self.chr = (keys >> 32).astype(np.int32)
        self.pos = (keys & ((1 << 32) - 1)).astype(np.int64)
        self.mc = mc
        self.cov = cov
        self.ids = list(sample_ids)
        self.meta = meta.set_index("sample_id").loc[self.ids].reset_index()
        self.row = {s: i for i, s in enumerate(self.ids)}

    def rows(self, split):
        return np.where(self.meta["split"].values == split)[0]

    def beta(self, rows, sites=None, mc=None, cov=None, min_cov=MIN_COV):
        mc = self.mc if mc is None else mc
        cov = self.cov if cov is None else cov
        sites = np.arange(cov.shape[1]) if sites is None else np.asarray(sites)
        rows = np.asarray(rows)
        b = np.full((len(rows), len(sites)), np.nan, dtype=np.float32)
        for j0 in range(0, len(sites), 100000):  # поблочно: пиковая память ~ размер результата
            sj = sites[j0:j0 + 100000]
            c = cov[np.ix_(rows, sj)].astype(np.float32)
            m = mc[np.ix_(rows, sj)].astype(np.float32)
            with np.errstate(invalid="ignore", divide="ignore"):
                b[:, j0:j0 + 100000] = np.where(c >= min_cov, m / c, np.nan)
        return b


def degrade_counts(mc, cov, mode, level, rng):
    """mode='mask': случайно удалить долю `level` наблюдаемых CpG;
    mode='thin': биномиальное прореживание чтений с вероятностью сохранения `level`."""
    mc = mc.astype(np.int64); cov = cov.astype(np.int64)
    if mode == "mask":
        drop = rng.random(cov.shape) < level
        cov = np.where(drop, 0, cov); mc = np.where(drop, 0, mc)
    elif mode == "thin":
        m2 = rng.binomial(mc, level)
        u2 = rng.binomial(cov - mc, level)
        mc, cov = m2, m2 + u2
    return mc.astype(np.uint16), cov.astype(np.uint16)


# ----------------------------------------------------------------------------- регионы
def regions_from_labels(chr_, pos, labels, site_idx):
    """labels >= 0 — номер региона для сайтов site_idx; возвращает таблицу регионов и массив принадлежности."""
    df = pd.DataFrame({"site": site_idx, "chr": chr_[site_idx], "pos": pos[site_idx], "lab": labels})
    df = df[df.lab >= 0]
    g = df.groupby("lab", sort=False)
    reg = pd.DataFrame({"chr": g.chr.first(), "start": g.pos.min(), "end": g.pos.max() + 2, "n_cpg": g.size()})
    reg = reg.sort_values(["chr", "start"]).reset_index()
    remap = pd.Series(np.arange(len(reg)), index=reg["lab"].values)
    member = np.full(len(chr_), -1, dtype=np.int64)
    member[df.site.values] = remap.loc[df.lab.values].values
    return reg.drop(columns="lab"), member


def dbscan_regions(chr_, pos, site_idx, eps=1000, min_samples=5):
    """DBSCAN по геномной координате (как DRegBC в Simpson et al., 2023: eps в п.н., minPts=5)."""
    labels = np.full(len(site_idx), -1, dtype=np.int64)
    offset = 0
    c = chr_[site_idx]
    for ch in np.unique(c):
        w = np.where(c == ch)[0]
        x = pos[site_idx[w]].reshape(-1, 1).astype(np.float64)
        lab = DBSCAN(eps=eps, min_samples=min_samples, algorithm="kd_tree").fit_predict(x)
        ok = lab >= 0
        labels[w[ok]] = lab[ok] + offset
        offset += (lab.max() + 1) if ok.any() else 0
    return regions_from_labels(chr_, pos, labels, site_idx)


def window_regions(chr_, pos, site_idx, width=5000, min_cpg=1):
    """Скользящие (непересекающиеся) окна фиксированной длины, как RegBC 5 kb."""
    lab = chr_[site_idx].astype(np.int64) * 10**7 + pos[site_idx] // width
    u, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    inv = np.where(cnt[inv] >= min_cpg, inv, -1)
    return regions_from_labels(chr_, pos, inv, site_idx)


def interval_regions(chr_, pos, site_idx, intervals, min_cpg=3):
    """Регионы по внешним интервалам (BED, 0-based полуоткрытые) — например, MethyLasso или ChromHMM.
    intervals: DataFrame chr(int-индекс), start, end, неперекрывающиеся."""
    labels = np.full(len(site_idx), -1, dtype=np.int64)
    c = chr_[site_idx]; p = pos[site_idx]
    for ch, iv in intervals.groupby("chr"):
        iv = iv.sort_values("start")
        w = np.where(c == ch)[0]
        if len(w) == 0:
            continue
        # CGfinal-позиции трактуются как 1-based координата C; BED-интервал [start, end) 0-based
        p0 = p[w] - 1
        k = np.searchsorted(iv.start.values, p0, side="right") - 1
        ok = (k >= 0) & (p0 < iv.end.values[np.clip(k, 0, None)])
        labels[w[ok]] = iv.index.values[k[ok]]
    reg, member = regions_from_labels(chr_, pos, labels, site_idx)
    keep = reg.n_cpg.values >= min_cpg
    newid = np.full(len(reg), -1); newid[keep] = np.arange(keep.sum())
    member = np.where(member >= 0, newid[np.clip(member, 0, None)], -1)
    return reg[keep].reset_index(drop=True), member


def intersect_partitions(member_a, member_b, chr_, pos, min_cpg=3):
    """Пересечение двух разбиений: сайт остаётся в регионе, только если он принадлежит обоим."""
    ok = (member_a >= 0) & (member_b >= 0)
    lab = np.where(ok, member_a.astype(np.int64) * 10**7 + member_b, -1)
    site_idx = np.where(ok)[0]
    u, inv = np.unique(lab[site_idx], return_inverse=True)
    reg, member = regions_from_labels(chr_, pos, inv, site_idx)
    keep = reg.n_cpg.values >= min_cpg
    newid = np.full(len(reg), -1); newid[keep] = np.arange(keep.sum())
    member = np.where(member >= 0, newid[np.clip(member, 0, None)], -1)
    return reg[keep].reset_index(drop=True), member


# ----------------------------------------------------------------------------- признаки
def _assign(member, xnorm=None, nbins=None):
    sites = np.where(member >= 0)[0]
    nreg = member.max() + 1
    if nbins is None:
        cols = member[sites]; ncol = nreg
    else:
        b = np.minimum((xnorm[sites] * nbins).astype(np.int64), nbins - 1)
        cols = member[sites] * nbins + b; ncol = nreg * nbins
    A = sp.csr_matrix((np.ones(len(sites), np.float32), (np.arange(len(sites)), cols)), shape=(len(sites), ncol))
    return sites, A


def relative_position(member, pos):
    """Нормированная координата CpG внутри региона: 0 — первый CpG, 1 — последний."""
    x = np.zeros(len(member), np.float32)
    s = np.where(member >= 0)[0]
    m = member[s]
    mn = pd.Series(pos[s]).groupby(m).transform("min").values
    mx = pd.Series(pos[s]).groupby(m).transform("max").values
    x[s] = (pos[s] - mn) / np.maximum(mx - mn, 1)
    return x


def region_features(M, rows, member, kinds=("mean",), nbins=4, mc=None, cov=None, chunk=40, min_obs_pattern=3):
    """Признаки региона по наблюдаемым CpG (покрытие >= MIN_COV) каждого образца.
    mean — среднее beta; sd — стандартное отклонение между CpG; slope — наклон beta по нормированной
    координате; enddiff — среднее последней трети минус среднее первой трети; bins — средние nbins
    последовательных бинов. Пустые значения = NaN (далее импутируются медианой train)."""
    x = relative_position(member, M.pos)
    sites, A = _assign(member)
    xs = x[sites]
    Ax = sp.csr_matrix(A.multiply(xs[:, None])); Ax2 = sp.csr_matrix(A.multiply((xs ** 2)[:, None]))
    first = sp.csr_matrix(A.multiply((xs <= 1 / 3)[:, None].astype(np.float32)))
    last = sp.csr_matrix(A.multiply((xs >= 2 / 3)[:, None].astype(np.float32)))
    if "bins" in kinds:
        _, Ab = _assign(member, x, nbins)
    out = {k: [] for k in kinds}
    for s0 in range(0, len(rows), chunk):
        r = rows[s0:s0 + chunk]
        B = M.beta(r, sites, mc=mc, cov=cov)
        O = np.isfinite(B).astype(np.float32)
        B0 = np.nan_to_num(B)
        n = O @ A; sb = B0 @ A
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sb / n
            if "mean" in kinds:
                out["mean"].append(mean)
            if "sd" in kinds:
                sb2 = (B0 ** 2) @ A
                var = sb2 / n - mean ** 2
                sd = np.sqrt(np.clip(var * n / np.maximum(n - 1, 1), 0, None))
                sd[n < min_obs_pattern] = np.nan
                out["sd"].append(sd)
            if "slope" in kinds:
                sx = O @ Ax; sx2 = O @ Ax2; sxb = B0 @ Ax
                den = n * sx2 - sx ** 2
                slope = (n * sxb - sx * sb) / den
                slope[(n < min_obs_pattern) | (np.abs(den) < 1e-6)] = np.nan
                out["slope"].append(slope)
            if "enddiff" in kinds:
                nf = O @ first; nl = O @ last
                d = (B0 @ last) / nl - (B0 @ first) / nf
                d[(nf < 1) | (nl < 1)] = np.nan
                out["enddiff"].append(d)
            if "bins" in kinds:
                nb = O @ Ab
                out["bins"].append((B0 @ Ab) / nb)
    return {k: np.vstack(v).astype(np.float32) for k, v in out.items()}


def cpg_features(M, rows, sites, mc=None, cov=None):
    return M.beta(rows, sites, mc=mc, cov=cov)


# ----------------------------------------------------------------------------- модель
class Preprocessor:
    """Импутация медианой train и стандартизация по train; признаки без наблюдений в train удаляются."""

    def fit(self, X):
        p = X.shape[1]
        self.med = np.full(p, np.nan, np.float32); self.mu = np.zeros(p, np.float32); self.sd = np.zeros(p, np.float32)
        for j0 in range(0, p, 50000):
            blk = np.array(X[:, j0:j0 + 50000], dtype=np.float32)
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    med = np.nanmedian(blk, axis=0)
            bad = ~np.isfinite(blk)
            blk[bad] = np.broadcast_to(med, blk.shape)[bad]
            self.med[j0:j0 + 50000] = med; self.mu[j0:j0 + 50000] = blk.mean(0); self.sd[j0:j0 + 50000] = blk.std(0)
        self.keep = np.isfinite(self.med) & (self.sd > 1e-8)
        return self

    def transform(self, X):
        # одна копия: отбор столбцов, затем импутация и стандартизация на месте (экономия памяти при ~10^6 CpG)
        Z = np.array(X[:, self.keep], dtype=np.float32, order="F")
        med, mu, sd = self.med[self.keep], self.mu[self.keep], self.sd[self.keep]
        for j0 in range(0, Z.shape[1], 50000):
            blk = Z[:, j0:j0 + 50000]
            bad = ~np.isfinite(blk)
            if bad.any():
                blk[bad] = np.broadcast_to(med[j0:j0 + 50000], blk.shape)[bad]
            blk -= mu[j0:j0 + 50000]; blk /= sd[j0:j0 + 50000]
        return Z

    def missing_fraction(self, X):
        return float(np.mean(~np.isfinite(X[:, self.keep])))


class AgeModel:
    """LASSO; alpha выбирается 5-кратной CV внутри train (одинаково для всех представлений)."""

    def __init__(self, seed=0, n_alphas=40, cv=5):
        self.seed, self.n_alphas, self.cv = seed, n_alphas, cv

    def fit(self, X, y):
        self.pre = Preprocessor().fit(X)
        Z = self.pre.transform(X)
        self.ymu = y.mean()
        self.m = LassoCV(alphas=self.n_alphas, cv=KFold(self.cv, shuffle=True, random_state=self.seed),
                         max_iter=20000, tol=1e-4, n_jobs=1, copy_X=False, random_state=self.seed).fit(Z, y)
        self.n_features = int(self.pre.keep.sum())
        self.n_selected = int((self.m.coef_ != 0).sum())
        return self

    def predict(self, X):
        return self.m.predict(self.pre.transform(X))

    def feature_missing(self, X):
        """Доля пропусков среди признаков с ненулевым коэффициентом."""
        idx = np.where(self.pre.keep)[0][self.m.coef_ != 0]
        return float(np.mean(~np.isfinite(X[:, idx]))) if len(idx) else np.nan


def metrics(y, p, n_boot=2000, seed=0):
    y = np.asarray(y, float); p = np.asarray(p, float)
    err = np.abs(p - y)
    rng = np.random.default_rng(seed)
    bs = [err[rng.integers(0, len(err), len(err))].mean() for _ in range(n_boot)]
    ss = ((y - y.mean()) ** 2).sum()
    return {"n": len(y), "MAE": err.mean(), "MAE_lo": np.percentile(bs, 2.5), "MAE_hi": np.percentile(bs, 97.5),
            "MedAE": np.median(err), "R2": 1 - ((y - p) ** 2).sum() / ss, "r": np.corrcoef(y, p)[0, 1],
            "bias": (p - y).mean()}


def paired_mae_diff(y, p1, p2, n_boot=2000, seed=0):
    """Бутстреп разности MAE(p1) - MAE(p2) на одних и тех же образцах."""
    e1 = np.abs(p1 - y); e2 = np.abs(p2 - y); rng = np.random.default_rng(seed)
    d = [(e1[i] - e2[i]).mean() for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return float((e1 - e2).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def age_acceleration(pred_ctrl, age_ctrl, pred_case, age_case):
    """Остатки относительно линии pred ~ age, оценённой на контролях вне обучения."""
    b = np.polyfit(age_ctrl, pred_ctrl, 1)
    rc = pred_ctrl - np.polyval(b, age_ctrl); rk = pred_case - np.polyval(b, age_case)
    u = stats.mannwhitneyu(rk, rc)
    return {"median_resid_case": float(np.median(rk)), "median_resid_ctrl": float(np.median(rc)),
            "MWU_p": float(u.pvalue), "n_case": len(rk), "n_ctrl": len(rc)}


# ----------------------------------------------------------------------------- графовый штраф
def region_bins(reg, res=25000):
    return ((reg.start.values + reg.end.values) // 2 // res) * res


def contact_edges(reg, contacts, res=25000, nodes=None):
    """Рёбра между регионами, чьи 25-кб бины имеют контакт (O/E >= порога при извлечении).
    contacts: chr(int), start1, start2, oe. Вес ребра = log2(O/E)."""
    nodes = np.arange(len(reg)) if nodes is None else np.asarray(nodes)
    nd = pd.DataFrame({"node": np.arange(len(nodes)), "chr": reg.chr.values[nodes], "bin": region_bins(reg, res)[nodes]})
    e = contacts.merge(nd.rename(columns={"bin": "start1", "node": "a"}), on=["chr", "start1"]) \
                .merge(nd.rename(columns={"bin": "start2", "node": "b"}), on=["chr", "start2"])
    e = e[e.a != e.b]
    return pd.DataFrame({"a": e.a.values, "b": e.b.values, "w": np.log2(e.oe.values).astype(np.float32),
                         "chr": e.chr.values})


def linear_edges(reg, nodes, max_gap=100000):
    r = reg.iloc[nodes].reset_index(drop=True)
    o = np.lexsort((r.start.values, r.chr.values))
    a, b = o[:-1], o[1:]
    ok = (r.chr.values[a] == r.chr.values[b]) & (r.start.values[b] - r.end.values[a] <= max_gap)
    return pd.DataFrame({"a": a[ok], "b": b[ok], "w": np.ones(ok.sum(), np.float32), "chr": r.chr.values[a[ok]]})


def rewire_edges(edges, seed=0):
    """Рандомизация с сохранением степеней узлов: перестановка вторых концов рёбер внутри хромосомы."""
    rng = np.random.default_rng(seed)
    out = []
    for ch, e in edges.groupby("chr"):
        b = rng.permutation(e.b.values)
        out.append(pd.DataFrame({"a": e.a.values, "b": b, "w": e.w.values, "chr": ch}))
    e = pd.concat(out, ignore_index=True)
    e = e[e.a != e.b]
    return e


def incidence(edges, p):
    m = len(edges)
    if m == 0:
        return sp.csr_matrix((0, p))
    sw = np.sqrt(np.clip(edges.w.values, 0, None)).astype(np.float64)
    rows = np.repeat(np.arange(m), 2)
    cols = np.column_stack([edges.a.values, edges.b.values]).ravel()
    vals = np.column_stack([sw, -sw]).ravel()
    return sp.csr_matrix((vals, (rows, cols)), shape=(m, p))


def _graph_path(Z, yc, B, lam2, alphas):
    Xa = sp.vstack([sp.csr_matrix(Z), np.sqrt(lam2) * B]).tocsc() if lam2 > 0 else Z
    ya = np.concatenate([yc, np.zeros(B.shape[0])]) if lam2 > 0 else yc
    # lasso_path масштабирует квадратичную часть на 1/N; приводим alpha к масштабу числа реальных образцов
    N = Xa.shape[0]; scale = len(yc) / N
    _, coefs, _ = lasso_path(Xa, ya, alphas=alphas * scale, max_iter=5000, tol=1e-3)
    return coefs  # p × n_alphas


def graph_lasso_cv(Z, y, B, lam2_grid, n_alphas=30, cv=5, seed=0):
    """Выбор (lambda2, alpha) 5-кратной CV на train; Z — стандартизованные признаки train."""
    yc = y - y.mean()
    amax = np.abs(Z.T @ yc).max() / len(y)
    alphas = amax * np.logspace(0, -3, n_alphas)
    folds = list(KFold(cv, shuffle=True, random_state=seed).split(Z))
    res = []
    for lam2 in lam2_grid:
        mse = np.zeros(n_alphas)
        for tr, va in folds:
            mu = Z[tr].mean(0); ytr = y[tr]
            coefs = _graph_path(Z[tr] - mu, ytr - ytr.mean(), B, lam2, alphas)
            pred = (Z[va] - mu) @ coefs + ytr.mean()
            mse += ((pred - y[va][:, None]) ** 2).mean(0) / cv
        k = int(np.argmin(mse)); res.append((mse[k], lam2, alphas[k]))
    best = min(res, key=lambda t: t[0])
    mu = Z.mean(0)
    coef = _graph_path(Z - mu, yc, B, best[1], np.array([best[2]]))[:, 0]
    return {"coef": coef, "mu": mu, "ymu": y.mean(), "lam2": best[1], "alpha": best[2],
            "cv_mse": best[0], "grid": res}
