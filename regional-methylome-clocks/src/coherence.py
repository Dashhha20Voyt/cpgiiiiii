"""Пайплайн 2: регионы согласованного возрастного изменения, сдвиг и несогласованность региона,
пространственные модули и переносимость региональных оценок."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.sparse.csgraph import connected_components
import core

MIN_COV = core.MIN_COV


def mvalues(mc, cov, min_cov=MIN_COV):
    """M = log((mC+0.5)/(uC+0.5)) и дельта-метод дисперсии выборочной ошибки v = 1/(mC+0.5) + 1/(uC+0.5)."""
    mc = mc.astype(np.float32); cov = cov.astype(np.float32)
    uc = cov - mc
    Mv = np.log((mc + 0.5) / (uc + 0.5))
    v = 1.0 / (mc + 0.5) + 1.0 / (uc + 0.5)
    ok = cov >= min_cov
    Mv[~ok] = np.nan; v[~ok] = np.nan
    return Mv, v


# ----------------------------------------------------------------------------- 1. возрастные эффекты CpG
def cpg_age_effects(M, rows, age, sites=None, chunk=200000, min_obs=10):
    """МНК-наклон M-value по возрасту для каждого CpG по обучающим образцам (только наблюдаемые значения)."""
    sites = np.arange(len(M.keys)) if sites is None else sites
    b = np.full(len(sites), np.nan, np.float32); se = np.full(len(sites), np.nan, np.float32)
    a = age.astype(np.float64)
    for s0 in range(0, len(sites), chunk):
        s = sites[s0:s0 + chunk]
        Mv, _ = mvalues(M.mc[rows][:, s], M.cov[rows][:, s])
        O = np.isfinite(Mv); X = np.nan_to_num(Mv).astype(np.float64)
        n = O.sum(0); A1 = (O * a[:, None]).sum(0); A2 = (O * a[:, None] ** 2).sum(0)
        S1 = X.sum(0); SA = (X * a[:, None]).sum(0); S2 = (X ** 2).sum(0)
        sxx = A2 - A1 ** 2 / np.maximum(n, 1); sxy = SA - A1 * S1 / np.maximum(n, 1); syy = S2 - S1 ** 2 / np.maximum(n, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            slope = sxy / sxx
            rss = np.clip(syy - slope * sxy, 0, None)
            sev = np.sqrt(rss / np.maximum(n - 2, 1) / sxx)
        ok = (n >= min_obs) & (sxx > 0)
        b[s0:s0 + chunk] = np.where(ok, slope, np.nan); se[s0:s0 + chunk] = np.where(ok, sev, np.nan)
    return b, se


# ----------------------------------------------------------------------------- 2. сегментация по возрастным эффектам
def _optimal_partition(x, w, pen, min_size):
    """Точное разбиение последовательности на отрезки с постоянным средним (взвешенный МНК + штраф за отрезок)."""
    n = len(x)
    cw = np.concatenate([[0], np.cumsum(w)]); cwx = np.concatenate([[0], np.cumsum(w * x)])
    cwx2 = np.concatenate([[0], np.cumsum(w * x * x)])
    F = np.full(n + 1, np.inf); F[0] = -pen; last = np.zeros(n + 1, dtype=np.int64)
    for t in range(min_size, n + 1):
        s = np.arange(0, t - min_size + 1)
        s = s[np.isfinite(F[s])]
        if len(s) == 0:
            continue
        W = cw[t] - cw[s]; WX = cwx[t] - cwx[s]; WX2 = cwx2[t] - cwx2[s]
        cost = WX2 - WX ** 2 / np.maximum(W, 1e-12)
        tot = F[s] + cost + pen
        k = int(np.argmin(tot)); F[t] = tot[k]; last[t] = s[k]
    cuts = []; t = n
    while t > 0:
        cuts.append(t); t = last[t]
    return sorted(cuts)  # концы отрезков


def coherent_change_regions(M, container_member, b, se, pen_mult=2.0, min_size=3, min_w_se=1e-3):
    """Внутри каждого координатного кластера (контейнера) CpG делятся на отрезки с однородным наклоном
    по возрасту. Вес CpG — 1/se²; штраф за отрезок = pen_mult·log(n). CpG без оценки наклона
    сохраняются внутри отрезка, к которому относятся по координате."""
    sites = np.where(container_member >= 0)[0]
    order = np.lexsort((M.pos[sites], container_member[sites]))
    sites = sites[order]; cont = container_member[sites]
    labels = np.full(len(M.keys), -1, np.int64)
    bounds = np.flatnonzero(np.diff(cont)) + 1
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(sites)]])
    nxt = 0; n_split = 0
    for s0, e0 in zip(starts, ends):
        idx = sites[s0:e0]
        est = np.isfinite(b[idx]) & np.isfinite(se[idx]) & (se[idx] > 0)
        if est.sum() < 2 * min_size:
            labels[idx] = nxt; nxt += 1; continue
        xi = b[idx][est].astype(np.float64); wi = 1.0 / np.maximum(se[idx][est].astype(np.float64), min_w_se) ** 2
        cuts = _optimal_partition(xi, wi, pen_mult * np.log(len(xi)), min_size)
        pos_est = np.flatnonzero(est)
        seg_of_est = np.zeros(len(xi), np.int64)
        prev = 0
        for k, c in enumerate(cuts):
            seg_of_est[prev:c] = k; prev = c
        # отрезок для каждого CpG контейнера: по ближайшему предыдущему оценённому CpG
        j = np.searchsorted(pos_est, np.arange(len(idx)), side="right") - 1
        seg = seg_of_est[np.clip(j, 0, None)]
        labels[idx] = nxt + seg; nxt += len(cuts); n_split += (len(cuts) > 1)
    reg, member = core.regions_from_labels(M.chr, M.pos, labels[sites], sites)
    return reg, member, n_split


# ----------------------------------------------------------------------------- 3. сдвиг и несогласованность
class RegionState:
    """Измерительная модель региона: M_si = α_i + u_sr + e_si, e_si ~ (0, v_si + σ²_i).
    α_i и σ²_i оцениваются по train; для образца вычисляются сдвиг u_sr (взвешенный), его стандартная ошибка
    и несогласованность D_sr = [Q/(n_obs-1)] / ref_r - 1, где Q = Σ w (M - α - u)², ref_r — средняя Q/(n_obs-1)
    региона в train. D ≈ 0 — профиль согласован с обучающим референсом так же, как в среднем в train;
    D > 0 — избыточная несогласованность с учётом покрытия конкретного образца."""

    def __init__(self, member, sigma2_floor=0.01):
        self.member = member; self.sites, self.A = core._assign(member); self.floor = sigma2_floor

    def fit(self, M, rows, block=150000):
        """Оценка alpha_i и sigma2_i по train поблочно (блоки сайтов выровнены по границам регионов)."""
        msites = self.member[self.sites]
        assert np.all(np.diff(msites) >= 0), "регионы должны быть непрерывными по геномному порядку"
        self.alpha = np.zeros(len(self.sites), np.float32); self.sigma2 = np.zeros(len(self.sites), np.float32)
        s0 = 0
        while s0 < len(self.sites):
            s1 = min(len(self.sites), s0 + block)
            while s1 < len(self.sites) and msites[s1] == msites[s1 - 1]:
                s1 += 1
            idx = self.sites[s0:s1]; loc = msites[s0:s1] - msites[s0]
            nreg = loc.max() + 1
            A = sp.csr_matrix((np.ones(len(idx), np.float32), (np.arange(len(idx)), loc)), shape=(len(idx), nreg))
            Mv, v = mvalues(M.mc[rows][:, idx], M.cov[rows][:, idx])
            O = np.isfinite(Mv)
            with np.errstate(invalid="ignore"):
                al = np.nanmean(Mv, axis=0)
            al = np.where(np.isfinite(al), al, 0).astype(np.float32)
            R0 = np.nan_to_num(Mv - al)
            n = O.astype(np.float32) @ A; u0 = (R0 @ A) / np.maximum(n, 1)
            e = np.where(O, Mv - al - np.asarray(u0 @ A.T), np.nan)
            with np.errstate(invalid="ignore"):
                s2 = np.nanmean(e ** 2, axis=0) - np.nanmean(v, axis=0)
            self.alpha[s0:s1] = al
            self.sigma2[s0:s1] = np.clip(np.nan_to_num(s2, nan=self.floor), self.floor, None)
            del Mv, v, O, R0, e
            s0 = s1
        # калибровка: средняя Q/df каждого региона в train принимается за референс (D_train ≈ 0 в среднем)
        self.qref = None
        q = self.transform(M, rows)["disc"] + 1
        with np.errstate(invalid="ignore"):
            self.qref = np.nanmean(q, axis=0)
        self.qref = np.where(np.isfinite(self.qref) & (self.qref > 0), self.qref, np.nan).astype(np.float32)
        return self

    def transform(self, M, rows, mc=None, cov=None, chunk=20, site_mask=None):
        mc = M.mc if mc is None else mc; cov = M.cov if cov is None else cov
        U, SE, D, N = [], [], [], []
        for s0 in range(0, len(rows), chunk):
            r = rows[s0:s0 + chunk]
            Mv, v = mvalues(mc[r][:, self.sites], cov[r][:, self.sites])
            if site_mask is not None:
                keep = site_mask[s0:s0 + chunk]
                Mv[~keep] = np.nan; v[~keep] = np.nan
            O = np.isfinite(Mv)
            W = np.where(O, 1.0 / (v + self.sigma2), 0).astype(np.float32)
            Rr = np.nan_to_num(Mv - self.alpha)
            sw = W @ self.A; swr = (W * Rr) @ self.A; swr2 = (W * Rr ** 2) @ self.A; n = O.astype(np.float32) @ self.A
            with np.errstate(invalid="ignore", divide="ignore"):
                u = swr / sw; se = 1 / np.sqrt(sw)
                Q = swr2 - swr ** 2 / sw
                d = Q / (n - 1)
                d = d / self.qref - 1 if getattr(self, "qref", None) is not None else d - 1
            u[n < 1] = np.nan; se[n < 1] = np.nan; d[n < 3] = np.nan
            U.append(u); SE.append(se); D.append(d); N.append(n)
        return {k: np.vstack(x).astype(np.float32) for k, x in zip(["shift", "se", "disc", "n"], [U, SE, D, N])}


# ----------------------------------------------------------------------------- 4. ассоциации несогласованности
def region_assoc(Dm, U, age, logcov, min_n=15):
    """Для каждого региона: t-статистика возраста в регрессии D ~ 1 + age + u + log(покрытие)."""
    nreg = Dm.shape[1]; t = np.full(nreg, np.nan); p = np.full(nreg, np.nan)
    for j in range(nreg):
        ok = np.isfinite(Dm[:, j]) & np.isfinite(U[:, j])
        if ok.sum() < min_n:
            continue
        X = np.column_stack([np.ones(ok.sum()), age[ok], U[ok, j], logcov[ok]])
        yv = Dm[ok, j]
        try:
            beta, res, rank, _ = np.linalg.lstsq(X, yv, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if rank < 4:
            continue
        r = yv - X @ beta; dof = ok.sum() - 4
        s2 = (r @ r) / dof
        cov_b = s2 * np.linalg.inv(X.T @ X)
        t[j] = beta[1] / np.sqrt(cov_b[1, 1]); p[j] = 2 * stats.t.sf(abs(t[j]), dof)
    return t, p


def bh(p):
    p = np.asarray(p); q = np.full(len(p), np.nan); ok = np.isfinite(p)
    pv = p[ok]; o = np.argsort(pv); r = np.arange(1, len(pv) + 1)
    qq = np.minimum.accumulate((pv[o] * len(pv) / r)[::-1])[::-1]
    out = np.empty(len(pv)); out[o] = np.minimum(qq, 1); q[ok] = out
    return q


# ----------------------------------------------------------------------------- 5. пространственные модули
def residualize_age(X, age):
    """Остатки каждого признака после регрессии на 1, age, age² (внутри когорты); пропуски — медиана."""
    Xi = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
    Xi = np.where(np.isfinite(Xi), Xi, 0)
    A = np.column_stack([np.ones(len(age)), age, age ** 2])
    beta, *_ = np.linalg.lstsq(A, Xi, rcond=None)
    Rz = Xi - A @ beta
    sd = Rz.std(0); sd[sd == 0] = 1
    return (Rz - Rz.mean(0)) / sd


def edge_corr(Z, edges):
    return (Z[:, edges.a.values] * Z[:, edges.b.values]).mean(0)


def distance_matched_null(reg, edges, nodes, res=25000, seed=0):
    """Для каждого контактного ребра — случайная пара регионов той же хромосомы на том же расстоянии в бинах."""
    rng = np.random.default_rng(seed)
    r = reg.iloc[nodes].reset_index(drop=True)
    binv = core.region_bins(r, res) // res
    key = r.chr.values.astype(np.int64) * 10**6 + binv
    by_key = pd.Series(np.arange(len(r))).groupby(key).apply(np.array).to_dict()
    da = binv[edges.b.values] - binv[edges.a.values]
    out_a, out_b = [], []
    chr_nodes = pd.Series(np.arange(len(r))).groupby(r.chr.values).apply(np.array).to_dict()
    for ch, d in zip(r.chr.values[edges.a.values], da):
        for _ in range(20):
            a = rng.choice(chr_nodes[ch]); kb = ch * 10**6 + binv[a] + d
            if kb in by_key:
                out_a.append(a); out_b.append(rng.choice(by_key[kb])); break
    e = pd.DataFrame({"a": out_a, "b": out_b}); e = e[e.a != e.b]
    e["w"] = 1.0; e["chr"] = r.chr.values[e.a.values]
    return e


def build_modules(edges_sel, n_nodes, min_size=2):
    if len(edges_sel) == 0:
        return np.full(n_nodes, -1)
    G = sp.coo_matrix((np.ones(len(edges_sel)), (edges_sel.a.values, edges_sel.b.values)), shape=(n_nodes, n_nodes))
    _, lab = connected_components(G, directed=False)
    size = np.bincount(lab)
    lab = np.where(size[lab] >= min_size, lab, -1)
    u, inv = np.unique(lab[lab >= 0], return_inverse=True)
    out = np.full(n_nodes, -1); out[lab >= 0] = inv
    return out


class ModuleFeatures:
    """Признак модуля = среднее стандартизованных (по train) сдвигов регионов-членов с выравниванием знака
    по нагрузкам первой главной компоненты в train."""

    def fit(self, U_train, module_of_node):
        self.mod = module_of_node
        self.mu = np.nanmean(U_train, 0); self.sd = np.nanstd(U_train, 0); self.sd[~(self.sd > 0)] = 1
        Z = np.nan_to_num((U_train - self.mu) / self.sd)
        self.sign = np.ones(U_train.shape[1])
        for m in range(self.mod.max() + 1):
            idx = np.where(self.mod == m)[0]
            _, _, vt = np.linalg.svd(Z[:, idx] - Z[:, idx].mean(0), full_matrices=False)
            s = np.sign(vt[0]); s[s == 0] = 1
            self.sign[idx] = s * (np.sign(s.sum()) or 1)
        return self

    def transform(self, U):
        Z = (U - self.mu) / self.sd * self.sign
        out = np.full((U.shape[0], self.mod.max() + 1), np.nan, np.float32)
        for m in range(self.mod.max() + 1):
            idx = np.where(self.mod == m)[0]
            with np.errstate(invalid="ignore"):
                out[:, m] = np.nanmean(Z[:, idx], 1)
        return out


# ----------------------------------------------------------------------------- 6. переносимость
def split_half_masks(M, rows, sites, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.random((len(rows), len(sites))) < 0.5
    return h, ~h


def reliability(XA, XB, min_samples=15):
    """Корреляция половин по образцам для каждого региона; медиана по регионам."""
    ok = np.isfinite(XA) & np.isfinite(XB)
    r = np.full(XA.shape[1], np.nan)
    for j in range(XA.shape[1]):
        o = ok[:, j]
        if o.sum() >= min_samples and XA[o, j].std() > 0 and XB[o, j].std() > 0:
            r[j] = np.corrcoef(XA[o, j], XB[o, j])[0, 1]
    return r
