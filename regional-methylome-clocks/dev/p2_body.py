# %% [markdown]
# ## 1. Конфигурация
# Все пути задаются здесь. По умолчанию в Colab используется `/content/regclock_work`; чтобы не скачивать
# данные повторно, можно указать каталог на Google Drive (ячейка ниже монтирует Drive при `USE_DRIVE=True`).

# %%
import os, sys, json, time, subprocess, shutil
IN_COLAB = "google.colab" in sys.modules
USE_DRIVE = False                      # True — хранить данные и кэш на Google Drive
if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    WORK = "/content/drive/MyDrive/regclock_work"
else:
    WORK = os.environ.get("REGCLOCK_WORK", "/content/regclock_work" if IN_COLAB else os.path.abspath("regclock_work"))
DATA = os.environ.get("REGCLOCK_DATA", os.path.join(WORK, "data"))
OUT = os.path.join(WORK, "results_pipeline2")
CODE = os.path.join(WORK, "code")
for d in (WORK, DATA, OUT, CODE):
    os.makedirs(d, exist_ok=True)
sys.path.insert(0, CODE)

SEED = 20261014          # фиксированное зерно для разбиения и всех случайных процедур
MIN_COV = 5              # минимальное покрытие CpG (как в RegBC)
MIN_FRAC_TRAIN = 0.5     # словарь CpG: покрытие >= MIN_COV у >= 50% обучающих образцов
QC_FRAC_CPG = 0.9        # ветка A: отдельные CpG с покрытием >= MIN_COV у >= 90% обучающих образцов
DBSCAN_EPS, DBSCAN_MINPTS = 1000, 5   # параметры eps1000 DRegBC (Simpson et al., 2023)
RUN_HIC = True           # загрузка Hi-C из 4DN (CH12 ~4.5 ГБ, mESC ~1.5 ГБ; после извлечения удаляются)
print("WORK =", WORK)

# %% [markdown]
# ## 2. Установка зависимостей
# Python: `cooler` для чтения `.mcool` Hi-C. Пайплайн 2 не требует R.

# %%
def sh(cmd, check=True):
    print("$", cmd)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if r.stdout.strip():
        print(r.stdout[-3000:])
    if r.returncode != 0:
        print(r.stderr[-3000:])
        if check:
            raise RuntimeError(f"Команда завершилась с кодом {r.returncode}")
    return r

sh(f"{sys.executable} -m pip install -q cooler pyarrow scikit-learn scipy pandas numpy matplotlib")


# %% [markdown]
# ## 3. Код модулей
# Ячейки ниже записывают четыре модуля в `WORK/code`: сборка когорт (`data_build.py`), регионы/признаки/модели
# (`core.py`), извлечение контактов Hi-C (`hic_extract.py`) и новые методы пайплайна 2 (`coherence.py`). Код полностью приведён в ноутбуке.

# %% WRITEFILE data_build.py
# %% WRITEFILE core.py
# %% WRITEFILE hic_extract.py
# %% WRITEFILE coherence.py

# %% [markdown]
# ## 4. Реальные данные: метаданные, загрузка, разбиение
# * **Обучение и внутренний тест** — контрольные мыши C57BL/6 дикого типа на стандартной диете из
#   GSE80672 (Petkovich et al., 2017), mm10-переобработка GSE120136. Разбиение 80/20 стратифицировано по
#   децилям возраста; один образец соответствует одному животному (повторных измерений в GEO не указано).
# * **CR** — мыши C57BL/6 на ограничении калорий из того же GSE80672 (только для оценки ускорения возраста).
# * **Внешний тест** — кровь C57BL/6J из GSE120132 (Thompson et al., 2018); дополнительно BALB/cByJ
#   (другая линия) как более жёсткий сдвиг.
# Словарь CpG строится только по покрытию обучающих образцов, без возраста.

# %%
import numpy as np, pandas as pd
from scipy import stats
import importlib, data_build as db, core, coherence as coh
importlib.reload(db); importlib.reload(core); importlib.reload(coh)

meta_p = db.petkovich_metadata(WORK)
meta_t = db.thompson_metadata(WORK)
paths_p = db.download_petkovich(os.path.join(DATA, "petkovich"), set(meta_p.sample_id))
paths_t = db.download_thompson(os.path.join(DATA, "thompson"), set(meta_t.sample_id))
missing = sorted(set(meta_p.sample_id) - set(paths_p))
print(f"Petkovich: {len(meta_p)} в метаданных, {len(paths_p)} файлов в GSE120136; нет файла у {len(missing)}: {missing}")
meta_p = meta_p[meta_p.sample_id.isin(paths_p)]
meta = pd.concat([meta_p, meta_t], ignore_index=True)
paths = {**paths_p, **paths_t}

META_TSV = os.path.join(WORK, "meta.tsv")
if os.path.exists(META_TSV):
    meta = pd.read_csv(META_TSV, sep="\t")
    print("Используется сохранённое разбиение", META_TSV)
else:
    rng = np.random.default_rng(SEED)
    ctrl = meta[meta.group == "petkovich_control"].copy()
    ctrl["stratum"] = pd.qcut(ctrl.age_months.rank(method="first"), 10, labels=False)
    test = []
    for _, g in ctrl.groupby("stratum"):
        test += list(rng.choice(g.sample_id.values, int(round(0.2 * len(g))), replace=False))
    meta["split"] = np.select(
        [meta.sample_id.isin(test), meta.group == "petkovich_control", meta.group == "petkovich_CR",
         meta.group == "thompson_B6"], ["internal_test", "train", "CR", "external_B6"], "external_BALB")
    meta.to_csv(META_TSV, sep="\t", index=False)
print(meta.groupby("split").agg(n=("sample_id", "size"), age_min=("age_months", "min"),
                                age_median=("age_months", "median"), age_max=("age_months", "max"),
                                male=("sex", lambda s: (s == "Male").sum())))

train_ids = meta.sample_id[meta.split == "train"].tolist()
keys, MC, CV, ids = db.build_matrix(meta.sample_id.tolist(), paths, train_ids, os.path.join(WORK, "matrix.npz"),
                                    min_cov=MIN_COV, min_frac_train=MIN_FRAC_TRAIN)
M = core.Methylome(keys, MC, CV, ids, meta)
y = M.meta.age_months.values
SPLITS = ["train", "internal_test", "external_B6", "external_BALB", "CR"]
R = {s: M.rows(s) for s in SPLITS}
print("Матрица:", MC.shape, "CpG в словаре:", len(keys))
obs = pd.DataFrame({s: [(CV[R[s]] >= MIN_COV).mean()] for s in SPLITS}, index=["доля наблюдаемых CpG словаря"])
print(obs.round(3))

# %% [markdown]
# ## 5. Внешние аннотации: состояния хроматина и контакты Hi-C
# * **ChromHMM** — универсальная (pan-tissue) 100-состоянийная full-stack аннотация mm10
#   (Vu & Ernst, 2023; github.com/ernstlab/mouse_fullStack_annotations). Соседние сегменты одной группы
#   состояний (Prom, TSS, enhancers, ReprPC, HET, quiescent и т.д.) объединяются в один интервал.
#   Аннотация не специфична для крови — это ограничение интерпретации.
# * **Hi-C** — in situ Hi-C клеток CH12 (B-лимфома мыши, ближайший к крови доступный mm10-референс 4DN,
#   4DNFIA7GQD4X) и mESC E14 (4DNFIMN7XO4J) как контроль «другой тип клеток». Из `.mcool` на разрешении
#   25 кб извлекаются внутрихромосомные пары бинов на расстоянии 50 кб–2 Мб с балансированным
#   O/E >= 2 и сырым числом контактов >= 5; исходный `.mcool` после извлечения удаляется.

# %%
import hic_extract
CI = {f"chr{i+1}": i for i in range(19)}
CHM_DIR = os.path.join(DATA, "chromhmm"); os.makedirs(CHM_DIR, exist_ok=True)
CHM_BASE = "https://public.hoffman2.idre.ucla.edu/ernst/2K9RS//full_stack/full_stack_annotation_public_release/mm10/"
for f in ["mm10_100_segments_segments.bed.gz", "state_annotation_processed.tsv"]:
    db._get(CHM_BASE + f, os.path.join(CHM_DIR, f))
st = pd.read_csv(os.path.join(CHM_DIR, "state_annotation_processed.tsv"), sep="\t")
grp = dict(zip(st.mneumonics, st.group))
chm = pd.read_csv(os.path.join(CHM_DIR, "mm10_100_segments_segments.bed.gz"), sep="\t", header=None,
                  names=["chr", "start", "end", "state"])
chm = chm[chm.chr.isin(CI)].copy()
chm["grp"] = chm.state.str.split("_", n=1).str[1].map(grp)
assert chm.grp.notna().all()
chm["chr"] = chm.chr.map(CI); chm = chm.sort_values(["chr", "start"])
brk = (chm.grp != chm.grp.shift()) | (chm.chr != chm.chr.shift()) | (chm.start != chm.end.shift())
chm["blk"] = brk.cumsum()
chm_blocks = chm.groupby("blk").agg(chr=("chr", "first"), start=("start", "min"), end=("end", "max"),
                                    grp=("grp", "first")).reset_index(drop=True)
print("Интервалов ChromHMM после объединения групп:", len(chm_blocks))

HIC_DIR = os.path.join(DATA, "hic"); os.makedirs(HIC_DIR, exist_ok=True)
HIC_SRC = {"CH12": "https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/78844e4c-c98a-4692-acf2-6c4347199685/4DNFIA7GQD4X.mcool",
           "mESC": "https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/c6181de2-2df3-4f17-a84a-3304c4c66511/4DNFIMN7XO4J.mcool"}
contacts = {}
if RUN_HIC:
    for name, url in HIC_SRC.items():
        pq = os.path.join(HIC_DIR, f"{name}_25kb_contacts.parquet")
        if not os.path.exists(pq):
            mc_path = os.path.join(HIC_DIR, f"{name}.mcool")
            sh(f"curl -sSfL --retry 5 -o {mc_path} {url}")
            hic_extract.extract_contacts(mc_path).to_parquet(pq)
            os.remove(mc_path)
        h = pd.read_parquet(pq); h["chr"] = h.chr.map(CI)
        contacts[name] = h
        print(name, "пар бинов с O/E>=2:", len(h))


# %% [markdown]
# ## 6. Регионы согласованного изменения (M1)
# Шаг 1. По обучающим образцам для каждого CpG оценивается наклон M-value по возрасту \(b_i\) и его
# стандартная ошибка (только наблюдаемые значения с покрытием >= 5).
# Шаг 2. Координатный каркас — DBSCAN (eps = 1000, minPts = 5). Внутри каждого кластера CpG делятся на
# отрезки с однородным возрастным наклоном: точное разбиение (динамическое программирование), взвешенный
# МНК с весами \(1/se_i^2\), штраф \(2\log n\) за отрезок, минимум 3 CpG в отрезке.
# Возраст используется только из train; внутренний тест, CR и внешние когорты в построении не участвуют.
# Примечание: alpha LASSO затем выбирается CV на тех же train-образцах, поэтому CV-ошибка внутри train
# оптимистична; итоговые оценки — только на отложенных когортах.

# %%
ALL = np.arange(len(M.ids))
allsites = np.arange(len(M.keys))
reg_db, mem_db = core.dbscan_regions(M.chr, M.pos, allsites, eps=DBSCAN_EPS, min_samples=DBSCAN_MINPTS)
t0 = time.time()
b_age, se_age = coh.cpg_age_effects(M, R["train"], y[R["train"]])
print(f"Возрастные наклоны: {np.isfinite(b_age).sum()} CpG, {time.time()-t0:.0f} с")
reg_dyn, mem_dyn, n_split = coh.coherent_change_regions(M, mem_db, b_age, se_age, pen_mult=2.0, min_size=3)
print(f"DBSCAN-кластеров: {len(reg_db)}; разделено на подрегионы: {n_split}; динамических регионов: {len(reg_dyn)}")

def within_between(member, b, se):
    """Однородность наклонов: взвешенная внутрирегиональная дисперсия b относительно ожидаемой (≈1 при однородности)."""
    s = np.where((member >= 0) & np.isfinite(b) & (se > 0))[0]
    w = 1 / se[s] ** 2; m = member[s]
    wb = pd.DataFrame({"m": m, "w": w, "wb": w * b[s], "wb2": w * b[s] ** 2, "one": 1}).groupby("m").sum()
    q = wb.wb2 - wb.wb ** 2 / wb.w
    dfree = wb.one - 1
    return float(q[dfree > 0].sum() / dfree[dfree > 0].sum())
homog = pd.DataFrame([{"разбиение": "DBSCAN", "регионов": len(reg_db), "Q/df наклонов": within_between(mem_db, b_age, se_age)},
                      {"разбиение": "Coherent (DBSCAN + наклоны)", "регионов": len(reg_dyn), "Q/df наклонов": within_between(mem_dyn, b_age, se_age)}])
print(homog.round(3).to_string(index=False))
homog.to_csv(os.path.join(OUT, "M1_partition_homogeneity.tsv"), sep="\t", index=False)

# %% [markdown]
# Проверка, что сегментация по train выделяет воспроизводимую структуру: наклоны CpG в независимой
# внешней когорте (Thompson C57BL/6J) — согласуются ли они внутри динамических регионов сильнее, чем
# внутри исходных DBSCAN-кластеров? Используется та же статистика однородности, рассчитанная по внешним
# наклонам (возраст внешней когорты используется только здесь, для валидации, не для построения).

# %%
b_ext, se_ext = coh.cpg_age_effects(M, R["external_B6"], y[R["external_B6"]], min_obs=15)
homog_ext = pd.DataFrame([{"разбиение": "DBSCAN", "Q/df наклонов (внешн.)": within_between(mem_db, b_ext, se_ext)},
                          {"разбиение": "Coherent", "Q/df наклонов (внешн.)": within_between(mem_dyn, b_ext, se_ext)}])
ok = np.isfinite(b_age) & np.isfinite(b_ext)
print(homog_ext.round(3).to_string(index=False))
print("Корреляция наклонов CpG train vs внешняя когорта:", round(np.corrcoef(b_age[ok], b_ext[ok])[0, 1], 3))
homog_ext.to_csv(os.path.join(OUT, "M1_partition_homogeneity_external.tsv"), sep="\t", index=False)

# %% [markdown]
# ## 7. Измерительная модель региона: сдвиг и несогласованность (M2)
# \[ M_{si}=\alpha_i+u_{sr}+e_{si},\qquad \operatorname{Var}(e_{si})=v_{si}+\sigma_i^2, \]
# где \(v_{si}=1/(m+0.5)+1/(u+0.5)\) — выборочная дисперсия M-value при данном покрытии, \(\alpha_i\) — уровень
# CpG в train, \(\sigma_i^2\) — его собственная (не объяснённая сдвигом региона) дисперсия в train.
# Для каждого образца и региона: сдвиг \(\hat u_{sr}\) (взвешенное среднее отклонений от \(\alpha_i\)), его
# стандартная ошибка и несогласованность \(D_{sr}=\dfrac{Q_{sr}/(n_{sr}-1)}{\overline{Q/(n-1)}_{r,\text{train}}}-1\), где
# \(Q_{sr}=\sum_i w_{si}(M_{si}-\alpha_i-\hat u_{sr})^2\), \(w_{si}=1/(v_{si}+\sigma_i^2)\). Нормировка на средний уровень региона в train
# делает \(D\approx0\) типичным для обучающей когорты; \(D>0\) — профиль менее согласован, чем в среднем в train,
# с учётом покрытия конкретного образца. В отличие от среднего β, сдвиг не зависит от того, какие именно
# CpG региона измерены, если модель верна.

# %%
STATE = {}
for name, mem in [("DBSCAN", mem_db), ("Coherent", mem_dyn)]:
    t0 = time.time()
    st_ = coh.RegionState(mem).fit(M, R["train"])
    out = st_.transform(M, ALL)
    out["mean"] = core.region_features(M, ALL, mem, kinds=("mean",))["mean"]
    STATE[name] = (st_, out)
    print(f"{name}: регионов {mem.max()+1}, {time.time()-t0:.0f} с; медиана D в train {np.nanmedian(out['disc'][R['train']]):.3f}, "
          f"во внешней {np.nanmedian(out['disc'][R['external_B6']]):.3f}")

# %% [markdown]
# ### Предсказание возраста: разбиение × представление
# Одна и та же модель (LASSO, alpha по 5-кратной CV на train). Представления: `mean` (среднее β),
# `shift` (сдвиг \(\hat u\)), `shift+disc` (сдвиг и несогласованность). Референс без регионализации — отдельные
# CpG с покрытием >= 5 у >= 90% обучающих образцов.

# %%
MODELS, PRED, ROWS = {}, {}, []
def evaluate(rep, X):
    t0 = time.time()
    m = core.AgeModel(seed=SEED).fit(X[R["train"]], y[R["train"]])
    MODELS[rep] = m
    for s in SPLITS[1:]:
        p = m.predict(X[R[s]]); PRED[(rep, s)] = p
        if s != "CR":
            ROWS.append({"representation": rep, "split": s, **core.metrics(y[R[s]], p, seed=SEED),
                         "n_features": m.n_features, "n_selected": m.n_selected,
                         "missing_in_selected": m.feature_missing(X[R[s]])})
    print(f"{rep:28s} признаков {m.n_features:>8d}, выбрано {m.n_selected:>4d}, {time.time()-t0:5.0f} с | " +
          " ".join(f"{r['split']}: MAE {r['MAE']:.2f}" for r in ROWS[-3:]), flush=True)

def rep_X(part, kind, out=None):
    out = STATE[part][1] if out is None else out
    if kind == "mean":
        return out["mean"]
    if kind == "shift":
        return out["shift"]
    if kind == "shift+disc":
        return np.hstack([out["shift"], out["disc"]])
    raise ValueError(kind)

qc_sites = np.where((CV[R["train"]] >= MIN_COV).mean(0) >= QC_FRAC_CPG)[0]
evaluate("CpG_all_QC", core.cpg_features(M, ALL, qc_sites))
for part in ["DBSCAN", "Coherent"]:
    for kind in ["mean", "shift", "shift+disc"]:
        evaluate(f"{part}_{kind}", rep_X(part, kind))
res = pd.DataFrame(ROWS)
res.to_csv(os.path.join(OUT, "M2_prediction_metrics.tsv"), sep="\t", index=False)
print(res.pivot_table(index="representation", columns="split", values=["MAE", "R2"]).round(2).to_string())

CONTRASTS = [("Coherent против DBSCAN (mean)", "Coherent_mean", "DBSCAN_mean"),
             ("Coherent против DBSCAN (shift)", "Coherent_shift", "DBSCAN_shift"),
             ("shift против mean (DBSCAN)", "DBSCAN_shift", "DBSCAN_mean"),
             ("shift против mean (Coherent)", "Coherent_shift", "Coherent_mean"),
             ("shift+disc против shift (DBSCAN)", "DBSCAN_shift+disc", "DBSCAN_shift"),
             ("shift+disc против shift (Coherent)", "Coherent_shift+disc", "Coherent_shift"),
             ("Coherent shift+disc против CpG", "Coherent_shift+disc", "CpG_all_QC"),
             ("DBSCAN mean против CpG", "DBSCAN_mean", "CpG_all_QC")]
crow = []
for name, a, b in CONTRASTS:
    for s in ["internal_test", "external_B6", "external_BALB"]:
        d, lo, hi = core.paired_mae_diff(y[R[s]], PRED[(a, s)], PRED[(b, s)], seed=SEED)
        crow.append({"сравнение": name, "split": s, "ΔMAE": d, "ДИ низ": lo, "ДИ верх": hi})
contr = pd.DataFrame(crow); contr.to_csv(os.path.join(OUT, "M2_paired_contrasts.tsv"), sep="\t", index=False)
print(contr.round(2).to_string(index=False))

# %% [markdown]
# ### Несёт ли несогласованность возрастную информацию сверх сдвига? (проверка по регионам)
# Для каждого региона с >= 15 образцами: \(D \sim 1 + \text{age} + \hat u + \log(\text{покрытие})\) отдельно в train и
# во внешней когорте. Регионы, значимые в train (FDR BH < 0.05), проверяются на воспроизводимость во внешней
# когорте: тот же знак и p < 0.05. Покрытие и сдвиг в модели контролируют технический и уровневый вклад.

# %%
ASSOC = {}
for part in ["DBSCAN", "Coherent"]:
    st_, out = STATE[part]
    logcov = np.log(np.maximum(CV.astype(np.float32).mean(1), 1))  # среднее покрытие образца
    rows_ = {}
    for s in ["train", "external_B6"]:
        r = R[s]
        good = np.where(np.nanmedian(out["n"][R["train"]], 0) >= 5)[0]  # регионы с >= 5 наблюдаемыми CpG
        t, p = coh.region_assoc(out["disc"][r][:, good], out["shift"][r][:, good], y[r], logcov[r])
        rows_[s] = (good, t, p)
    good, t_tr, p_tr = rows_["train"]; _, t_ex, p_ex = rows_["external_B6"]
    q_tr = coh.bh(p_tr)
    sig = np.isfinite(q_tr) & (q_tr < 0.05)
    both = np.isfinite(t_tr) & np.isfinite(t_ex)
    rep_ok = sig & both & (np.sign(t_tr) == np.sign(t_ex)) & (p_ex < 0.05)
    ASSOC[part] = pd.DataFrame({"region": good, "t_train": t_tr, "p_train": p_tr, "q_train": q_tr,
                                "t_external": t_ex, "p_external": p_ex})
    summ = {"разбиение": part, "регионов проверено": int(np.isfinite(t_tr).sum()),
            "значимы в train (FDR<0.05)": int(sig.sum()),
            "из них ↑ с возрастом": int((sig & (t_tr > 0)).sum()),
            "воспроизведены во внешней (тот же знак, p<0.05)": int(rep_ok.sum()),
            "доля совпадения знака среди значимых": float(np.mean(np.sign(t_tr[sig & both]) == np.sign(t_ex[sig & both]))) if (sig & both).any() else np.nan,
            "r(t_train, t_external) все регионы": float(np.corrcoef(t_tr[both], t_ex[both])[0, 1])}
    print(summ)
    pd.DataFrame([summ]).to_csv(os.path.join(OUT, f"M2_discordance_assoc_summary_{part}.tsv"), sep="\t", index=False)
    ASSOC[part].to_csv(os.path.join(OUT, f"M2_discordance_assoc_{part}.tsv.gz"), sep="\t", index=False)

# %% [markdown]
# ### Глобальный индекс несогласованности и состояния хроматина
# Индекс образца — среднее \(D\) по регионам, где в train несогласованность значимо растёт с возрастом
# (FDR < 0.05, t > 0). Возрастная зависимость индекса проверяется во внешней когорте; для CR — остатки
# относительно контролей внутреннего теста. Дополнительно: в каких группах состояний хроматина (full-stack
# ChromHMM) находятся регионы с возрастным ростом несогласованности (обогащение относительно всех проверенных).

# %%
part = "Coherent"
A_ = ASSOC[part]; st_, out = STATE[part]
up = A_.region.values[(A_.q_train < 0.05) & (A_.t_train > 0)]
glob_rows = []
if len(up):
    G = np.nanmean(out["disc"][:, up], axis=1)
    for s in ["internal_test", "external_B6", "external_BALB"]:
        r = R[s]; rr = stats.spearmanr(y[r], G[r])
        glob_rows.append({"split": s, "Spearman rho(age, индекс)": rr.correlation, "p": rr.pvalue, "n": len(r)})
    b = np.polyfit(y[R["internal_test"]], G[R["internal_test"]], 1)
    rc = G[R["internal_test"]] - np.polyval(b, y[R["internal_test"]]); rk = G[R["CR"]] - np.polyval(b, y[R["CR"]])
    glob_rows.append({"split": "CR против контролей (остатки)", "Spearman rho(age, индекс)": np.nan,
                      "p": stats.mannwhitneyu(rk, rc).pvalue, "n": len(rk),
                      "медиана остатка CR": np.median(rk), "медиана остатка контроль": np.median(rc)})
glob = pd.DataFrame(glob_rows); glob.to_csv(os.path.join(OUT, "M2_global_discordance_index.tsv"), sep="\t", index=False)
print(glob.round(4).to_string(index=False))

reg_tab = reg_dyn.copy()
mid = ((reg_tab.start + reg_tab.end) // 2).values - 1
reg_tab["chrom_group"] = None
for ch, iv in chm_blocks.groupby("chr"):
    w = np.where(reg_tab.chr.values == ch)[0]
    k = np.searchsorted(iv.start.values, mid[w], side="right") - 1
    okk = (k >= 0) & (mid[w] < iv.end.values[np.clip(k, 0, None)])
    reg_tab.loc[reg_tab.index[w[okk]], "chrom_group"] = iv.grp.values[k[okk]]
tested = A_.region.values[np.isfinite(A_.t_train.values)]
bg = reg_tab.chrom_group.iloc[tested].value_counts(normalize=True)
fg = reg_tab.chrom_group.iloc[up].value_counts(normalize=True) if len(up) else pd.Series(dtype=float)
enr = pd.DataFrame({"доля среди проверенных": bg, "доля среди D↑ с возрастом": fg}).fillna(0)
enr["обогащение"] = enr["доля среди D↑ с возрастом"] / enr["доля среди проверенных"]
enr = enr.sort_values("обогащение", ascending=False)
enr.to_csv(os.path.join(OUT, "M2_discordance_chromatin_enrichment.tsv"), sep="\t")
print(enr.round(3).to_string())

# %% [markdown]
# ## 8. Пространственные модули (M3)
# Узлы — регионы согласованного изменения (Coherent), признак — сдвиг \(\hat u\).
# (1) **Проверка на независимой когорте.** Остаточная ко-метилированность пары регионов — корреляция сдвигов
# после удаления возрастного тренда (1, age, age²) внутри когорты. Сравниваются пары с контактом Hi-C CH12 и
# случайные пары той же хромосомы на том же расстоянии в 25-кб бинах (distance-matched null), в train и
# отдельно во внешней когорте. Если контакт несёт информацию сверх линейной близости, контактные пары
# должны быть ко-метилированы сильнее в обеих когортах.
# (2) **Модули.** Рёбро сохраняется, если есть контакт и остаточная корреляция в train выше 95-го перцентиля
# нулевого распределения; модули — связные компоненты (>= 2 региона). Контроль — та же процедура на
# distance-matched случайных парах. Признак модуля — среднее стандартизованных сдвигов участников с
# выравниванием знака по PC1 в train.

# %%
from scipy import stats
M3 = {}
if RUN_HIC:
    st_, out = STATE["Coherent"]
    U = out["shift"]
    n_med = np.nanmedian(out["n"][R["train"]], 0)
    nodes = np.where((n_med >= 3) & (np.isfinite(U[R["train"]]).mean(0) >= 0.8))[0]
    print("Узлов (регионов с >= 3 CpG и наблюдением у >= 80% train):", len(nodes))
    Zc = {s: coh.residualize_age(U[R[s]][:, nodes], y[R[s]]) for s in ["train", "external_B6"]}
    E_ct = core.contact_edges(reg_dyn, contacts["CH12"], nodes=nodes)
    E_ct = E_ct[E_ct.a < E_ct.b].drop_duplicates(["a", "b"])
    E_null = coh.distance_matched_null(reg_dyn, E_ct, nodes, seed=SEED).drop_duplicates(["a", "b"])
    E_mesc = core.contact_edges(reg_dyn, contacts["mESC"], nodes=nodes); E_mesc = E_mesc[E_mesc.a < E_mesc.b].drop_duplicates(["a", "b"])
    co_rows = []
    for s in ["train", "external_B6"]:
        rc = coh.edge_corr(Zc[s], E_ct); rn = coh.edge_corr(Zc[s], E_null); rm = coh.edge_corr(Zc[s], E_mesc)
        M3[s] = (rc, rn, rm)
        co_rows.append({"когорта": s, "пар CH12": len(rc), "средняя r (CH12)": rc.mean(),
                        "пар mESC": len(rm), "средняя r (mESC)": rm.mean(),
                        "пар null": len(rn), "средняя r (null)": rn.mean(),
                        "p MWU CH12>null": stats.mannwhitneyu(rc, rn, alternative="greater").pvalue,
                        "p MWU mESC>null": stats.mannwhitneyu(rm, rn, alternative="greater").pvalue})
    rc_tr, rn_tr, _ = M3["train"]; rc_ex, rn_ex, _ = M3["external_B6"]
    co = pd.DataFrame(co_rows); co.to_csv(os.path.join(OUT, "M3_contact_comethylation.tsv"), sep="\t", index=False)
    print(co.round(4).to_string(index=False))
    print("Согласованность остаточных корреляций контактных пар train vs внешняя: r =",
          round(np.corrcoef(rc_tr, rc_ex)[0, 1], 3))
    # стратификация по силе контакта
    E_ct["r_train"] = rc_tr; E_ct["r_ext"] = rc_ex
    E_ct["oe_bin"] = pd.qcut(E_ct.w, 4, labels=["Q1 (слабее)", "Q2", "Q3", "Q4 (сильнее)"])
    strat = E_ct.groupby("oe_bin", observed=True)[["r_train", "r_ext"]].mean()
    strat["null_train"] = rn_tr.mean(); strat["null_ext"] = rn_ex.mean()
    print(strat.round(4).to_string())
    strat.to_csv(os.path.join(OUT, "M3_comethylation_by_contact_strength.tsv"), sep="\t")

    thr = np.percentile(rn_tr, 95)
    sel_ct = E_ct[rc_tr > thr]; sel_null = E_null[rn_tr > thr]
    mod_ct = coh.build_modules(sel_ct, len(nodes)); mod_null = coh.build_modules(sel_null, len(nodes))
    print(f"Порог r = {thr:.3f}; модулей по контактам: {mod_ct.max()+1} (регионов {int((mod_ct>=0).sum())}); "
          f"по случайным парам: {mod_null.max()+1} (регионов {int((mod_null>=0).sum())})")
    # доля пар модулей, остающихся выше порога во внешней когорте
    print("Доля отобранных рёбер с r > порога во внешней когорте: контакты",
          round(float(np.mean(rc_ex[rc_tr > thr] > thr)), 3), "| случайные",
          round(float(np.mean(rn_ex[rn_tr > thr] > thr)), 3))

    Utr = U[R["train"]][:, nodes]
    MF = {"modules_CH12": coh.ModuleFeatures().fit(Utr, mod_ct), "modules_null": coh.ModuleFeatures().fit(Utr, mod_null)}
    Xmod = {k: v.transform(U[:, nodes]) for k, v in MF.items()}
    Xreg = U  # все регионы Coherent (shift)
    assert Xmod["modules_CH12"].shape[1] > 0 and Xmod["modules_null"].shape[1] > 0, "Модули не найдены"
    evaluate("M3_modules_CH12_only", Xmod["modules_CH12"])
    evaluate("M3_modules_null_only", Xmod["modules_null"])
    evaluate("M3_regions+modules_CH12", np.hstack([Xreg, Xmod["modules_CH12"]]))
    evaluate("M3_regions+modules_null", np.hstack([Xreg, Xmod["modules_null"]]))
    res = pd.DataFrame(ROWS)
    m3 = res[res.representation.str.startswith("M3_") | (res.representation == "Coherent_shift")]
    print(m3.pivot_table(index="representation", columns="split", values=["MAE", "R2"]).round(2).to_string())
    crow3 = []
    for a, b in [("M3_modules_CH12_only", "M3_modules_null_only"), ("M3_regions+modules_CH12", "Coherent_shift"),
                 ("M3_regions+modules_CH12", "M3_regions+modules_null")]:
        for s in ["internal_test", "external_B6"]:
            d, lo, hi = core.paired_mae_diff(y[R[s]], PRED[(a, s)], PRED[(b, s)], seed=SEED)
            crow3.append({"первый": a, "второй": b, "split": s, "ΔMAE": d, "ДИ низ": lo, "ДИ верх": hi})
    c3 = pd.DataFrame(crow3); c3.to_csv(os.path.join(OUT, "M3_paired_contrasts.tsv"), sep="\t", index=False)
    print(c3.round(2).to_string(index=False))

# %% [markdown]
# ## 9. Переносимость оценки состояния региона (M4)
# (1) **Надёжность при смене набора CpG.** У каждого образца наблюдаемые CpG случайно делятся на две
# половины; региональная величина вычисляется по каждой половине отдельно. Для каждого региона считается
# корреляция половин по образцам (выше — меньше зависимость от того, какие CpG измерены). Сравниваются
# среднее β и сдвиг \(\hat u\) на одних и тех же Coherent-регионах.
# (2) **Калибровка неопределённости.** \(z=(\hat u_A-\hat u_B)/\sqrt{se_A^2+se_B^2}\): при корректных стандартных
# ошибках SD(z) ≈ 1.
# (3) **Предсказание при удалении CpG и прореживании чтений** во внешней когорте для mean / shift /
# shift+disc (модели не переобучаются).

# %%
st_, out = STATE["Coherent"]
rel_rows, rel_store = [], {}
for s in ["internal_test", "external_B6"]:
    r = R[s]
    hA, hB = coh.split_half_masks(M, r, st_.sites, seed=SEED)
    oA = st_.transform(M, r, site_mask=hA); oB = st_.transform(M, r, site_mask=hB)
    # среднее β по половине: покрытие CpG другой половины обнуляется
    cA = np.zeros_like(CV[r]); cB = np.zeros_like(CV[r])
    cA[:, st_.sites] = np.where(hA, CV[r][:, st_.sites], 0); cB[:, st_.sites] = np.where(hB, CV[r][:, st_.sites], 0)
    mA = core.region_features(M, np.arange(len(r)), mem_dyn, kinds=("mean",), mc=MC[r], cov=cA)["mean"]
    mB = core.region_features(M, np.arange(len(r)), mem_dyn, kinds=("mean",), mc=MC[r], cov=cB)["mean"]
    del cA, cB
    keep = np.where(np.nanmedian(out["n"][R["train"]], 0) >= 6)[0]
    rel_mean = coh.reliability(mA[:, keep], mB[:, keep]); rel_shift = coh.reliability(oA["shift"][:, keep], oB["shift"][:, keep])
    z = (oA["shift"][:, keep] - oB["shift"][:, keep]) / np.sqrt(oA["se"][:, keep] ** 2 + oB["se"][:, keep] ** 2)
    z = z[np.isfinite(z)]
    both = np.isfinite(rel_mean) & np.isfinite(rel_shift)
    rel_rows.append({"когорта": s, "регионов (>= 6 CpG)": int(both.sum()),
                     "медиана надёжности: mean β": float(np.median(rel_mean[both])),
                     "медиана надёжности: shift": float(np.median(rel_shift[both])),
                     "доля регионов shift > mean": float(np.mean(rel_shift[both] > rel_mean[both])),
                     "p Уилкоксона": float(stats.wilcoxon(rel_shift[both], rel_mean[both]).pvalue),
                     "SD(z) калибровка": float(np.std(z)), "доля |z|>1.96": float(np.mean(np.abs(z) > 1.96))})
    rel_store[s] = (rel_mean, rel_shift)
rel = pd.DataFrame(rel_rows); rel.to_csv(os.path.join(OUT, "M4_split_half_reliability.tsv"), sep="\t", index=False)
print(rel.round(3).to_string(index=False))

# %%
E6_REPS = [r for r in ["CpG_all_QC", "DBSCAN_mean", "Coherent_mean", "Coherent_shift", "Coherent_shift+disc"] if r in MODELS]
LEVELS = [("mask", 0.25), ("mask", 0.5), ("mask", 0.75), ("thin", 0.5), ("thin", 0.25), ("thin", 0.1)]
rows_ext = R["external_B6"]; local = np.arange(len(rows_ext))
E6 = [{"representation": rep, "mode": "none", "level": 0, "seed": 0,
       **core.metrics(y[rows_ext], PRED[(rep, "external_B6")], n_boot=200)} for rep in E6_REPS]
for mode, lev in LEVELS:
    for sd in range(2):
        mc_d, cv_d = core.degrade_counts(MC[rows_ext], CV[rows_ext], mode, lev, np.random.default_rng(SEED + sd))
        cache = {}
        for rep in E6_REPS:
            if rep == "CpG_all_QC":
                X = core.cpg_features(M, local, qc_sites, mc=mc_d, cov=cv_d)
            else:
                part, kind = rep.split("_", 1)
                if part not in cache:
                    o = STATE[part][0].transform(M, local, mc=mc_d, cov=cv_d)
                    o["mean"] = core.region_features(M, local, mem_db if part == "DBSCAN" else mem_dyn, kinds=("mean",), mc=mc_d, cov=cv_d)["mean"]
                    cache[part] = o
                X = rep_X(part, kind, cache[part])
            E6.append({"representation": rep, "mode": mode, "level": lev, "seed": sd,
                       **core.metrics(y[rows_ext], MODELS[rep].predict(X), n_boot=200)})
        print(mode, lev, sd, "готово", flush=True)
e6 = pd.DataFrame(E6); e6.to_csv(os.path.join(OUT, "M4_robustness.tsv"), sep="\t", index=False)
print(e6.groupby(["representation", "mode", "level"]).MAE.mean().round(2).unstack(["mode", "level"]).to_string())

# %% [markdown]
# ## 10. Ускорение возраста при CR и сохранение результатов

# %%
cr = pd.DataFrame([{"representation": rep, **core.age_acceleration(PRED[(rep, "internal_test")], y[R["internal_test"]],
                                                                    PRED[(rep, "CR")], y[R["CR"]])} for rep in MODELS])
cr.to_csv(os.path.join(OUT, "CR_age_acceleration.tsv"), sep="\t", index=False)
print(cr.round(3).to_string(index=False))

import matplotlib.pyplot as plt
res = pd.DataFrame(ROWS); res.to_csv(os.path.join(OUT, "all_prediction_metrics.tsv"), sep="\t", index=False)
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
order = list(MODELS)
for k, s in enumerate(["internal_test", "external_B6"]):
    d = res[res.split == s].set_index("representation").reindex(order)
    ax[k].barh(range(len(d)), d.MAE, xerr=[d.MAE - d.MAE_lo, d.MAE_hi - d.MAE], color="#55A868")
    ax[k].set_yticks(range(len(d))); ax[k].set_yticklabels(d.index, fontsize=8); ax[k].invert_yaxis()
    ax[k].set_xlabel("MAE, мес. (95% ДИ)"); ax[k].set_title({"internal_test": "Внутренний тест", "external_B6": "Внешний тест C57BL/6J"}[s])
rm, rs = rel_store["external_B6"]; okr = np.isfinite(rm) & np.isfinite(rs)
ax[2].hist(rs[okr] - rm[okr], bins=60, color="#8172B2"); ax[2].axvline(0, color="k", lw=1)
ax[2].set_xlabel("надёжность shift − надёжность mean β"); ax[2].set_title("Split-half, внешняя когорта")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_pipeline2_summary.png"), dpi=150); plt.show()

if RUN_HIC and "train" in M3:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, s in enumerate(["train", "external_B6"]):
        rc, rn, rm_ = M3[s]
        ax[k].hist(rn, bins=80, alpha=.6, density=True, label="случайные пары (то же расстояние)")
        ax[k].hist(rc, bins=80, alpha=.6, density=True, label="контакты CH12")
        ax[k].set_title({"train": "Train (Petkovich)", "external_B6": "Внешняя когорта (Thompson)"}[s])
        ax[k].set_xlabel("остаточная корреляция сдвигов"); ax[k].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_M3_contact_comethylation.png"), dpi=150); plt.show()

pd.DataFrame([{"representation": r, "sample_id": M.ids[i], "split": s, "age_months": y[i], "pred": p}
              for (r, s), pp in PRED.items() for i, p in zip(R[s], pp)]).to_csv(os.path.join(OUT, "predictions.tsv"), sep="\t", index=False)
for name, reg in [("DBSCAN", reg_db), ("Coherent", reg_dyn)]:
    o = reg.copy(); o["chr"] = "chr" + (o.chr + 1).astype(str); o["start"] = o.start - 1
    o[["chr", "start", "end", "n_cpg"]].to_csv(os.path.join(OUT, f"regions_{name}.bed"), sep="\t", header=False, index=False)
json.dump(dict(SEED=SEED, MIN_COV=MIN_COV, MIN_FRAC_TRAIN=MIN_FRAC_TRAIN, QC_FRAC_CPG=QC_FRAC_CPG, DBSCAN_EPS=DBSCAN_EPS,
               DBSCAN_MINPTS=DBSCAN_MINPTS, segmentation_penalty="2*log(n)", min_segment_cpg=3, RUN_HIC=RUN_HIC),
          open(os.path.join(OUT, "config.json"), "w"), indent=2)
meta.to_csv(os.path.join(OUT, "samples_and_split.tsv"), sep="\t", index=False)
shutil.make_archive(os.path.join(WORK, "results_pipeline2"), "zip", OUT)
print("Архив:", os.path.join(WORK, "results_pipeline2.zip"))
if IN_COLAB:
    from google.colab import files
    files.download(os.path.join(WORK, "results_pipeline2.zip"))
