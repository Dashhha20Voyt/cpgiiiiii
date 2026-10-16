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
OUT = os.path.join(WORK, "results_pipeline1")
CODE = os.path.join(WORK, "code")
for d in (WORK, DATA, OUT, CODE):
    os.makedirs(d, exist_ok=True)
sys.path.insert(0, CODE)

SEED = 20261014          # фиксированное зерно для разбиения и всех случайных процедур
MIN_COV = 5              # минимальное покрытие CpG (как в RegBC)
MIN_FRAC_TRAIN = 0.5     # словарь CpG: покрытие >= MIN_COV у >= 50% обучающих образцов
QC_FRAC_CPG = 0.9        # ветка A: отдельные CpG с покрытием >= MIN_COV у >= 90% обучающих образцов
DBSCAN_EPS, DBSCAN_MINPTS = 1000, 5   # параметры eps1000 DRegBC (Simpson et al., 2023)
RUN_METHYLASSO = True    # оригинальный MethyLasso (R) из github.com/bardetlab/methylasso
RUN_HIC = True           # загрузка Hi-C из 4DN (CH12 ~4.5 ГБ, mESC ~1.5 ГБ; после извлечения удаляются)
print("WORK =", WORK)

# %% [markdown]
# ## 2. Установка зависимостей
# Python: `cooler` для чтения `.mcool`. R: зависимости MethyLasso ставятся бинарными пакетами
# Posit Package Manager (быстро в Colab). В `src/Makevars` MethyLasso стандарт C++11 заменяется на C++17:
# новые заголовки Boost (пакет BH) не компилируются в режиме C++11. Алгоритм не изменяется.

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

METHYLASSO_DIR = os.path.join(WORK, "methylasso")
def install_methylasso():
    if shutil.which("Rscript") is None:
        sh("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq r-base-core")
    ok = sh("Rscript -e 'quit(status = !requireNamespace(\"MethyLasso\", quietly=TRUE))'", check=False)
    if ok.returncode == 0:
        print("MethyLasso уже установлен"); return
    sh("apt-get install -y -qq libgsl-dev > /dev/null 2>&1 || true", check=False)
    codename = subprocess.run("lsb_release -cs", shell=True, text=True, capture_output=True).stdout.strip() or "jammy"
    pkgs = 'c("Rcpp","RcppEigen","BH","RcppGSL","matrixStats","scales","Matrix","data.table","ggplot2","foreach","doParallel","stringr","R.utils","optparse")'
    sh(f"""Rscript -e 'options(repos=c(CRAN="https://packagemanager.posit.co/cran/__linux__/{codename}/latest"), HTTPUserAgent=sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version$platform, R.version$arch, R.version$os))); need <- setdiff({pkgs}, rownames(installed.packages())); if (length(need)) install.packages(need, Ncpus=2)'""")
    if not os.path.exists(METHYLASSO_DIR):
        sh(f"git clone -q https://github.com/bardetlab/methylasso.git {METHYLASSO_DIR}")
    sh(f"cd {METHYLASSO_DIR} && git rev-parse HEAD > COMMIT && sed -i 's/CXX_STD = CXX11/CXX_STD = CXX17/' src/Makevars")
    sh(f"R CMD INSTALL --preclean {METHYLASSO_DIR} > {WORK}/methylasso_install.log 2>&1 || (tail -30 {WORK}/methylasso_install.log; false)")

if RUN_METHYLASSO:
    install_methylasso()
    if not os.path.exists(os.path.join(METHYLASSO_DIR, "MethyLasso.R")):
        sh(f"git clone -q https://github.com/bardetlab/methylasso.git {METHYLASSO_DIR}")
    sh(f"Rscript {METHYLASSO_DIR}/MethyLasso.R --version")

# %% [markdown]
# ## 3. Код модулей
# Ячейки ниже записывают три модуля в `WORK/code`: сборка когорт (`data_build.py`), регионы/признаки/модели
# (`core.py`) и извлечение контактов Hi-C (`hic_extract.py`). Код полностью приведён в ноутбуке.

# %% WRITEFILE data_build.py
# %% WRITEFILE core.py
# %% WRITEFILE hic_extract.py

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
import importlib, data_build as db, core
importlib.reload(db); importlib.reload(core)

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
# ## 6. Оригинальный MethyLasso на объединённых обучающих counts
# MethyLasso запускается в режиме одного условия: counts всех обучающих образцов суммируются по CpG словаря
# (псевдобалк; возраст не используется). Вход — шесть колонок `chr, start, end, meth, mC, uC`, координаты
# 0-based: `start = pos-1`, `end = pos+1`. С `--drop FALSE` сохраняется полная сегментация fused lasso,
# включая неаннотированные сегменты; в эксперименте используются сегменты с >= 3 CpG словаря.
# MethyLasso разработан для полногеномных данных; применение к RRBS — проверяемое допущение.

# %%
ml_regions_table = None
if RUN_METHYLASSO:
    ml_dir = os.path.join(WORK, "methylasso_out"); os.makedirs(ml_dir, exist_ok=True)
    ml_tsv = os.path.join(ml_dir, "pooledtrain_lmr_umr_dmv.tsv")
    if not os.path.exists(ml_tsv):
        tr = R["train"]
        mc_sum = MC[tr].astype(np.int64).sum(0); cv_sum = CV[tr].astype(np.int64).sum(0)
        chn = np.array([f"chr{i+1}" for i in range(19)])[M.chr]
        pooled = pd.DataFrame({"chr": chn, "start": M.pos - 1, "end": M.pos + 1,
                               "meth": np.round(mc_sum / np.maximum(cv_sum, 1), 4), "mC": mc_sum, "uC": cv_sum - mc_sum})
        pooled_path = os.path.join(ml_dir, "pooled_train.tsv.gz")
        pooled[cv_sum > 0].to_csv(pooled_path, sep="\t", header=False, index=False)
        sh(f"cd {ml_dir} && Rscript {METHYLASSO_DIR}/MethyLasso.R --n1 pooledtrain --c1 {pooled_path} "
           f"-t 2 -f FALSE --drop FALSE > methylasso.log 2>&1")
    ml = pd.read_csv(ml_tsv, sep="\t")
    print("Категории MethyLasso:", ml.category.fillna("unannotated").value_counts().to_dict())
    ml = ml[ml.chr.isin(CI)].copy(); ml["chr"] = ml.chr.map(CI); ml["end"] = ml.end + 1
    ml_regions_table = ml.reset_index(drop=True)

# %% [markdown]
# ## 7. Разбиения генома
# Все разбиения строятся до обучения модели и не используют возраст:
# DBSCAN (eps = 1000 п.н., minPts = 5) по координатам CpG словаря; окна 5 кб (RegBC);
# сегменты MethyLasso; интервалы ChromHMM; пересечения DBSCAN ∩ MethyLasso и DBSCAN ∩ ChromHMM.

# %%
allsites = np.arange(len(M.keys))
train_obs = (CV[R["train"]] >= MIN_COV).mean(0)
qc_sites = np.where(train_obs >= QC_FRAC_CPG)[0]
PART = {}
PART["DBSCAN"] = core.dbscan_regions(M.chr, M.pos, allsites, eps=DBSCAN_EPS, min_samples=DBSCAN_MINPTS)
PART["Window5kb"] = core.window_regions(M.chr, M.pos, allsites, width=5000)
if ml_regions_table is not None:
    PART["MethyLasso"] = core.interval_regions(M.chr, M.pos, allsites, ml_regions_table[["chr", "start", "end"]], min_cpg=3)
    PART["DBSCAN∩MethyLasso"] = core.intersect_partitions(PART["DBSCAN"][1], PART["MethyLasso"][1], M.chr, M.pos)
PART["ChromHMM"] = core.interval_regions(M.chr, M.pos, allsites, chm_blocks[["chr", "start", "end"]], min_cpg=3)
PART["DBSCAN∩ChromHMM"] = core.intersect_partitions(PART["DBSCAN"][1], PART["ChromHMM"][1], M.chr, M.pos)
# Для ветки C1: DBSCAN-кластеры, ограниченные CpG ветки A (тот же набор сайтов, что в ветке B)
mem_db = PART["DBSCAN"][1]
mem_same = np.where(np.isin(allsites, qc_sites), mem_db, -1)
PART["DBSCAN_sameSites"] = (PART["DBSCAN"][0], mem_same)
part_summary = pd.DataFrame([{"разбиение": k, "регионов": len(v[0]), "CpG в регионах": int((v[1] >= 0).sum()),
                              "медиана CpG/регион": float(np.median(np.bincount(v[1][v[1] >= 0])[np.bincount(v[1][v[1] >= 0]) > 0])),
                              "медиана длины, п.н.": float(v[0].eval("end-start").median())} for k, v in PART.items()])
print(part_summary.to_string(index=False))
part_summary.to_csv(os.path.join(OUT, "partitions_summary.tsv"), sep="\t", index=False)

# %% [markdown]
# ## 8. Признаки и общая процедура оценки
# Для каждого образца признаки региона вычисляются по его собственным CpG с покрытием >= 5:
# `mean` — среднее β; `sd` — стандартное отклонение между CpG; `slope` — наклон β по нормированной
# координате внутри региона; `enddiff` — среднее последней трети минус среднее первой трети; `bins4` —
# средние четырёх последовательных бинов (контроль числа признаков для 4-компонентного описания).
# Модель для всех представлений одна: импутация медианой train → стандартизация по train → LASSO с
# alpha, выбранным 5-кратной CV на train. Внутренний тест, CR и внешние когорты в настройке не участвуют.

# %%
ALL = np.arange(len(M.ids))
FEAT_CACHE = {}
def feats(part, kinds, mc=None, cov=None, rows=None):
    rows = ALL if rows is None else rows
    F = core.region_features(M, rows, PART[part][1], kinds=tuple(k for k in kinds if k != "bins4") +
                             (("bins",) if "bins4" in kinds else ()), nbins=4, mc=mc, cov=cov)
    F["bins4"] = F.pop("bins") if "bins" in F else None
    return np.hstack([F[k] for k in kinds])

def make_X(rep, mc=None, cov=None, rows=None):
    kind, part, kinds = REPS[rep]
    if kind == "cpg":
        return core.cpg_features(M, ALL if rows is None else rows, part, mc=mc, cov=cov)
    return feats(part, kinds, mc=mc, cov=cov, rows=rows)

FULL4 = ("mean", "sd", "slope", "enddiff")
in_db = qc_sites[mem_db[qc_sites] >= 0]
REPS = {
    # E1: отбор сайтов против агрегации
    "A_CpG_all_QC":            ("cpg", qc_sites, None),
    "B_CpG_in_DBSCAN":         ("cpg", in_db, None),
    "C1_DBSCAN_mean_sameSites": ("reg", "DBSCAN_sameSites", ("mean",)),
    "C2_DBSCAN_mean":          ("reg", "DBSCAN", ("mean",)),
    "D_DBSCAN_full4":          ("reg", "DBSCAN", FULL4),
    # E2: абляция описания паттерна на тех же DBSCAN-регионах
    "E2_DBSCAN_mean+sd":       ("reg", "DBSCAN", ("mean", "sd")),
    "E2_DBSCAN_mean+slope":    ("reg", "DBSCAN", ("mean", "slope")),
    "E2_DBSCAN_bins4":         ("reg", "DBSCAN", ("bins4",)),
    # E3: границы по сигналу
    "E3_Window5kb_mean":       ("reg", "Window5kb", ("mean",)),
    "E3_MethyLasso_mean":      ("reg", "MethyLasso", ("mean",)),
    "E3_MethyLasso_full4":     ("reg", "MethyLasso", FULL4),
    "E3_DBSCAN∩MethyLasso_mean":  ("reg", "DBSCAN∩MethyLasso", ("mean",)),
    "E3_DBSCAN∩MethyLasso_full4": ("reg", "DBSCAN∩MethyLasso", FULL4),
    # E4: границы состояний хроматина
    "E4_ChromHMM_mean":        ("reg", "ChromHMM", ("mean",)),
    "E4_DBSCAN∩ChromHMM_mean": ("reg", "DBSCAN∩ChromHMM", ("mean",)),
    "E4_DBSCAN∩ChromHMM_full4": ("reg", "DBSCAN∩ChromHMM", FULL4),
}
REPS = {k: v for k, v in REPS.items() if v[0] == "cpg" or v[1] in PART}

MODELS, PRED, ROWS = {}, {}, []
def evaluate(rep, X=None):
    t0 = time.time()
    X = make_X(rep) if X is None else X
    m = core.AgeModel(seed=SEED).fit(X[R["train"]], y[R["train"]])
    MODELS[rep] = m
    for s in SPLITS[1:]:
        p = m.predict(X[R[s]]); PRED[(rep, s)] = p
        if s == "CR":
            continue
        row = {"representation": rep, "split": s, **core.metrics(y[R[s]], p, seed=SEED),
               "n_features": m.n_features, "n_selected": m.n_selected,
               "missing_in_selected": m.feature_missing(X[R[s]])}
        ROWS.append(row)
    print(f"{rep:32s} признаков {m.n_features:>8d}, выбрано {m.n_selected:>4d}, {time.time()-t0:5.0f} с | " +
          " ".join(f"{s}: MAE {r['MAE']:.2f}" for r in ROWS[-3:] for s in [r['split']]), flush=True)
    del X

for rep in REPS:
    evaluate(rep)
res = pd.DataFrame(ROWS)
res.to_csv(os.path.join(OUT, "E1-E4_metrics.tsv"), sep="\t", index=False)

# %% [markdown]
# ### Результаты E1–E4
# Таблица: MAE (мес.) с 95% бутстреп-интервалом, R², смещение, число признаков, число выбранных LASSO
# признаков и доля пропусков среди выбранных признаков в соответствующей когорте (до импутации).

# %%
show = res.pivot_table(index="representation", columns="split", values=["MAE", "R2"]).round(2)
show = show.reindex(list(REPS))
print(show.to_string())
CONTRASTS = [  # (что сравниваем, лучше/хуже)
    ("E1 отбор сайтов: B против A", "B_CpG_in_DBSCAN", "A_CpG_all_QC"),
    ("E1 агрегация на тех же сайтах: C1 против B", "C1_DBSCAN_mean_sameSites", "B_CpG_in_DBSCAN"),
    ("E1 все наблюдаемые CpG кластера: C2 против C1", "C2_DBSCAN_mean", "C1_DBSCAN_mean_sameSites"),
    ("E1/E2 паттерн: D против C2", "D_DBSCAN_full4", "C2_DBSCAN_mean"),
    ("E2 контроль размерности: full4 против bins4", "D_DBSCAN_full4", "E2_DBSCAN_bins4"),
    ("E3 MethyLasso против DBSCAN (mean)", "E3_MethyLasso_mean", "C2_DBSCAN_mean"),
    ("E3 DBSCAN∩MethyLasso против DBSCAN (mean)", "E3_DBSCAN∩MethyLasso_mean", "C2_DBSCAN_mean"),
    ("E3 окна 5 кб против DBSCAN (mean)", "E3_Window5kb_mean", "C2_DBSCAN_mean"),
    ("E4 ChromHMM против DBSCAN (mean)", "E4_ChromHMM_mean", "C2_DBSCAN_mean"),
    ("E4 DBSCAN∩ChromHMM против DBSCAN (mean)", "E4_DBSCAN∩ChromHMM_mean", "C2_DBSCAN_mean"),
    ("E4 DBSCAN∩ChromHMM: паттерн против mean", "E4_DBSCAN∩ChromHMM_full4", "E4_DBSCAN∩ChromHMM_mean"),
]
crow = []
for name, a, b in CONTRASTS:
    if (a, "external_B6") not in PRED or (b, "external_B6") not in PRED:
        continue
    for s in ["internal_test", "external_B6", "external_BALB"]:
        d, lo, hi = core.paired_mae_diff(y[R[s]], PRED[(a, s)], PRED[(b, s)], seed=SEED)
        crow.append({"сравнение": name, "split": s, "ΔMAE (первый − второй)": d, "95% ДИ низ": lo, "95% ДИ верх": hi})
contr = pd.DataFrame(crow)
contr.to_csv(os.path.join(OUT, "E1-E4_paired_contrasts.tsv"), sep="\t", index=False)
print(contr.round(2).to_string(index=False))

# %% [markdown]
# ## 9. E5 — пространственный контекст через графовый штраф
# Узлы — DBSCAN-регионы (признак `mean`). Чтобы графовая модель была вычислимой, внутри train отбираются
# 10 000 регионов с наибольшей |r| с возрастом (одинаково для всех графов). Модель:
# \[ \min_\theta \tfrac{1}{2n}\big(\|y-Z\theta\|^2+\lambda_2\textstyle\sum_{(r,q)\in E} w_{rq}(\theta_r-\theta_q)^2\big)+\alpha\|\theta\|_1, \]
# где \(w_{rq}=\log_2(O/E)\). \(\lambda_2\in\{0,1,10,100\}\) и \(\alpha\) выбираются 5-кратной CV на train.
# Графы: линейное соседство (соседние отобранные регионы ближе 100 кб), контакты CH12, CH12 с
# рандомизацией рёбер при сохранении степеней узлов, контакты mESC (другой тип клеток).

# %%
E5_ROWS, E5_INFO = [], []
if RUN_HIC:
    X5 = make_X("C2_DBSCAN_mean")
    pre5 = core.Preprocessor().fit(X5[R["train"]])
    Ztr = pre5.transform(X5[R["train"]]).astype(np.float64)
    ytr = y[R["train"]]
    r_age = np.abs(((Ztr - Ztr.mean(0)) * (ytr - ytr.mean())[:, None]).mean(0) / (Ztr.std(0) * ytr.std() + 1e-12))
    K = min(10000, Ztr.shape[1])
    top = np.argsort(-r_age)[:K]
    reg_db = PART["DBSCAN"][0]
    kept_regions = np.where(pre5.keep)[0][top]
    graphs = {"none": pd.DataFrame(columns=["a", "b", "w", "chr"]),
              "linear_100kb": core.linear_edges(reg_db, kept_regions),
              "CH12_contacts": core.contact_edges(reg_db, contacts["CH12"], nodes=kept_regions)}
    graphs["CH12_rewired"] = core.rewire_edges(graphs["CH12_contacts"], seed=SEED)
    graphs["mESC_contacts"] = core.contact_edges(reg_db, contacts["mESC"], nodes=kept_regions)
    Zs = {s: pre5.transform(X5[R[s]]).astype(np.float64)[:, top] for s in SPLITS[1:]}
    for gname, E in graphs.items():
        t0 = time.time()
        B = core.incidence(E, K)
        fit = core.graph_lasso_cv(Ztr[:, top], ytr, B, [0.0] if gname == "none" else [1.0, 10.0, 100.0], seed=SEED)
        E5_INFO.append({"graph": gname, "edges": len(E), "lambda2": fit["lam2"], "alpha": fit["alpha"],
                        "cv_MSE": fit["cv_mse"], "n_selected": int((fit["coef"] != 0).sum())})
        for s in SPLITS[1:]:
            p = (Zs[s] - fit["mu"]) @ fit["coef"] + fit["ymu"]
            PRED[("E5_" + gname, s)] = p
            if s != "CR":
                E5_ROWS.append({"graph": gname, "split": s, **core.metrics(y[R[s]], p, seed=SEED)})
        print(f"{gname:15s} рёбер {len(E):>7d} λ2={fit['lam2']:<6} выбрано {E5_INFO[-1]['n_selected']:>4d} "
              f"CV-MSE {fit['cv_mse']:.2f} {time.time()-t0:.0f} с", flush=True)
    e5 = pd.DataFrame(E5_ROWS); e5i = pd.DataFrame(E5_INFO)
    e5.to_csv(os.path.join(OUT, "E5_graph_metrics.tsv"), sep="\t", index=False)
    e5i.to_csv(os.path.join(OUT, "E5_graph_selection.tsv"), sep="\t", index=False)
    print(e5i.round(3).to_string(index=False))
    print(e5.pivot_table(index="graph", columns="split", values=["MAE", "R2"]).round(2).to_string())
    crow5 = []
    for g in ["linear_100kb", "CH12_contacts", "CH12_rewired", "mESC_contacts"]:
        for ref in ["none", "CH12_rewired"]:
            if g == ref:
                continue
            for s in ["internal_test", "external_B6"]:
                d, lo, hi = core.paired_mae_diff(y[R[s]], PRED[("E5_" + g, s)], PRED[("E5_" + ref, s)], seed=SEED)
                crow5.append({"граф": g, "против": ref, "split": s, "ΔMAE": d, "ДИ низ": lo, "ДИ верх": hi})
    e5c = pd.DataFrame(crow5); e5c.to_csv(os.path.join(OUT, "E5_paired_contrasts.tsv"), sep="\t", index=False)
    print(e5c.round(2).to_string(index=False))
    del X5

# %% [markdown]
# ## 10. E6 — устойчивость к неполному наблюдению (внешняя когорта C57BL/6J)
# Модели, обученные выше, не переобучаются. Для образцов внешнего теста counts ухудшаются двумя
# разными способами: (а) случайное удаление доли наблюдаемых CpG (имитация несовпадения захваченных
# сайтов); (б) биномиальное прореживание чтений (имитация меньшей глубины; приближение, т.к. чтения,
# покрывающие несколько CpG, прореживаются независимо). Повторы с двумя зёрнами.

# %%
E6_REPS = [r for r in ["A_CpG_all_QC", "C2_DBSCAN_mean", "D_DBSCAN_full4", "E3_Window5kb_mean",
                       "E3_MethyLasso_mean", "E4_DBSCAN∩ChromHMM_mean"] if r in MODELS]
LEVELS = [("mask", 0.25), ("mask", 0.5), ("mask", 0.75), ("thin", 0.5), ("thin", 0.25), ("thin", 0.1)]
rows_ext = R["external_B6"]
E6 = []
for rep in E6_REPS:
    E6.append({"representation": rep, "mode": "none", "level": 0, "seed": 0,
               **core.metrics(y[rows_ext], PRED[(rep, "external_B6")], n_boot=200)})
for mode, lev in LEVELS:
    for sd in range(2):
        rng = np.random.default_rng(SEED + sd)
        mc_d, cv_d = core.degrade_counts(MC[rows_ext], CV[rows_ext], mode, lev, rng)
        obs_frac = float((cv_d >= MIN_COV).mean())
        for rep in E6_REPS:
            X = make_X(rep, mc=mc_d, cov=cv_d, rows=np.arange(len(rows_ext)))
            p = MODELS[rep].predict(X)
            E6.append({"representation": rep, "mode": mode, "level": lev, "seed": sd, "obs_frac": obs_frac,
                       **core.metrics(y[rows_ext], p, n_boot=200)})
        print(f"{mode} {lev} seed {sd}: доля наблюдаемых CpG {obs_frac:.3f}", flush=True)
e6 = pd.DataFrame(E6)
e6.to_csv(os.path.join(OUT, "E6_robustness.tsv"), sep="\t", index=False)
e6s = e6.groupby(["representation", "mode", "level"])[["MAE", "R2"]].mean().round(2).unstack(["mode", "level"])
print(e6s.to_string())

# %% [markdown]
# ## 11. Ускорение эпигенетического возраста при ограничении калорий (CR)
# Остатки предсказаний CR-мышей относительно линии `pred ~ age`, построенной на контролях внутреннего
# теста (они не участвовали в обучении). Отрицательное значение — замедление; p — критерий Манна–Уитни.

# %%
cr_rows = []
for rep in list(MODELS) + [k[0] for k in PRED if k[0].startswith("E5_") and k[1] == "CR"]:
    if (rep, "CR") not in PRED:
        continue
    a = core.age_acceleration(PRED[(rep, "internal_test")], y[R["internal_test"]], PRED[(rep, "CR")], y[R["CR"]])
    cr_rows.append({"representation": rep, **a})
cr = pd.DataFrame(cr_rows).drop_duplicates("representation")
cr.to_csv(os.path.join(OUT, "CR_age_acceleration.tsv"), sep="\t", index=False)
print(cr.round(3).to_string(index=False))

# %% [markdown]
# ## 12. Графики и сохранение результатов

# %%
import matplotlib
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
for k, s in enumerate(["internal_test", "external_B6"]):
    d = res[res.split == s].set_index("representation").reindex(list(REPS)).dropna(subset=["MAE"])
    ax[k].barh(range(len(d)), d.MAE, xerr=[d.MAE - d.MAE_lo, d.MAE_hi - d.MAE], color="#4C72B0", alpha=.85)
    ax[k].set_yticks(range(len(d))); ax[k].set_yticklabels(d.index, fontsize=8); ax[k].invert_yaxis()
    ax[k].set_xlabel("MAE, мес. (95% бутстреп-ДИ)"); ax[k].set_title({"internal_test": "Внутренний тест (Petkovich)",
                                                                       "external_B6": "Внешний тест (Thompson, C57BL/6J)"}[s])
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_E1-E4_MAE.png"), dpi=150); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for k, mode in enumerate(["mask", "thin"]):
    for rep in E6_REPS:
        d = e6[(e6.representation == rep) & (e6["mode"].isin(["none", mode]))].groupby("level").MAE.mean()
        x = d.index.values if mode == "mask" else np.where(d.index.values == 0, 1.0, d.index.values)
        o = np.argsort(x)
        ax[k].plot(x[o], d.values[o], "o-", label=rep)
    ax[k].set_xlabel("доля удалённых CpG" if mode == "mask" else "доля сохранённых чтений")
    if mode == "thin":
        ax[k].invert_xaxis()
    ax[k].set_ylabel("MAE, мес."); ax[k].set_title("Удаление CpG" if mode == "mask" else "Прореживание чтений")
ax[1].legend(fontsize=7, loc="upper left")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_E6_robustness.png"), dpi=150); plt.show()

pd.DataFrame([{"representation": r, "sample_id": M.ids[i], "split": s, "age_months": y[i], "pred": p}
              for (r, s), pp in PRED.items() for i, p in zip(R[s], pp)]).to_csv(
    os.path.join(OUT, "predictions.tsv"), sep="\t", index=False)
for k, (reg, mem) in PART.items():
    if k == "DBSCAN_sameSites":
        continue
    out = reg.copy(); out["chr"] = "chr" + (out.chr + 1).astype(str); out["start"] = out.start - 1
    out[["chr", "start", "end", "n_cpg"]].to_csv(os.path.join(OUT, f"regions_{k}.bed"), sep="\t", header=False, index=False)
config = {k: v for k, v in dict(SEED=SEED, MIN_COV=MIN_COV, MIN_FRAC_TRAIN=MIN_FRAC_TRAIN, QC_FRAC_CPG=QC_FRAC_CPG,
          DBSCAN_EPS=DBSCAN_EPS, DBSCAN_MINPTS=DBSCAN_MINPTS, RUN_METHYLASSO=RUN_METHYLASSO, RUN_HIC=RUN_HIC,
          n_cpg_dictionary=int(len(keys))).items()}
if RUN_METHYLASSO and os.path.exists(os.path.join(METHYLASSO_DIR, "COMMIT")):
    config["methylasso_commit"] = open(os.path.join(METHYLASSO_DIR, "COMMIT")).read().strip()
json.dump(config, open(os.path.join(OUT, "config.json"), "w"), indent=2, ensure_ascii=False)
meta.to_csv(os.path.join(OUT, "samples_and_split.tsv"), sep="\t", index=False)
shutil.make_archive(os.path.join(WORK, "results_pipeline1"), "zip", OUT)
print("Архив:", os.path.join(WORK, "results_pipeline1.zip"))
if IN_COLAB:
    from google.colab import files
    files.download(os.path.join(WORK, "results_pipeline1.zip"))
