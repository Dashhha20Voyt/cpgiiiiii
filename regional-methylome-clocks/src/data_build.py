"""Сборка реальных когорт RRBS крови мыши (mm10) для обоих пайплайнов.

Источники (NCBI GEO, открытый доступ):
  * GSE80672 (Petkovich et al., 2017) — кровь C57BL/6, переобработка в mm10
    из GSE120136 (BS-Seeker2, файлы CGfinal), используется как обучающая когорта;
  * GSE120132 (Thompson et al., 2018) — кровь C57BL/6J и BALB/cByJ, те же CGfinal mm10,
    используется как независимая внешняя когорта.
Формат CGfinal (описание GSE120136): chromosome, dinucleotide, position,
methylated counts, total counts, methylation frequency, 95% CI.
"""
import gzip, io, os, re, subprocess, time, urllib.request
import numpy as np
import pandas as pd

AUTOSOMES = [f"chr{i}" for i in range(1, 20)]
CHR_INDEX = {c: i for i, c in enumerate(AUTOSOMES)}
GEO = "https://ftp.ncbi.nlm.nih.gov/geo"
PETKOVICH_TAR = f"{GEO}/series/GSE120nnn/GSE120136/suppl/GSE120136_GSE80672_re-analyzed_Samples_processed_data.tar.gz"
PETKOVICH_MATRIX = f"{GEO}/series/GSE80nnn/GSE80672/matrix/GSE80672_series_matrix.txt.gz"
THOMPSON_MATRICES = [f"{GEO}/series/GSE120nnn/GSE120132/matrix/GSE120132-GPL17021_series_matrix.txt.gz",
                     f"{GEO}/series/GSE120nnn/GSE120132/matrix/GSE120132-GPL21103_series_matrix.txt.gz"]
THOMPSON_FILELIST = f"{GEO}/series/GSE120nnn/GSE120132/suppl/filelist.txt"


def _get(url, dest):
    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        subprocess.run(["curl", "-sSfL", "--retry", "5", "-o", dest, url], check=True)
    return dest


def parse_series_matrix(path):
    rows = {}
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if not line.startswith("!Sample_"):
                continue
            parts = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")]
            key, vals = parts[0], parts[1:]
            if key == "!Sample_characteristics_ch1":
                name = vals[0].split(":")[0].strip()
                rows[name] = [v.split(":", 1)[1].strip() if ":" in v else v for v in vals]
            else:
                rows[key.replace("!Sample_", "")] = vals
    n = len(rows["geo_accession"])
    return pd.DataFrame({k: v for k, v in rows.items() if len(v) == n})


def petkovich_metadata(workdir):
    p = _get(PETKOVICH_MATRIX, os.path.join(workdir, "GSE80672_series_matrix.txt.gz"))
    m = parse_series_matrix(p)
    # В GEO поле подписано "age (years)", но значения 0.67–35 соответствуют месяцам
    # (Petkovich et al., 2017: мыши 3–35 мес.). Перевод не выполняется, единица — месяцы.
    m["age_months"] = pd.to_numeric(m["age (years)"], errors="coerce")
    base = (m["tissue"].str.lower() == "whole blood") & (m["strain"] == "C57BL/6") & \
           (m["genetic condition"] == "Wild Type")
    m["group"] = np.where(base & (m["diet"] == "standard"), "petkovich_control",
                  np.where(base & (m["diet"] == "Calorie Restricted"), "petkovich_CR", "excluded"))
    m = m[m.group != "excluded"].copy()
    m["sample_id"] = m["geo_accession"]
    m["cohort"] = "GSE80672"
    m["sex"] = m["gender"]
    return m[["sample_id", "title", "cohort", "group", "age_months", "sex", "strain", "diet"]].reset_index(drop=True)


def thompson_metadata(workdir):
    ms = []
    for u in THOMPSON_MATRICES:
        ms.append(parse_series_matrix(_get(u, os.path.join(workdir, os.path.basename(u)))))
    m = pd.concat(ms, ignore_index=True)
    m = m[m["tissue"] == "Blood"].copy()
    m["age_months"] = m["age"].str.replace("mo", "", regex=False).astype(float)
    m["group"] = np.where(m["strain"] == "C57BL/6J", "thompson_B6", "thompson_BALB")
    m["sample_id"] = m["geo_accession"]
    m["cohort"] = "GSE120132"
    m["sex"] = m["Sex"]
    m["diet"] = "standard"
    return m[["sample_id", "title", "cohort", "group", "age_months", "sex", "strain", "diet"]].reset_index(drop=True)


def download_petkovich(datadir, keep_ids):
    """Потоковая распаковка архива GSE120136 (~2.4 ГБ); на диске остаются только нужные образцы."""
    os.makedirs(datadir, exist_ok=True)
    done = os.path.join(datadir, ".done")
    if not os.path.exists(done):
        cmd = f"curl -sSfL --retry 5 '{PETKOVICH_TAR}' | tar xzf - -C '{datadir}'"
        subprocess.run(cmd, shell=True, check=True)
        open(done, "w").close()
    paths = {}
    for f in os.listdir(datadir):
        mm = re.match(r"GSE80672_(GSM\d+)\.CGfinal\.txt\.gz", f)
        if mm:
            if mm.group(1) in keep_ids:
                paths[mm.group(1)] = os.path.join(datadir, f)
            else:
                os.remove(os.path.join(datadir, f))
    return paths


def download_thompson(datadir, keep_ids, workers=6):
    os.makedirs(datadir, exist_ok=True)
    fl = _get(THOMPSON_FILELIST, os.path.join(datadir, "filelist.txt"))
    names = [l.split("\t")[1] for l in open(fl) if "Mouse_Blood" in l]
    paths, jobs = {}, []
    for n in names:
        gsm = n.split("_")[0]
        if gsm not in keep_ids:
            continue
        dest = os.path.join(datadir, n)
        paths[gsm] = dest
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            jobs.append(f"{GEO}/samples/{gsm[:-3]}nnn/{gsm}/suppl/{n} {dest}")
    if jobs:
        lst = os.path.join(datadir, "jobs.txt")
        open(lst, "w").write("\n".join(jobs) + "\n")
        subprocess.run(f"cat '{lst}' | xargs -P {workers} -n 2 sh -c 'curl -sSfL --retry 5 -o \"$1\" \"$0\"'",
                       shell=True, check=True)
    return paths


def read_cgfinal(path, min_cov=1):
    """Возвращает (key, mC, cov) для аутосом; key = chr_index * 2^32 + pos (исходная позиция CGfinal)."""
    d = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 2, 3, 4],
                    names=["chr", "pos", "mc", "cov"], engine="c",
                    dtype={"chr": str, "pos": np.int64, "mc": np.int64, "cov": np.int64})
    d = d[d["chr"].isin(CHR_INDEX) & (d["cov"] >= min_cov)]
    if (d["mc"] > d["cov"]).any() or (d["mc"] < 0).any():
        raise ValueError(f"Нарушено 0 <= mC <= coverage в {path}")
    key = d["chr"].map(CHR_INDEX).values.astype(np.int64) * (1 << 32) + d["pos"].values
    if len(np.unique(key)) != len(key):
        raise ValueError(f"Дубликаты координат в {path}")
    return key, d["mc"].values, d["cov"].values


def build_matrix(samples, paths, train_ids, out_npz, min_cov=5, min_frac_train=0.5, log=print):
    """Два прохода: (1) словарь сайтов по обучающим образцам; (2) counts всех образцов на этом словаре.
    Словарь зависит только от покрытия обучающих образцов (без возраста)."""
    if os.path.exists(out_npz):
        z = np.load(out_npz, allow_pickle=True)
        return z["keys"], z["mc"], z["cov"], list(z["sample_ids"])
    t0 = time.time()
    batch_u, batch_c, buf = [], [], []
    def flush():
        if buf:
            u, c = np.unique(np.concatenate(buf), return_counts=True)
            batch_u.append(u); batch_c.append(c); buf.clear()
    for k, sid in enumerate(train_ids):
        key, mc, cov = read_cgfinal(paths[sid])
        buf.append(key[cov >= min_cov])
        if len(buf) == 20:
            flush()
        if (k + 1) % 20 == 0:
            log(f"  проход 1: {k+1}/{len(train_ids)} обучающих образцов, {time.time()-t0:.0f} с")
    flush()
    allu = np.concatenate(batch_u); allc = np.concatenate(batch_c)
    order = np.argsort(allu, kind="stable"); allu, allc = allu[order], allc[order]
    uu, start = np.unique(allu, return_index=True)
    tot = np.add.reduceat(allc, start)
    keys = uu[tot >= min_frac_train * len(train_ids)]
    log(f"  словарь: {len(keys)} CpG с покрытием >= {min_cov} у >= {min_frac_train:.0%} обучающих образцов")
    ids = list(samples)
    MC = np.zeros((len(ids), len(keys)), dtype=np.uint16)
    CV = np.zeros((len(ids), len(keys)), dtype=np.uint16)
    for k, sid in enumerate(ids):
        key, mc, cov = read_cgfinal(paths[sid])
        idx = np.searchsorted(keys, key)
        idx[idx == len(keys)] = 0
        hit = keys[idx] == key
        # Ограничение uint16: при покрытии > 65535 counts масштабируются пропорционально (практически не встречается)
        cv = cov[hit].astype(np.float64); mcv = mc[hit].astype(np.float64)
        scale = np.where(cv > 65535, 65535 / cv, 1.0)
        CV[k, idx[hit]] = np.round(cv * scale).astype(np.uint16)
        MC[k, idx[hit]] = np.round(mcv * scale).astype(np.uint16)
        if (k + 1) % 25 == 0:
            log(f"  проход 2: {k+1}/{len(ids)} образцов, {time.time()-t0:.0f} с")
    np.savez(out_npz, keys=keys, mc=MC, cov=CV, sample_ids=np.array(ids))
    return keys, MC, CV, ids


def key_chr(keys):
    return (keys >> 32).astype(np.int32)


def key_pos(keys):
    return (keys & ((1 << 32) - 1)).astype(np.int64)
