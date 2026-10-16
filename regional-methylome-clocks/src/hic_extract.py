import cooler, numpy as np, pandas as pd, sys, time
def extract_contacts(mcool, res=25000, min_d_bins=2, max_d_bins=80, oe_min=2.0, raw_min=5, chroms=None, block=400):
    c = cooler.Cooler(f"{mcool}::resolutions/{res}")
    chroms = chroms or [f"chr{i}" for i in range(1, 20)]
    out = []
    for ch in chroms:
        t = time.time()
        ext = c.extent(ch); nb = ext[1]-ext[0]
        w = c.bins().fetch(ch)['weight'].values
        valid = np.isfinite(w)
        parts = []
        for s in range(0, nb, block):
            e = min(nb, s+block); e2 = min(nb, e+max_d_bins)
            px = c.pixels(join=False).fetch((ch, s*res, min(e*res, int(c.chromsizes[ch]))))  # rows in [s,e)
            px = px[(px.bin2_id - ext[0] < e2)]
            i = px.bin1_id.values - ext[0]; j = px.bin2_id.values - ext[0]
            d = j - i
            keep = (d >= min_d_bins) & (d <= max_d_bins) & valid[i] & valid[j]
            i, j, d, cnt = i[keep], j[keep], d[keep], px['count'].values[keep]
            bal = cnt * w[i] * w[j]
            parts.append(pd.DataFrame({'i': i, 'j': j, 'd': d, 'raw': cnt, 'bal': bal}))
        P = pd.concat(parts, ignore_index=True)
        # expected per diagonal = sum(balanced) / number of valid pairs at that distance
        vcum = np.concatenate([[0], np.cumsum(valid)])
        npairs = {dd: int(sum(valid[k] and valid[k+dd] for k in range(0, nb-dd))) for dd in range(min_d_bins, max_d_bins+1)}
        exp = P.groupby('d')['bal'].sum() / pd.Series(npairs)
        P['oe'] = P['bal'] / exp.reindex(P['d']).values
        P = P[(P.oe >= oe_min) & (P.raw >= raw_min)]
        out.append(pd.DataFrame({'chr': ch, 'start1': P.i.values*res, 'start2': P.j.values*res,
                                 'raw': P.raw.values.astype(np.int32), 'oe': P.oe.values.astype(np.float32)}))
        print(ch, nb, len(P), round(time.time()-t, 1), flush=True)
    return pd.concat(out, ignore_index=True)
if __name__ == '__main__':
    df = extract_contacts(sys.argv[1]); df.to_parquet(sys.argv[2]); print(len(df))
