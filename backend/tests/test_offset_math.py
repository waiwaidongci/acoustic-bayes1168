"""事件法零偏分析的独立 oracle 测试。

对照：在密集偏移网格上对每条验证样本用冻结的 sklearn 模型直接重判（暴力真值），
验证事件法给出的 N(o) 上下限、对应偏移、端点与并列判读完全一致。
运行：backend 目录下  python -m tests.test_offset_math
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from src.services.offset_math import (CENTROID_POS, build_frozen_models,
                                      sample_segments)
from src.services.offset_pipeline import merge_and_summarize, count_at_offset


def make_data(n=180, seed=7):
    rng = np.random.default_rng(seed)
    centers = [(1.2, 900, 2.8), (3.0, 1600, 5.2), (2.0, 2300, 3.9)]
    X, y = [], []
    for i in range(n):
        c = i % 3
        rms, cen, kur = centers[c]
        X.append([max(0.1, rng.normal(rms, 0.35)),
                  max(100.0, rng.normal(cen, 170)),
                  max(0.1, rng.normal(kur, 0.45))])
        y.append(['正常', '轴承磨损', '转子偏心'][c])
    records = [{'reference': f'S-{i:04d}',
                'values': {'rms': float(a), 'centroid': float(b),
                           'kurtosis': float(d), 'label': y[i]}}
               for i, (a, b, d) in enumerate(X)]
    return records


def brute_force(records, parameters, target, lo, hi, grid=801):
    """密集网格暴力真值：每个偏移都用冻结折模型重判全部验证样本（按折向量化）。"""
    X, y, labels, fold_models, test_fold = build_frozen_models(records, parameters)
    offsets = np.linspace(lo, hi, grid)
    counts = np.zeros(grid, dtype=int)
    # 同一折的验证样本批量预测，避免逐样本逐点调用 sklearn
    for fold, (scaler, gnb) in enumerate(fold_models):
        rows = [i for i, f in enumerate(test_fold) if f == fold]
        if not rows:
            continue
        fold_target = list(gnb.classes_).index(target)
        Z = scaler.transform(X[rows])
        scale_c = scaler.scale_[CENTROID_POS]
        for gi, o in enumerate(offsets):
            Zz = Z.copy()
            Zz[:, CENTROID_POS] = Z[:, CENTROID_POS] + o / scale_c
            preds = np.argmax(gnb.predict_log_proba(Zz), axis=1)
            counts[gi] += int(np.sum(preds == fold_target))
    return offsets, counts


def segment_curve(records, parameters, target, lo, hi, offsets):
    """分段法在给定网格点上的 N(o)，用点/开区间口径逐点判定，用于与暴力真值比对。"""
    X, y, labels, fold_models, test_fold = build_frozen_models(records, parameters)
    counts = np.zeros(len(offsets), dtype=int)
    for row_idx, fold in enumerate(test_fold):
        if fold < 0:
            continue
        scaler, gnb = fold_models[fold]
        fold_target = list(gnb.classes_).index(target)
        z = scaler.transform(X[row_idx].reshape(1, -1))[0]
        z_other = np.delete(z, CENTROID_POS)
        scale = float(scaler.scale_[CENTROID_POS])
        segments, _ = sample_segments(
            scaler, gnb, z_other, float(z[CENTROID_POS]), fold_target, lo, hi)
        for gi, o in enumerate(offsets):
            t = o / scale
            tol = 1e-12 * max(1.0, abs(t))
            hit = 0
            for s_t, e_t, kind, flag in segments:
                if not flag:
                    continue
                if kind == 'point':
                    if abs(s_t - t) <= tol:
                        hit = 1
                        break
                elif s_t + tol < t < e_t - tol:
                    hit = 1
                    break
            counts[gi] += hit
    return counts


class FakeStudy:
    def __init__(self, spec, plan, shard_status):
        self.id = 1
        self.spec = spec
        self.shard_plan = plan
        self.shard_status = shard_status


def run_case(lo, hi, target, n=180, shard_size=37, seed=7):
    records = make_data(n, seed)
    parameters = {'folds': 3, 'smoothing': 1e-9}
    offsets, bf_counts = brute_force(records, parameters, target, lo, hi)
    seg_counts = segment_curve(records, parameters, target, lo, hi, offsets)
    mismatch = int(np.sum(bf_counts != seg_counts))
    print(f'[{target} | {lo},{hi}] 网格点 {len(offsets)} 不一致 {mismatch} 个')
    assert mismatch == 0, '分段法与暴力重判在网格上不一致'

    # 用分片流水线落盘后归并
    from src.services.offset_pipeline import prepare, process_shard
    from pathlib import Path
    from types import SimpleNamespace
    import tempfile
    run = SimpleNamespace(state='completed',
                          snapshot={'records': records},
                          parameters=parameters,
                          result={'checksum': 'test'})
    payload = {'offset_min': lo, 'offset_max': hi,
               'target_label': target, 'shard_size': shard_size}
    spec, plan, context = prepare(run, payload)
    tmp = Path(tempfile.mkdtemp())
    study = FakeStudy(spec, plan, {})
    status = {}
    for shard in plan:
        info = process_shard(study, shard, context, tmp)
        status[str(shard['index'])] = info
        study.shard_status = status
    study.shard_status = {**status, '_meta': {'checksum': context['checksum']}}
    result = merge_and_summarize(study)

    true_min, true_max = int(bf_counts.min()), int(bf_counts.max())
    assert result['minimum']['count'] == true_min, (result['minimum'], true_min)
    assert result['maximum']['count'] == true_max, (result['maximum'], true_max)
    # 极值偏移覆盖：每个暴力极值网格点都必须落在事件法给出的偏移区间内；
    # 每个非极值网格点都不得被误纳入极值区间
    def covered(point, ranges):
        return any(a - 1e-9 <= point <= b + 1e-9 for a, b in ranges)
    min_pts = offsets[bf_counts == true_min]
    max_pts = offsets[bf_counts == true_max]
    non_min_pts = offsets[bf_counts != true_min]
    non_max_pts = offsets[bf_counts != true_max]
    assert all(covered(p, result['minimum']['offsets']) for p in min_pts)
    assert all(covered(p, result['maximum']['offsets']) for p in max_pts)
    assert not any(covered(p, result['minimum']['offsets']) for p in non_min_pts)
    assert not any(covered(p, result['maximum']['offsets']) for p in non_max_pts)
    # count_at_offset 与暴力逐点判读完全一致（含端点口径）
    for gi, o in enumerate(offsets):
        assert count_at_offset(study, float(o)) == int(bf_counts[gi]), o
    # 零偏对照（区间含零时）
    if lo <= 0.0 <= hi:
        zidx = int(np.argmin(np.abs(offsets)))
        assert result['count_at_zero_offset'] == int(bf_counts[zidx])
    else:
        assert result['count_at_zero_offset'] is None
    # 端点值核对
    assert result['minimum']['count'] <= bf_counts[0] <= result['maximum']['count']
    assert result['validation_samples'] == len(context['validation_rows'])
    print('  极值 min=%d %s | max=%d %s | 临界事件 %d | 分片 %d' % (
        result['minimum']['count'], result['minimum']['offsets'],
        result['maximum']['count'], result['maximum']['offsets'],
        result['critical_offsets'], result['shards']))
    return result


def main():
    # 含零、对称与非对称区间；三个标签各跑一次（覆盖目标/非目标切换）
    run_case(-300.0, 300.0, '轴承磨损')
    run_case(-500.0, 200.0, '正常')
    run_case(0.0, 400.0, '转子偏心')
    run_case(-150.0, -20.0, '轴承磨损', n=240, shard_size=50, seed=11)
    # 区间退化为单点
    run_case(120.0, 120.0, '正常', n=120, shard_size=19)
    # 大分片（单片）与极小分片（每片1条）两种极端
    run_case(-250.0, 250.0, '转子偏心', n=150, shard_size=100000)
    run_case(-250.0, 250.0, '转子偏心', n=150, shard_size=1)
    print('ALL OFFSET MATH TESTS PASSED')


if __name__ == '__main__':
    main()
