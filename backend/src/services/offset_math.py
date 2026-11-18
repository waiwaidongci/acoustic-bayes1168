"""零偏偏移区间分析的核心数学。

口径（与 algorithm.calculate 完全一致的冻结模型）：
- 分层交叉验证（StratifiedKFold, n_splits=folds, shuffle=True, random_state=42）；
- 每折只在训练行拟合 StandardScaler 与 GaussianNB(var_smoothing=smoothing)；
- 验证样本（折外样本）才参与偏移分析；
- 仅给频谱质心 centroid 加同一偏移 o：z_c(o) = z_c(0) + o / scale_c，其余特征不变；
- 判读 = argmax 联合对数后验 log p(x) + log P(class)，与 sklearn 同口径（含并列时取编号最小类）。

关键事实：两类 a/b 的联合对数后验之差在 t = o / scale_c 上是二次函数，
因此每条样本的判读只可能在至多 K*(K-1) 个临界偏移处切换。
我们不枚举偏移、也不对每个偏移重判整批，而是为每条样本一次性求出区间内的全部临界偏移，
产出“判读进入/离开目标故障”的阶梯事件；同一偏移的跨片事件在归并阶段合并后再求极值。

参数化约定（务必统一，避免混用坐标系）：
令 zc0 为零偏时质心的标准化值，t = o / scale_c 为相对零偏的位移，则质心维取值 zc = zc0 + t。
高斯对数密度的质心维关于 t 展开为
    f_k(t) = A_k t^2 + B_k t + C_k
    A_k = -0.5 / sigma_k
    B_k = (theta_k - zc0) / sigma_k
    C_k = -0.5*log(2*pi*sigma_k) - 0.5*(zc0-theta_k)^2 / sigma_k
B、C 依赖具体样本的 zc0，逐样本计算；A 与其余维贡献为类级常量。
"""
import math
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

FEATURES = ('rms', 'centroid', 'kurtosis')
CENTROID_POS = FEATURES.index('centroid')


class OffsetStudyError(ValueError):
    pass


def validate_study_input(payload):
    """校验偏移研究入参，返回规整后的 dict。"""
    out = {}
    try:
        lo = float(payload.get('offset_min'))
        hi = float(payload.get('offset_max'))
    except (TypeError, ValueError):
        raise OffsetStudyError('偏移上下限必须是有限数值（Hz）')
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise OffsetStudyError('偏移上下限必须是有限数值（Hz）')
    if lo > hi:
        raise OffsetStudyError('偏移下限不得大于上限')
    out['offset_min'] = lo
    out['offset_max'] = hi
    target = payload.get('target_label')
    if not isinstance(target, str) or not target.strip() or len(target) > 600:
        raise OffsetStudyError('请指定要统计的故障判读类型')
    out['target_label'] = target.strip()
    shard = payload.get('shard_size', 1000)
    try:
        shard = int(shard)
    except (TypeError, ValueError):
        raise OffsetStudyError('分片样本数必须为 1–100000 的整数')
    if not 1 <= shard <= 100_000:
        raise OffsetStudyError('分片样本数必须为 1–100000 的整数')
    out['shard_size'] = shard
    return out


def build_frozen_models(records, parameters):
    """复现 algorithm.calculate 的冻结折模型，返回各折 (scaler, gnb) 与折外索引。"""
    data = [r['values'] for r in records]
    X = np.array([[r[f] for f in FEATURES] for r in data], dtype=float)
    y = np.array([r['label'] for r in data])
    labels, counts = np.unique(y, return_counts=True)
    folds = int(parameters['folds'])
    if len(labels) < 2 or min(counts) < folds:
        raise OffsetStudyError('至少两种故障类型，且每类样本数不少于交叉验证折数')
    if len(data) < 12:
        raise OffsetStudyError('分类研究至少需要12条样本')
    splitter = StratifiedKFold(folds, shuffle=True, random_state=42)
    fold_models = []
    test_index_by_row = np.full(len(y), -1, dtype=int)
    for fold, (train, test) in enumerate(splitter.split(X, y)):
        scaler = StandardScaler().fit(X[train])
        gnb = GaussianNB(var_smoothing=float(parameters['smoothing'])).fit(
            scaler.transform(X[train]), y[train])
        fold_models.append((scaler, gnb))
        test_index_by_row[test] = fold
    return X, y, labels, fold_models, test_index_by_row


def _fold_arrays(scaler, gnb):
    """抽取一折冻结模型的类级常量（不含样本相关的 zc0 项）。"""
    theta = np.asarray(gnb.theta_, dtype=float)          # (K, d)
    sigma = np.asarray(gnb.var_, dtype=float)            # (K, d)
    # sklearn 1.5 的 GaussianNB：class_prior_ 为原始先验概率，对数先验需取 log
    log_prior = np.log(np.asarray(gnb.class_prior_, dtype=float))
    scale = float(scaler.scale_[CENTROID_POS])
    if not np.isfinite(scale) or scale <= 0.0:
        raise OffsetStudyError('质心特征在训练折内退化（尺度非正），无法施加偏移')
    j = CENTROID_POS
    a = -0.5 / sigma[:, j]                                # 质心 t^2 系数 A_k
    c_norm = -0.5 * np.log(2.0 * np.pi * sigma[:, j])     # 质心维归一化常数
    return {'theta': theta, 'sigma': sigma, 'scale': scale,
            'a': a, 'c_norm': c_norm,
            'theta_other': np.delete(theta, j, axis=1),
            'sigma_other': np.delete(sigma, j, axis=1),
            'log_prior': log_prior,
            'classes': np.asarray(gnb.classes_)}


def _sample_quadratic(fa, z_other, zc0):
    """计算一条样本各类 jlp_k(t) = A_k t^2 + B_k t + C_k 的系数与固定贡献。

    返回 (A, B, C)，各为 (K,) 数组；C 已含其余维高斯贡献与类先验。
    """
    j = CENTROID_POS
    sigma_c = fa['sigma'][:, j]
    theta_c = fa['theta'][:, j]
    A = fa['a']
    B = (theta_c - zc0) / sigma_c
    centroid_c0 = fa['c_norm'] - 0.5 * ((zc0 - theta_c) ** 2) / sigma_c
    other = (np.sum(-0.5 * np.log(2.0 * np.pi * fa['sigma_other'])
                    - 0.5 * (z_other[None, :] - fa['theta_other']) ** 2
                    / fa['sigma_other'], axis=1)
             + fa['log_prior'])
    C = other + centroid_c0
    return A, B, C


def _joint_log_prob(fa, z_other, zc0, t):
    """单样本联合对数后验（与 sklearn GaussianNB 同口径），返回 (K,)。"""
    A, B, C = _sample_quadratic(fa, z_other, zc0)
    return A * (t ** 2) + B * t + C


def _quadratic_roots(A, B, C):
    """稳健求解 A t^2 + B t + C = 0 的实根。"""
    scale_ref = max(1.0, abs(A), abs(B), abs(C))
    if abs(A) <= 1e-14 * scale_ref:
        if abs(B) > 1e-14 * scale_ref:
            return [-C / B]
        return []
    disc = B * B - 4.0 * A * C
    if disc <= 0.0:
        return []  # 判别式为 0 的切点两侧判读不变，无需事件
    sq = math.sqrt(disc)
    q = -0.5 * (B + math.copysign(sq, B))
    r1 = q / A
    r2 = C / q if q != 0.0 else -B / A
    return [r1, r2]


def sample_segments(scaler, gnb, z_other, zc0, target_pos, lo, hi):
    """计算单条验证样本在 [lo, hi] 上“判读为目标故障”的分段指示。

    返回 (segments, target_at_lo)：
    - segments 为 [start_t, end_t, kind, target_flag] 列表（t 为标准化位移，未乘 scale），
      按 t 升序、互不重叠且并集恰为 [t_lo, t_hi]；
      kind='point' 表示单点段（临界并列点或端点），kind='open' 表示开区间段 (start,end)；
    - target_at_lo 为左端点（含端点、含并列按 sklearn 取编号最小类）是否判为目标。

    每段内部 jlp 大小关系恒定：开区间取段中点严格 argmax；
    分界点按“机器精度并列→编号最小类”单独判定，精确处理并列判读。
    """
    fa = _fold_arrays(scaler, gnb)
    scale = fa['scale']
    K = fa['theta'].shape[0]
    A, B, C = _sample_quadratic(fa, z_other, zc0)
    dA = A[:, None] - A[None, :]
    dB = B[:, None] - B[None, :]
    dC = C[:, None] - C[None, :]

    t_lo, t_hi = lo / scale, hi / scale
    crit_set = set()
    for u in range(K):
        for v in range(u + 1, K):
            for t in _quadratic_roots(dA[u, v], dB[u, v], dC[u, v]):
                if t_lo < t < t_hi:
                    crit_set.add(t)
    crit = sorted(crit_set)

    def jlp_at(t):
        return _joint_log_prob(fa, z_other, zc0, t)

    def point_pred(t):
        jlp = jlp_at(t)
        order = np.argsort(jlp)[::-1]
        top, second = int(order[0]), int(order[1])
        ref = max(1.0, abs(jlp[top]), abs(jlp[second]))
        if abs(jlp[top] - jlp[second]) <= 1e-12 * ref:
            return min(top, second)  # 数值并列取编号最小类，同 sklearn tie-break
        return top

    target_at_lo = point_pred(t_lo) == target_pos
    if not crit:
        # 无临界根：整段判读恒定。显式给出两端点段与覆盖内部的开区间段，
        # 确保端点（含端点口径）与内点都能被精确计入。
        flag = int(np.argmax(jlp_at((t_lo + t_hi) * 0.5))) == target_pos
        if t_lo == t_hi:
            return [[t_lo, t_lo, 'point', target_at_lo]], target_at_lo
        return ([[t_lo, t_lo, 'point', target_at_lo],
                 [t_lo, t_hi, 'open', flag],
                 [t_hi, t_hi, 'point', point_pred(t_hi) == target_pos]],
                target_at_lo)

    segments = [[t_lo, t_lo, 'point', target_at_lo]]
    prev_t = t_lo
    for t in crit:
        # 开区间 (prev_t, t)
        open_flag = int(np.argmax(jlp_at(prev_t + (t - prev_t) * 0.5))) == target_pos
        segments.append([prev_t, t, 'open', open_flag])
        # 临界点单点（含并列口径）
        segments.append([t, t, 'point', point_pred(t) == target_pos])
        prev_t = t
    # 末开区间 (last_crit, t_hi)
    last_t = crit[-1]
    open_flag = int(np.argmax(jlp_at(last_t + (t_hi - last_t) * 0.5))) == target_pos
    segments.append([last_t, t_hi, 'open', open_flag])
    # 右端点始终单列点段（含端点/并列口径），即使与末开区间判读相同也不能省略，
    # 否则闭区间右端点不会被点匹配计入。
    segments.append([t_hi, t_hi, 'point', point_pred(t_hi) == target_pos])
    return segments, target_at_lo


def plan_shards(n_samples, shard_size):
    """把验证样本行（在验证序列中的序号）切成有限内存分片。"""
    return [(i, min(i + shard_size, n_samples))
            for i in range(0, n_samples, shard_size)]
