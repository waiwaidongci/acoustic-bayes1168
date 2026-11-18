"""零偏研究的分片处理、可恢复执行与跨片归并求极值。

有限内存策略：
- 验证样本按 shard_size 切片，逐片把每条样本的“目标判读分段指示”写入该片分段文件；
  分段区分单点段（临界并列点/端点）与开区间段，精确承载点判读与开区间判读两种口径；
- 单片刻画完成即原子落盘并在数据库记账（含输入校验和），中断重启后已完成片直接跳过，
  重试不会重复纳入；
- 全部片完成后用 heapq 对有序分段做多路归并，在每个边界（点/开区间）重算总判读数 N(o)，
  同一偏移的跨片变化先合并，再求精确上下限及达到它们的全部偏移（含端点与并列判读）。
"""
import hashlib
import heapq
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .offset_math import (CENTROID_POS, OffsetStudyError, build_frozen_models,
                          plan_shards, sample_segments, validate_study_input)

SHARD_VERSION = 1
EPS_SPAN = 1e-12
POINT = 'p'   # 单点段
OPEN = 'o'    # 开区间段 (start, end)


def study_dir(artifact_dir: Path, study_id: int) -> Path:
    d = Path(artifact_dir) / f'offset-study-{study_id}'
    d.mkdir(parents=True, exist_ok=True)
    return d


def shard_path(artifact_dir: Path, study_id: int, index: int) -> Path:
    return study_dir(artifact_dir, study_id) / f'shard-{index:05d}.segments'


def _records_checksum(records, spec, run_checksum):
    payload = json.dumps(
        {'run_checksum': run_checksum, 'spec': spec, 'records': records},
        sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def prepare(run, payload=None, spec_override=None):
    """基于已完成 run 的冻结快照构建研究规格与分片计划（不做任何重判）。"""
    if run.state != 'completed':
        raise OffsetStudyError('仅已完成的研究可发起零偏偏移分析')
    spec = spec_override if spec_override is not None else validate_study_input(payload)
    records = run.snapshot['records']
    X, y, labels, fold_models, test_fold = build_frozen_models(records, run.parameters)
    if spec['target_label'] not in set(map(str, labels)):
        raise OffsetStudyError('目标故障类型不在研究标签内：' + spec['target_label'])
    validation_rows = [i for i, f in enumerate(test_fold) if f >= 0]
    if not validation_rows:
        raise OffsetStudyError('没有可用的同批验证样本')
    shards = plan_shards(len(validation_rows), spec['shard_size'])
    plan = [{'index': k, 'start': s, 'end': e} for k, (s, e) in enumerate(shards)]
    run_checksum = run.result.get('checksum', '') if run.result else ''
    checksum = _records_checksum(
        [records[i]['reference'] for i in validation_rows], spec, run_checksum)
    context = {
        'records': records, 'X': X, 'y': y, 'labels': [str(x) for x in labels],
        'fold_models': fold_models, 'test_fold': test_fold,
        'validation_rows': validation_rows, 'checksum': checksum,
    }
    return spec, plan, context


def _shard_input_checksum(context, rows, spec):
    payload = json.dumps({
        'version': SHARD_VERSION,
        'study': context['checksum'],
        'rows': [context['records'][i]['reference'] for i in rows],
        'offset_min': spec['offset_min'], 'offset_max': spec['offset_max'],
        'target_label': spec['target_label'],
    }, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def process_shard(study, shard, context, artifact_dir: Path):
    """刻画一个分片；已存在且校验通过的分片直接复用（幂等，重试不重复纳入）。"""
    idx = shard['index']
    spec = study.spec
    rows = context['validation_rows'][shard['start']:shard['end']]
    checksum = _shard_input_checksum(context, rows, spec)
    path = shard_path(artifact_dir, study.id, idx)
    prior = study.shard_status.get(str(idx)) if study.shard_status else None
    if prior and prior.get('checksum') == checksum and Path(prior.get('file', '')).exists():
        return prior  # 中断恢复：本片已完整落盘，绝不重复计算/重复纳入

    target_label = spec['target_label']
    lo, hi = spec['offset_min'], spec['offset_max']
    # 统一用真实偏移（Hz）持久化：start,end,kind(p/o),flag(1/0)
    encoded = []
    at_lo = 0
    for row_idx in rows:
        fold = context['test_fold'][row_idx]
        scaler, gnb = context['fold_models'][fold]
        target_pos = list(gnb.classes_).index(target_label)
        x = context['X'][row_idx]
        z = scaler.transform(x.reshape(1, -1))[0]
        zc0 = float(z[CENTROID_POS])
        z_other = np.delete(z, CENTROID_POS)
        scale = float(scaler.scale_[CENTROID_POS])
        segments, is_target_lo = sample_segments(
            scaler, gnb, z_other, zc0, target_pos, lo, hi)
        at_lo += 1 if is_target_lo else 0
        for s_t, e_t, kind, flag in segments:
            encoded.append((float(s_t * scale), float(e_t * scale),
                            POINT if kind == 'point' else OPEN,
                            1 if flag else 0))
    # 按 (start, 单点优先于开区间, end) 排序，保证归并流有序
    encoded.sort(key=lambda r: (r[0], 0 if r[2] == POINT else 1, r[1]))

    fd, tmp_name = tempfile.mkstemp(prefix=f'.shard-{idx:05d}-', suffix='.tmp',
                                    dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            for s, e, kind, flag in encoded:
                fh.write(f'{s!r}\t{e!r}\t{kind}\t{flag:d}\n')
            fh.write(f'__META__\t{at_lo:d}\t.\t.\n')
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

    return {'index': idx, 'segments': len(encoded), 'samples': len(rows),
            'at_lo': at_lo, 'checksum': checksum, 'file': str(path)}


def _iter_shard_segments(path: Path):
    """产出 (start, end, kind, flag) 有序分段流；元信息行跳过。"""
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            s, e, kind, flag = line.rstrip('\n').split('\t')
            if s == '__META__':
                continue
            yield float(s), float(e), kind, int(flag)


def _read_shard_at_lo(path: Path) -> int:
    at_lo = 0
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            s, e, kind, flag = line.rstrip('\n').split('\t')
            if s == '__META__':
                at_lo = int(e)
    return at_lo


def _iter_shard_events(path: Path):
    """把分段转成全局有序的扫描线事件流（仅 flag=1 的段），供 heapq.merge 归并。

    - 单点段 (p)：(p, 0, +1)，类别 0 先于开区间事件处理；
    - 开区间 (s,e)：(s, 1, +1) 起点激活、(e, 1, -1) 终点注销，
      激活语义对严格内点 o∈(s,e) 生效（s 不激活、e 已注销）。
    先收集再排序，保证单流按 (offset, category) 全局有序（merge 的前提）。
    """
    events = []
    for s, e, kind, flag in _iter_shard_segments(path):
        if not flag:
            continue
        if kind == POINT:
            events.append((s, 1, 1))
        else:
            # 同偏移先注销终点(类别0)再激活起点(类别2)，点段(类别1)居中，
            # 保证内点计数不含恰在该偏移结束的开区间
            events.append((e, 0, -1))
            events.append((s, 2, 1))
    events.sort()
    for ev in events:
        yield ev


def _grouped_events(study, lo=None, hi=None):
    """heapq 多路归并各片事件，按偏移聚合成 (offset, end_delta, point_add, start_add)。

    end_delta 为开区间终点注销（负数），point_add 为点段增量，start_add 为开区间起点激活。
    贴近区间端点（lo/hi）的浮点误差先吸附到精确端点，避免同一逻辑点被拆成两个键。
    """
    span = EPS_SPAN * max(1.0, abs(lo or 0.0), abs(hi or 0.0))

    def key_of(v):
        if lo is not None and abs(v - lo) <= span:
            return lo
        if hi is not None and abs(v - hi) <= span:
            return hi
        return v

    streams = [_iter_shard_events(Path(study.shard_status[str(sh['index'])]['file']))
               for sh in study.shard_plan]
    cur = None
    end_delta = point_add = start_add = 0
    for offset, category, delta in heapq.merge(*streams):
        offset = key_of(offset)
        if cur is None:
            cur = offset
        if offset != cur:
            yield cur, end_delta, point_add, start_add
            cur, end_delta, point_add, start_add = offset, 0, 0, 0
        if category == 0:
            end_delta += delta
        elif category == 1:
            point_add += delta
        else:
            start_add += delta
    if cur is not None:
        yield cur, end_delta, point_add, start_add


def _union_ranges(ranges, lo, hi):
    """合并相邻/重叠（含数值上几乎相接）的闭区间，输出 [start,end] 列表。"""
    if not ranges:
        return []
    span = max(abs(hi - lo), abs(lo), abs(hi), 1.0) * EPS_SPAN

    def snap(v):  # 把贴近区间端点的浮点误差吸附到精确端点
        if abs(v - lo) <= span:
            return lo
        if abs(v - hi) <= span:
            return hi
        return v

    pts = sorted((snap(float(a)), snap(float(b))) for a, b in ranges)
    merged = []
    cur_a, cur_b = pts[0]
    for a, b in pts[1:]:
        if a <= cur_b + span:
            cur_b = max(cur_b, b)
        else:
            merged.append([cur_a, cur_b])
            cur_a, cur_b = a, b
    merged.append([cur_a, cur_b])
    return merged


def merge_and_summarize(study):
    """线性扫描归并：逐偏移求精确 N(o)（点判读 + 开区间内点），扫描得上下限与全部偏移。

    interior 维护“当前偏移左侧开区间内、满足 s<o<e 的开区间计数”。
    在边界 x：先注销终点 e=x（点 x 不属于该开区间），点判读 N(x)=interior+point_add[x]，
    再激活起点 s=x 供右侧开区间 (x,next) 使用。端点与并列点由 point 段精确承载。
    """
    spec = study.spec
    lo, hi = spec['offset_min'], spec['offset_max']
    shard_status = study.shard_status or {}
    done = {k: v for k, v in shard_status.items() if k != '_meta'}
    if len(done) != len(study.shard_plan):
        raise OffsetStudyError('存在未完成分片，不能归并')
    for shard in study.shard_plan:
        if not Path(shard_status[str(shard['index'])]['file']).exists():
            raise OffsetStudyError(f"分片 {shard['index']} 分段文件缺失，请重试该研究")

    grouped = list(_grouped_events(study, lo, hi))
    at_lo = _read_shard_at_lo_all(study)
    lo_point = sum(pa for x, _end, pa, _st in grouped
                   if abs(x - lo) <= EPS_SPAN * max(1.0, abs(lo), abs(x)))
    # 还未处理任何事件时，覆盖 (lo, 首个边界) 的开区间数
    interior = at_lo - lo_point

    cells = []          # [count, [start, end]]
    prev_x = lo
    first = grouped[0][0] if grouped else lo
    if not grouped or first > lo + EPS_SPAN * max(1.0, abs(lo), abs(first)):
        # lo 无事件或首事件在 lo 右侧：先记录基准开区间段
        cells.append([interior, [lo, first if grouped else hi]])
    for x, end_delta, point_add, start_add in grouped:
        if x > prev_x + EPS_SPAN * max(1.0, abs(x), abs(prev_x)):
            # 开区间 (prev_x, x) 的稳定计数
            cells.append([interior, [prev_x, x]])
        # 点 x：先注销在 x 结束的开区间（x 不是其内点），再加单点段，最后激活新起点。
        # 注意 lo 处的起点激活同样必须执行，否则初始 interior 漏掉从 lo 出发的开区间。
        interior += end_delta          # end_delta 为负
        cells.append([interior + point_add, [x, x]])
        interior += start_add          # 激活从 x 出发的开区间，供 (x, next) 使用
        prev_x = x
    if not cells:
        cells.append([at_lo, [lo, hi]])

    n_min = min(c for c, _ in cells)
    n_max = max(c for c, _ in cells)
    min_ranges = _union_ranges([r for c, r in cells if c == n_min], lo, hi)
    max_ranges = _union_ranges([r for c, r in cells if c == n_max], lo, hi)

    total_samples = sum(int(s['samples']) for s in done.values())
    result = {
        'target_label': spec['target_label'],
        'offset_min': lo, 'offset_max': hi,
        'validation_samples': total_samples,
        'count_at_zero_offset': count_at_offset(study, 0.0) if lo <= 0.0 <= hi else None,
        'minimum': {'count': int(n_min), 'offsets': min_ranges},
        'maximum': {'count': int(n_max), 'offsets': max_ranges},
        'critical_offsets': _count_critical(study, lo, hi),
        'shards': len(study.shard_plan),
        'checksum': shard_status.get('_meta', {}).get('checksum'),
    }
    return result


def _count_critical(study, lo, hi):
    n = 0
    for sh in study.shard_plan:
        for s, e, kind, _ in _iter_shard_segments(
                Path(study.shard_status[str(sh['index'])]['file'])):
            if kind == POINT and lo < s < hi:
                n += 1
    return n


def _read_shard_at_lo_all(study):
    return sum(_read_shard_at_lo(Path(study.shard_status[str(sh['index'])]['file']))
               for sh in study.shard_plan)


def count_at_offset(study, offset):
    """精确复算指定偏移（含端点/并列口径）的判读数，供零偏对照。"""
    if not (study.spec['offset_min'] <= offset <= study.spec['offset_max']):
        return None
    tol = EPS_SPAN * max(1.0, abs(offset))
    total = 0
    for sh in study.shard_plan:
        for s, e, kind, flag in _iter_shard_segments(
                Path(study.shard_status[str(sh['index'])]['file'])):
            if not flag:
                continue
            if kind == POINT:
                if abs(s - offset) <= tol:
                    total += 1
            elif s + tol < offset < e - tol:
                total += 1
    return int(total)


def execute_study(study, run, artifact_dir: Path, on_shard=None):
    """worker 调用的可恢复主流程：逐片刻画并回调记账 → 全部齐活后归并求极值。

    on_shard 每完成一片回调一次，由 worker 提交事务，保证中断重启后已记账分片被跳过。
    """
    spec, plan, context = prepare(run, spec_override=study.spec)
    status = {k: v for k, v in (study.shard_status or {}).items() if k != '_meta'}
    meta = (study.shard_status or {}).get('_meta')
    if meta is None or meta.get('checksum') != context['checksum']:
        status = {}  # 冻结口径变化时作废旧片，避免跨口径分段混入
    for shard in plan:
        info = process_shard(study, shard, context, artifact_dir)
        status[str(shard['index'])] = info
        study.shard_status = {**status, '_meta': {
            'checksum': context['checksum'],
            'validation_samples': len(context['validation_rows'])}}
        if on_shard is not None:
            on_shard(shard['index'], study.shard_status)
    study.shard_status = {**status, '_meta': {
        'checksum': context['checksum'],
        'validation_samples': len(context['validation_rows'])}}
    return merge_and_summarize(study)
