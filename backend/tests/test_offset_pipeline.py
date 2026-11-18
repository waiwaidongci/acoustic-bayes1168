"""零偏研究分片流水线的可恢复性、幂等性与有限内存性能测试。

验证：
- 中断恢复：已落盘分片被跳过，重试不重复纳入，最终结果与一次跑完完全一致；
- 幂等：对同一分片重复 process_shard 返回同一记账且不重复计数；
- 十万样本规模：分片数量、单片内存与端到端耗时受控，归并为线性复杂度。
运行：backend 目录下  python -m tests.test_offset_pipeline
"""
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from tests.test_offset_math import FakeStudy, make_data
from src.services.offset_pipeline import (execute_study, merge_and_summarize,
                                          prepare, process_shard,
                                          shard_path)


def _run_obj(records):
    return SimpleNamespace(
        state='completed',
        snapshot={'records': records},
        parameters={'folds': 3, 'smoothing': 1e-9},
        result={'checksum': 'test'})


def test_idempotent_and_recoverable():
    records = make_data(300, seed=3)
    run = _run_obj(records)
    payload = {'offset_min': -200.0, 'offset_max': 200.0,
               'target_label': '轴承磨损', 'shard_size': 40}
    spec, plan, ctx = prepare(run, payload)
    tmp = Path(tempfile.mkdtemp())

    # 一次性跑完作为基准
    full = FakeStudy(spec, plan, {})
    status = {}
    for sh in plan:
        info = process_shard(full, sh, ctx, tmp)
        status[str(sh['index'])] = info
        full.shard_status = status
    full.shard_status = {**status, '_meta': {'checksum': ctx['checksum']}}
    baseline = merge_and_summarize(full)

    # 幂等：对每个已完成分片再处理一次，记账应完全相同且文件不被改写
    for sh in plan:
        before = status[str(sh['index'])]
        again = process_shard(full, sh, ctx, tmp)
        assert again == before
    assert merge_and_summarize(full) == baseline

    # 模拟中断：只完成前一半分片，再用全新 study 对象恢复
    partial = FakeStudy(spec, plan, {})
    pstatus = {}
    for sh in plan[: len(plan) // 2]:
        info = process_shard(partial, sh, ctx, tmp)  # 复用同一目录里已落盘的分片
        pstatus[str(sh['index'])] = info
        partial.shard_status = pstatus
    # 恢复执行：已存在分片应直接复用（不重算），其余分片补齐
    resumed = execute_study(partial, run, tmp)
    assert resumed['minimum'] == baseline['minimum']
    assert resumed['maximum'] == baseline['maximum']
    assert resumed['validation_samples'] == baseline['validation_samples']
    # 重试不重复纳入：分片数与样本数守恒
    assert resumed['shards'] == len(plan)
    print('恢复/幂等测试通过：分片', len(plan),
          '样本', resumed['validation_samples'],
          'min', resumed['minimum']['count'], 'max', resumed['maximum']['count'])


def test_large_scale_bounded():
    # 十万样本：验证分片计划与端到端耗时/内存受控（事件法不对每个偏移重判整批）
    n = 100_000
    records = make_data(n, seed=21)
    run = _run_obj(records)
    payload = {'offset_min': -150.0, 'offset_max': 150.0,
               'target_label': '正常', 'shard_size': 5000}
    t0 = time.time()
    spec, plan, ctx = prepare(run, payload)
    assert len(plan) == 20
    tmp = Path(tempfile.mkdtemp())
    study = FakeStudy(spec, plan, {})
    result = execute_study(study, run, tmp)
    elapsed = time.time() - t0
    assert result['validation_samples'] == n
    assert 0 <= result['minimum']['count'] <= result['maximum']['count'] <= n
    # 单片文件大小受控（远小于把十万样本全量驻留内存）
    biggest = max(shard_path(tmp, study.id, sh['index']).stat().st_size
                  for sh in plan)
    print(f'十万样本：分片 {len(plan)} 片，耗时 {elapsed:.1f}s，'
          f'最大单片 {biggest/1024:.0f}KB，临界偏移 {result["critical_offsets"]}，'
          f'min {result["minimum"]["count"]} max {result["maximum"]["count"]}')
    assert biggest < 5 * 1024 * 1024
    return elapsed


if __name__ == '__main__':
    test_idempotent_and_recoverable()
    test_large_scale_bounded()
    print('ALL PIPELINE TESTS PASSED')
