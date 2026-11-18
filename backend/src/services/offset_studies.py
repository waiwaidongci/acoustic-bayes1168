from fastapi import HTTPException
from sqlalchemy import select

from ..config.settings import ARTIFACT_DIR
from ..models.offset_study import OffsetStudy
from ..models.run import Run
from ..utils.serialization import now, row
from .audit import log
from .offset_math import OffsetStudyError, validate_study_input
from .offset_pipeline import execute_study
from .runs import get_run


def _study_or_404(db, study_id):
    s = db.get(OffsetStudy, study_id)
    if not s:
        raise HTTPException(404, '未找到零偏偏移研究')
    return s


def list_studies(db):
    """旧研究读取口径不变：列表不返回分段/结果大字段，仅账册信息。"""
    out = []
    for s in db.scalars(select(OffsetStudy).order_by(OffsetStudy.id.desc())):
        d = {k: v for k, v in row(s).items() if k not in ('shard_plan', 'shard_status')}
        d['shard_total'] = len(s.shard_plan or [])
        d['shard_done'] = len([k for k in (s.shard_status or {}) if k != '_meta'])
        out.append(d)
    return out


def create_study(db, user, run_id, payload):
    run = get_run(db, run_id)
    data = payload.model_dump(exclude_none=True) if hasattr(payload, 'model_dump') else dict(payload)
    data.pop('run_id', None)
    try:
        spec = validate_study_input(data)
    except OffsetStudyError as e:
        raise HTTPException(422, str(e))
    try:
        from .offset_pipeline import prepare
        # 入队前先校验并构建分片计划（不做任何样本重判）
        spec, plan, _ = prepare(run, data)
    except OffsetStudyError as e:
        raise HTTPException(422, str(e))
    study = OffsetStudy(
        run_id=run.id,
        name=str(data.get('name') or f'{run.name}-零偏偏移分析')[:160],
        state='queued', spec=spec, shard_plan=plan, shard_status={},
        created_at=now(), updated_at=now())
    db.add(study)
    db.flush()
    log(db, user, 'offset_study.queued', study.id, study.name)
    db.commit()
    try:
        from .runs import queue
        queue.rpush('offset:queue', study.id)
    except Exception:
        study.state = 'failed'
        study.error = '队列暂不可用，请重新创建零偏研究'
        db.commit()
        raise HTTPException(503, study.error)
    return row(study)


def get_study(db, study_id):
    """页面在全批算齐前不显示数量上下限：未完成只回进度，完成才回 result。"""
    s = _study_or_404(db, study_id)
    d = {k: v for k, v in row(s).items() if k not in ('shard_plan', 'shard_status')}
    d['shard_total'] = len(s.shard_plan or [])
    d['shard_done'] = len([k for k in (s.shard_status or {}) if k != '_meta'])
    if s.state != 'completed':
        d.pop('result', None)
    return d


def run_study(study_id):
    """worker 入口：可恢复地逐片刻画并归并；每片提交一次事务。"""
    from ..config.database import Session
    with Session() as db:
        study = db.get(OffsetStudy, study_id)
        if not study or study.state not in ('queued', 'running'):
            return
        run = db.get(Run, study.run_id)
        if not run:
            study.state = 'failed'
            study.error = '关联分析任务不存在'
            db.commit()
            return
        study.state = 'running'
        study.updated_at = now()
        db.commit()

        def on_shard(index, status):
            study.updated_at = now()
            study.shard_status = status
            db.commit()  # 每片落账，中断重启后已完成片被幂等跳过

        try:
            result = execute_study(study, run, ARTIFACT_DIR, on_shard=on_shard)
            study.result = result
            study.state = 'completed'
            study.updated_at = now()
            log(db, 'worker', 'offset_study.completed', study.id)
        except OffsetStudyError as e:
            study.state = 'failed'
            study.error = str(e)[:2000]
            study.updated_at = now()
            log(db, 'worker', 'offset_study.failed', study.id, study.error)
        except Exception as e:  # noqa: BLE001 - worker 需兜底保留失败态
            import logging
            logging.exception('offset study failed')
            study.state = 'failed'
            study.error = str(e)[:2000]
            study.updated_at = now()
            log(db, 'worker', 'offset_study.failed', study.id, study.error)
        db.commit()
