"""零偏研究 HTTP 全生命周期测试（不依赖真实 Redis/MySQL）。

用假队列捕获入队 id，再调用 worker 入口 run_study 模拟出队处理；
验证鉴权、入参校验、分片进度口径（全批算齐前不返回 result）与完成后的上下限。
运行：backend 目录下  python -m tests.test_offset_http
"""
from fastapi.testclient import TestClient

from src.main import app
from src.config.database import Session
from src.models.user import User
from src.models.source import Source
from src.models.record import Record
from src.models.run import Run
from src.models.offset_study import OffsetStudy
from src.services.algorithm import calculate
from src.services import runs as runs_service
from src.services.offset_studies import run_study
from src.utils.security import hash_password
from src.utils.serialization import now
from tests.test_offset_math import make_data


class FakeQueue:
    def __init__(self):
        self.items = []

    def rpush(self, key, value):
        self.items.append((key, int(value)))


def main():
    fake = FakeQueue()
    runs_service.queue = fake  # offset_studies 在函数内 from .runs import queue
    with TestClient(app) as client:
        with Session() as db:
            db.add(User(email='admin@t.demo',
                        password_hash=hash_password('pw'), role='admin'))
            db.add(User(email='viewer@t.demo',
                        password_hash=hash_password('pw'), role='viewer'))
            src = Source(key='k-http', name='批次', filename='f.csv', description='d')
            db.add(src)
            db.flush()
            recs = make_data(90, seed=4)
            for r in recs:
                db.add(Record(source_id=src.id, reference=r['reference'],
                              values=r['values']))
            result = calculate(recs, {'folds': 3, 'smoothing': 1e-9})
            run = Run(source_id=src.id, name='HTTP 研究', state='completed',
                      parameters={'folds': 3, 'smoothing': 1e-9},
                      snapshot={'source': {'filename': 'f.csv'}, 'records': recs},
                      result=result, created_at=now())
            db.add(run)
            db.commit()
            run_id = run.id

        admin = client.post('/auth/login',
                            json={'email': 'admin@t.demo', 'password': 'pw'}).json()
        headers = {'Authorization': 'Bearer ' + admin['token']}
        viewer = client.post('/auth/login',
                             json={'email': 'viewer@t.demo', 'password': 'pw'}).json()
        vh = {'Authorization': 'Bearer ' + viewer['token']}

        # 匿名 401、viewer 写 403
        assert client.get('/offset-studies').status_code == 401
        forbidden = client.post('/offset-studies', headers=vh, json={
            'run_id': run_id, 'offset_min': -10, 'offset_max': 10,
            'target_label': '正常'})
        assert forbidden.status_code == 403

        # 非法区间 422
        bad = client.post('/offset-studies', headers=headers, json={
            'run_id': run_id, 'offset_min': 5, 'offset_max': -5,
            'target_label': '正常'})
        assert bad.status_code == 422

        # 合法创建并入假队列
        ok = client.post('/offset-studies', headers=headers, json={
            'run_id': run_id, 'offset_min': -120, 'offset_max': 120,
            'target_label': '轴承磨损', 'shard_size': 30})
        assert ok.status_code == 200, ok.text
        study_id = ok.json()['id']
        assert fake.items[-1] == ('offset:queue', study_id)

        # 完成前列表/详情只给进度，不泄露 result（即不显示上下限）
        pending = client.get(f'/offset-studies/{study_id}', headers=headers).json()
        assert pending['state'] == 'queued'
        assert 'result' not in pending
        assert pending['shard_total'] >= 1 and pending['shard_done'] == 0

        # 模拟 worker 出队处理
        run_study(study_id)
        done = client.get(f'/offset-studies/{study_id}', headers=headers).json()
        assert done['state'] == 'completed', done
        assert done['shard_done'] == done['shard_total']
        r = done['result']
        assert set(['minimum', 'maximum', 'count_at_zero_offset',
                    'validation_samples']).issubset(r)
        assert r['minimum']['count'] <= r['count_at_zero_offset'] <= r['maximum']['count']
        assert r['minimum']['offsets'] and r['maximum']['offsets']
        # 极值偏移都落在请求区间内
        for a, b in r['minimum']['offsets'] + r['maximum']['offsets']:
            assert -120 - 1e-9 <= a <= b <= 120 + 1e-9
        print('HTTP 生命周期通过：研究', study_id,
              'min', r['minimum']['count'], 'max', r['maximum']['count'],
              'zero', r['count_at_zero_offset'], 'shards', r['shards'])
    print('ALL HTTP TESTS PASSED')


if __name__ == '__main__':
    main()
