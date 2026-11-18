from sqlalchemy import Column,Integer,String,JSON,Text,ForeignKey
from ..config.database import Base
from ..config.settings import CONFIG
class OffsetStudy(Base):
    """零偏偏移区间研究：挂在已完成的 ClassifierRun 上，冻结偏移区间与目标故障。"""
    __tablename__='offset_studies'
    id=Column(Integer,primary_key=True)
    run_id=Column(Integer,ForeignKey(CONFIG['run_table']+'.id'),nullable=False,index=True)
    name=Column(String(160),nullable=False)
    state=Column(String(20),default='queued',index=True)  # queued/running/completed/failed
    spec=Column(JSON,nullable=False)       # offset_min/offset_max/target_label/shard_size
    shard_plan=Column(JSON,nullable=False) # [{index,start,end}]
    shard_status=Column(JSON,nullable=False)# {index: {events, samples, file, checksum}}
    result=Column(JSON,nullable=True)
    error=Column(Text,default='')
    created_at=Column(String(40),nullable=False)
    updated_at=Column(String(40),nullable=False)
