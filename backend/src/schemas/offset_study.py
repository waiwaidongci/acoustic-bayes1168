from pydantic import BaseModel, Field


class OffsetStudyInput(BaseModel):
    run_id: int
    name: str | None = Field(default=None, max_length=160)
    offset_min: float
    offset_max: float
    target_label: str = Field(min_length=1, max_length=600)
    shard_size: int | None = None
