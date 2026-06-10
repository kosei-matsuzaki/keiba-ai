"""master テーブル (horse / jockey / trainer) の共通 upsert ヘルパー。

jobs/ingest.py（結果 ingest）と jobs/ingest_shutuba.py（出馬表 ingest）が同じ
on_conflict_do_update パターンを重複保持していたのを 1 箇所に集約したもの。

いずれも既存値を COALESCE で保護する（新しい値が NULL のときは既存値を残す）。
horse は呼び出し側が name のみ／detail 込み（sex・birth_date・sire・dam）の双方を
渡せる。値を渡さなかったフィールドの set_ は coalesce(NULL, 既存) = 既存 の no-op。
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from db.models.horse import Horse
from db.models.jockey import Jockey
from db.models.trainer import Trainer


def upsert_horse(session: Session, horse_kwargs: dict[str, object]) -> None:
    """horse を upsert する。

    horse_kwargs は最低限 horse_id / name を含む。detail fetch 済みなら
    sex / birth_date / sire / dam も渡せる（未指定フィールドは既存値を保持）。
    """
    stmt = sqlite_insert(Horse).values(**horse_kwargs)
    stmt = stmt.on_conflict_do_update(
        index_elements=["horse_id"],
        set_={
            "name": sa.func.coalesce(stmt.excluded.name, Horse.name),
            "sex": sa.func.coalesce(stmt.excluded.sex, Horse.sex),
            "birth_date": sa.func.coalesce(stmt.excluded.birth_date, Horse.birth_date),
            "sire": sa.func.coalesce(stmt.excluded.sire, Horse.sire),
            "dam": sa.func.coalesce(stmt.excluded.dam, Horse.dam),
        },
    )
    session.execute(stmt)


def upsert_jockey(session: Session, jockey_id: str, name: str | None) -> None:
    """jockey を upsert する（name は既存値を COALESCE で保護）。"""
    stmt = sqlite_insert(Jockey).values(jockey_id=jockey_id, name=name)
    stmt = stmt.on_conflict_do_update(
        index_elements=["jockey_id"],
        set_={"name": sa.func.coalesce(stmt.excluded.name, Jockey.name)},
    )
    session.execute(stmt)


def upsert_trainer(session: Session, trainer_id: str, name: str | None) -> None:
    """trainer を upsert する（name は既存値を COALESCE で保護）。"""
    stmt = sqlite_insert(Trainer).values(trainer_id=trainer_id, name=name)
    stmt = stmt.on_conflict_do_update(
        index_elements=["trainer_id"],
        set_={"name": sa.func.coalesce(stmt.excluded.name, Trainer.name)},
    )
    session.execute(stmt)
