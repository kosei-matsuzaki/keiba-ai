"""ORM model re-exports.

Import all models here so that Base.metadata is populated when this package is imported.
Alembic env.py imports this module to make autogen aware of all tables.
"""

from db.models.bet_record import BetRecord
from db.models.entry import Entry
from db.models.horse import Horse
from db.models.jockey import Jockey
from db.models.live_odds import LiveOdds
from db.models.model_run import ModelRun
from db.models.payout import Payout
from db.models.race import Race
from db.models.scrape_log import ScrapeLog
from db.models.simulation_run import SimulationRun
from db.models.trainer import Trainer

__all__ = [
    "BetRecord",
    "Entry",
    "Horse",
    "Jockey",
    "LiveOdds",
    "ModelRun",
    "Payout",
    "Race",
    "ScrapeLog",
    "SimulationRun",
    "Trainer",
]
