"""NN artifact (preprocessor.pkl / temperature_scaler.pkl) の旧パス互換ロード。

pickle 化済み artifact にはクラスのモジュールパスが埋め込まれているため、
refactor でクラスが移設されると再学習なしの旧 .pkl が読めなくなる。本モジュールは
以下を透過的に吸収する Unpickler を提供する:

1. `keiba_ai.*` — 旧 monorepo パスのプレフィクス除去
2. `_MODULE_REMAP` — クラス移設に伴う旧 import パス -> 現行 import パスのリマップ

リマップは「新パスを試し、無ければ旧パスにフォールバック」する寛容な実装のため、
移設の前後どちらの時点でも (旧 .pkl / 新 .pkl のどちらでも) 正しく解決できる。
クラスを移設したら `_MODULE_REMAP` に旧→新を追記すること。
"""

from __future__ import annotations

import pickle
from typing import IO

# 旧 import パス -> 現行 import パス。pickle 化される artifact は
# NNPreprocessor (ai.nn.preprocess) と TemperatureScaler (ai.temperature) のみ。
# model.pt は torch の state_dict (テンソルのみ・クラスパスを持たない) なので対象外。
_MODULE_REMAP = {
    "ai.temperature": "ai.core.temperature",
    "ai.nn.preprocess": "ai.model.preprocess",
    "ai.nn.model": "ai.model.net",
}


class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "keiba_ai":
            raise ImportError("keiba_ai は廃止済みパッケージです")
        if module.startswith("keiba_ai."):
            module = module[len("keiba_ai."):]
        remapped = _MODULE_REMAP.get(module)
        if remapped is not None:
            try:
                return super().find_class(remapped, name)
            except (ImportError, AttributeError):
                # 新パス未配置 (移設前) なら旧パスにフォールバック
                pass
        return super().find_class(module, name)


def legacy_pickle_load(fp: IO[bytes]) -> object:
    """旧パス対応の pickle.load ラッパー。"""
    return LegacyUnpickler(fp).load()
