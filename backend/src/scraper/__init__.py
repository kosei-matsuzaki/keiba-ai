"""netkeiba スクレイパー。

import 時に TLS 環境ワークアラウンドを適用する。TLS 傍受を行うアンチウイルス
(Norton 等) が仕込む ``SSLKEYLOGFILE`` はプロセスを abort させるため、
例外として拾えず、HTTPS を張る entry point (API / 各 ingest CLI) のどれか 1 つで
漏らすだけで落ちる。個々の呼び出し側ではなくここに集約している。
詳細は :mod:`core.tls` を参照。
"""

from __future__ import annotations

from core.tls import install_tls_workarounds

install_tls_workarounds()
