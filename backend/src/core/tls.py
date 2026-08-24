"""TLS 傍受アンチウイルス環境で HTTPS を通すためのワークアラウンド。

Norton / Avast 等の「暗号化された接続のスキャン」を有効にした Windows では、
Python の HTTPS が 2 段階で壊れる。どちらもアプリのバグではなく環境要因だが、
1 つ目は **プロセスが即死する** ためライブラリ側で握り潰すことができない。

1. ``SSLKEYLOGFILE`` にデバイスパスが仕込まれる
   Norton は TLS を復号するため、プロセス環境に次のような値を注入する::

       SSLKEYLOGFILE=\\\\.\\nllMonFltProxy\\40b050b03e676c72

   ``ssl.create_default_context()`` はこの環境変数を見て keylog ファイルを
   開こうとする。相手は通常ファイルではなくフィルタドライバのデバイスパスで、
   さらに Norton は自前の MSVC ランタイム (``Norton\\Suite\\local.crt\\
   VCRUNTIME140.dll``) をプロセスに注入するため、OpenSSL のランタイム不整合
   検出に引っかかって ``OPENSSL_Uplink(...,08): no OPENSSL_Applink`` で
   **プロセスごと abort する**。例外ではないので try/except では拾えない。

2. 傍受用の CA 証明書が OpenSSL 3.5 の厳格チェックに落ちる
   Norton が差し替える証明書チェーンのルート
   (``CN=Norton Web/Mail Shield Root``) は basicConstraints が critical
   指定されておらず、Python 3.13 が既定で有効にする ``VERIFY_X509_STRICT``
   に違反する::

       CERTIFICATE_VERIFY_FAILED: Basic Constraints of CA cert not marked critical

本来はアンチウイルス側で SSL スキャンを切る (もしくは対象ドメインを除外する)
のが筋で、その場合ここは何もしなくてよい。それができない環境向けに、
``KEIBA_TLS_RELAX_STRICT=1`` で 2 の緩和を **明示的に opt-in** できるように
してある (既定は無効)。

緩和されるのは ``VERIFY_X509_STRICT`` (Python 3.13 で既定になった追加の
厳格チェック) だけで、証明書チェーンの検証・ホスト名検証・有効期限チェックは
そのまま残る。実質 Python 3.12 以前と同じ厳格度に戻すもので、
``verify=False`` のような検証無効化ではない。
"""

from __future__ import annotations

import os
import ssl

from core.logging import get_logger

logger = get_logger(__name__)

RELAX_STRICT_ENV = "KEIBA_TLS_RELAX_STRICT"
_KEYLOG_ENV = "SSLKEYLOGFILE"

# install_tls_workarounds() は複数の entry point から呼ばれうるので冪等にする
_installed = False


def relax_strict_enabled() -> bool:
    """``KEIBA_TLS_RELAX_STRICT`` が有効かどうか。"""
    return os.getenv(RELAX_STRICT_ENV, "") not in ("", "0", "false", "False")


def _is_device_path(value: str) -> bool:
    r"""``\\.\name`` 形式の Windows デバイスパスなら True。

    通常ファイルを指す正規の SSLKEYLOGFILE (TLS デバッグ用途) を壊さないよう、
    デバイスパスのときだけ取り除く。
    """
    normalized = value.replace("/", "\\")
    return normalized.startswith("\\\\.\\")


def sanitize_ssl_keylog_env() -> str | None:
    """デバイスパスが入った ``SSLKEYLOGFILE`` を環境から取り除く。

    取り除いた値を返す (何もしなければ None)。os.environ の変更はこのプロセス
    内だけに閉じており、システムの環境変数には影響しない。
    """
    value = os.environ.get(_KEYLOG_ENV)
    if not value or not _is_device_path(value):
        return None
    os.environ.pop(_KEYLOG_ENV, None)
    logger.warning(
        "%s=%r はデバイスパス (TLS 傍受アンチウイルスによる注入) のため、"
        "このプロセスの環境から除去しました。残したままだと HTTPS 接続時に "
        "OpenSSL applink エラーでプロセスが異常終了します。",
        _KEYLOG_ENV,
        value,
    )
    return value


def relaxed_ssl_context(*args: object, **kwargs: object) -> ssl.SSLContext:
    """``VERIFY_X509_STRICT`` を外し、OS の証明書ストアも信頼する SSLContext。

    2 点やっている:

    * ``VERIFY_X509_STRICT`` を外す — 傍受用 CA の basicConstraints が
      critical 指定されていないため (Python 3.13 の既定では弾かれる)
    * OS の証明書ストアを追加で読み込む — httpx は既定で certifi のバンドルを
      ``cafile`` に渡すため、Windows 証明書ストアにしか入っていない傍受用
      ルート証明書が見つからず ``unable to get local issuer certificate`` に
      なる。certifi の CA は残したまま、OS が既に信頼している CA を足す。
    """
    context = _original_create_default_context(*args, **kwargs)  # type: ignore[arg-type]
    context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    # cafile 指定があっても OS ストアを併せて信頼する (置き換えではなく追加)
    context.load_default_certs(ssl.Purpose.SERVER_AUTH)
    return context


# パッチ適用前の本来の実装を保持しておく (relaxed_ssl_context から呼ぶ)
_original_create_default_context = ssl.create_default_context


def install_tls_workarounds() -> None:
    """SSLKEYLOGFILE の除去と、opt-in の X509_STRICT 緩和を適用する。

    scraper パッケージの import 時に呼ばれる。HTTPS を張る entry point が
    (API / 各 ingest CLI で) 10 箇所以上あり、1 つ漏らすとプロセス abort に
    なるため、個々の呼び出し側ではなくパッケージ import に集約している。

    冪等。2 回目以降は何もしない。
    """
    global _installed
    if _installed:
        return
    _installed = True

    sanitize_ssl_keylog_env()

    if not relax_strict_enabled():
        return

    # httpx も urllib も既定 context の生成はこの 2 つを経由するため、
    # ここを差し替えれば全 HTTPS 呼び出し側に一律で効く。
    ssl.create_default_context = relaxed_ssl_context  # type: ignore[assignment]
    ssl._create_default_https_context = relaxed_ssl_context  # type: ignore[assignment]
    logger.warning(
        "%s が有効です: TLS 検証から VERIFY_X509_STRICT を外しました "
        "(チェーン検証・ホスト名検証・有効期限チェックは維持)。"
        "TLS 傍受を行うアンチウイルス環境向けの回避策です。",
        RELAX_STRICT_ENV,
    )
