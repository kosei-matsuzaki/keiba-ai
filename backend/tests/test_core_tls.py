"""TLS 傍受アンチウイルス環境向けワークアラウンドのテスト。

背景は :mod:`core.tls` の docstring を参照。
"""

from __future__ import annotations

import ssl

import pytest

from core import tls


@pytest.fixture(autouse=True)
def _reset_install_state(monkeypatch: pytest.MonkeyPatch):
    """install_tls_workarounds() の冪等フラグと ssl のパッチをテストごとに戻す。"""
    monkeypatch.setattr(tls, "_installed", False)
    monkeypatch.setattr(ssl, "create_default_context", tls._original_create_default_context)
    monkeypatch.setattr(
        ssl, "_create_default_https_context", tls._original_create_default_context
    )
    yield


class TestSanitizeSslKeylogEnv:
    def test_removes_device_path(self, monkeypatch: pytest.MonkeyPatch):
        """Norton が注入するデバイスパスは除去する (残すとプロセスが abort する)。"""
        injected = r"\\.\nllMonFltProxy\40b050b03e676c72"
        monkeypatch.setenv("SSLKEYLOGFILE", injected)

        removed = tls.sanitize_ssl_keylog_env()

        assert removed == injected
        assert "SSLKEYLOGFILE" not in __import__("os").environ

    def test_keeps_regular_file_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path):
        """TLS デバッグ用の正規の keylog ファイル指定は壊さない。"""
        keylog = str(tmp_path / "keylog.txt")
        monkeypatch.setenv("SSLKEYLOGFILE", keylog)

        assert tls.sanitize_ssl_keylog_env() is None
        assert __import__("os").environ["SSLKEYLOGFILE"] == keylog

    def test_noop_when_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("SSLKEYLOGFILE", raising=False)
        assert tls.sanitize_ssl_keylog_env() is None


class TestRelaxStrictEnabled:
    @pytest.mark.parametrize("value", ["1", "true", "yes"])
    def test_enabled_values(self, monkeypatch: pytest.MonkeyPatch, value: str):
        monkeypatch.setenv(tls.RELAX_STRICT_ENV, value)
        assert tls.relax_strict_enabled() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "False"])
    def test_disabled_values(self, monkeypatch: pytest.MonkeyPatch, value: str):
        monkeypatch.setenv(tls.RELAX_STRICT_ENV, value)
        assert tls.relax_strict_enabled() is False

    def test_default_is_disabled(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(tls.RELAX_STRICT_ENV, raising=False)
        assert tls.relax_strict_enabled() is False


class TestRelaxedSslContext:
    def test_clears_only_x509_strict(self):
        baseline = tls._original_create_default_context()
        relaxed = tls.relaxed_ssl_context()

        assert not relaxed.verify_flags & ssl.VERIFY_X509_STRICT
        # 他の検証は据え置き: チェーン検証もホスト名検証も残る
        assert relaxed.verify_mode == baseline.verify_mode == ssl.CERT_REQUIRED
        assert relaxed.check_hostname is baseline.check_hostname is True
        # X509_STRICT 以外のフラグは元のまま
        other_flags = baseline.verify_flags & ~ssl.VERIFY_X509_STRICT
        assert relaxed.verify_flags == other_flags


class TestInstallTlsWorkarounds:
    def test_does_not_patch_ssl_by_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(tls.RELAX_STRICT_ENV, raising=False)

        tls.install_tls_workarounds()

        assert ssl.create_default_context is tls._original_create_default_context
        assert ssl.create_default_context().verify_flags & ssl.VERIFY_X509_STRICT

    def test_patches_ssl_when_enabled(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(tls.RELAX_STRICT_ENV, "1")

        tls.install_tls_workarounds()

        assert ssl.create_default_context is tls.relaxed_ssl_context
        assert ssl._create_default_https_context is tls.relaxed_ssl_context
        assert not ssl.create_default_context().verify_flags & ssl.VERIFY_X509_STRICT

    def test_sanitizes_keylog_even_when_relax_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """crash guard は opt-in と無関係に常に効く。"""
        monkeypatch.delenv(tls.RELAX_STRICT_ENV, raising=False)
        monkeypatch.setenv("SSLKEYLOGFILE", r"\\.\nllMonFltProxy\deadbeef")

        tls.install_tls_workarounds()

        assert "SSLKEYLOGFILE" not in __import__("os").environ

    def test_is_idempotent(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(tls.RELAX_STRICT_ENV, "1")

        tls.install_tls_workarounds()
        # 2 回目でパッチ済みの relaxed_ssl_context を _original として掴み直さない
        tls.install_tls_workarounds()

        assert tls._original_create_default_context is not tls.relaxed_ssl_context
        assert ssl.create_default_context().verify_mode == ssl.CERT_REQUIRED
