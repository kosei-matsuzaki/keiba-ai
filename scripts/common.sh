#!/usr/bin/env bash
# 共通シェル関数。各 build_*.sh / dev.sh / seed_sample.sh から `source` して読み込む。
# ── path 解決と OS 判定をここに集約し、子スクリプトの定型句を排除する。

# このファイルを source した呼び出し側で SCRIPT_DIR/GAME_DIR/BACKEND_DIR を使えるよう、
# 呼び出し元のスクリプトパスから派生させる。
# - source 経由なので $0 はインタラクティブシェル名（bash 等）になり得るため、
#   呼び出し元は事前に SCRIPT_PATH=$(readlink -f "$0") を渡すか、関数で計算する。

# 呼び出し元のスクリプトディレクトリを返す。引数 1 に呼び出し元 $0 を渡す。
script_dir_of() {
    dirname "$(readlink -f "$1")"
}

# 実行環境が Windows (Git Bash / MSYS / Cygwin) かどうか。
is_windows() {
    [[ "$(uname -s)" == *MINGW* ]] || [[ "$(uname -s)" == *CYGWIN* ]] || \
    [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "win"* ]]
}

# rustc が見つかればその host triple を返す。無ければ uname から推定する。
detect_target_triple() {
    if command -v rustc &>/dev/null; then
        rustc -vV | sed -n 's/host: //p'
        return
    fi
    local os arch
    os=$(uname -s)
    arch=$(uname -m)
    case "$os" in
        Linux*)  echo "${arch}-unknown-linux-gnu" ;;
        Darwin*) echo "${arch}-apple-darwin" ;;
        MINGW*|MSYS*|CYGWIN*) echo "${arch}-pc-windows-msvc" ;;
        *) echo "${arch}-unknown-unknown" ;;
    esac
}
