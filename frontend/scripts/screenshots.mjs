/**
 * README 用の画面キャプチャを撮り直す。
 *
 *   node scripts/screenshots.mjs            # 全部
 *   node scripts/screenshots.mjs race-list  # 1 枚だけ
 *
 * 前提: `bash scripts/dev.sh` 相当で Vite(:5173) と FastAPI(:8765) が動いていること。
 * **データが要る** — 取込済みの DB と学習済みモデルが無いと空の画面が撮れる。
 *
 * playwright は入れていない。**Windows に元からある Chrome / Edge を CDP で
 * 直接叩く**ので追加インストールが要らない (この環境は Norton の TLS 傍受で
 * pypi / npm への到達が不安定 — CLAUDE.local.md)。`ws` は vite の依存から借りる。
 *
 * 撮るだけでなく**押せる**のが要点。レース詳細は「予想を見る」を押さないと
 * AI の列が空のままで、製品として意味のない絵になる。
 */
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const WebSocket = require('./../node_modules/.pnpm/ws@8.20.0/node_modules/ws');

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(HERE, '../../docs/images');
const BASE = process.env.KEIBA_UI_BASE ?? 'http://localhost:5173';
const PORT = 9222;
/** 縦に長すぎる絵は README で読まれないので切る。 */
const MAX_HEIGHT = 2000;

const BROWSERS = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
];

/**
 * 撮る対象。`ready` は「この式が true になるまで待つ」、`click` は撮る前に押す。
 * 日付とレースは**データがある日**に合わせて直す (固定しているのは、撮り直しても
 * 同じ絵が出るようにするため)。
 */
const SHOTS = [
  {
    name: 'race-list',
    path: '/race?date=2026-09-06',
    width: 1440,
    ready: `document.querySelectorAll('table tbody tr').length > 10`,
  },
  {
    name: 'race-detail',
    path: '/race/202609040211?date=2026-09-06',
    width: 1440,
    // 「予想を見る」を押さないと AI の列が空のまま。押した直後は skeleton なので、
    // **消えるまで待つ** (1 レース十数秒かかる)。
    click: `[...document.querySelectorAll('button')].find((b) => b.textContent.includes('予想を見る'))`,
    ready: `document.body.innerText.includes('推奨買目') && document.querySelectorAll('table tbody tr').length > 10`,
    settle: 2500,
  },
  { name: 'models', path: '/models', width: 1440, ready: `document.querySelectorAll('table tbody tr').length > 1` },
  // 購入明細が全期間ぶん伸びて実高 2000px になる。図版としては縦長すぎるので、
  // 内訳の表までで切る (指標カード・損益推移・内訳が入り、ちょうど表の下端で終わる)。
  { name: 'ledger', path: '/ledger', width: 1440, maxHeight: 1140 },
  { name: '_chart', path: '/ledger', width: 1440,
    click: `[...document.querySelectorAll('button')].find((b) => b.textContent.trim() === '全期間')`,
    settle: 2500 },
];

// ── CDP ───────────────────────────────────────────────────────────────────

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function browserWs() {
  for (let i = 0; i < 40; i += 1) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/json/version`);
      return (await r.json()).webSocketDebuggerUrl;
    } catch {
      await sleep(250);
    }
  }
  throw new Error('Chrome の CDP に繋がらない');
}

function connect(url) {
  const ws = new WebSocket(url, { maxPayload: 256 * 1024 * 1024 });
  const pending = new Map();
  let id = 0;
  const ready = new Promise((res, rej) => {
    ws.on('open', res);
    ws.on('error', rej);
  });
  ws.on('message', (raw) => {
    const msg = JSON.parse(raw);
    if (msg.id && pending.has(msg.id)) {
      const { res, rej } = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? rej(new Error(JSON.stringify(msg.error))) : res(msg.result);
    }
  });
  const send = (method, params = {}, sessionId) =>
    new Promise((res, rej) => {
      id += 1;
      pending.set(id, { res, rej });
      ws.send(JSON.stringify({ id, method, params, sessionId }));
    });
  return { ready, send, close: () => ws.close() };
}

/** どの画面でも共通: skeleton が 1 つも残っていないこと。 */
const NO_SKELETON = `document.querySelectorAll('.animate-skeleton-shimmer').length === 0`;

/** 式が true になるまで待つ。false のままでも撮る (空の画面も情報なので)。 */
async function waitFor(send, sid, expr, timeout = 40000) {
  const until = Date.now() + timeout;
  while (Date.now() < until) {
    const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true }, sid);
    if (r.result?.value) return true;
    await sleep(300);
  }
  return false;
}

async function main() {
  const only = process.argv[2];
  const targets = only ? SHOTS.filter((s) => s.name === only) : SHOTS;
  if (targets.length === 0) throw new Error(`知らない対象: ${only}`);

  const exe = BROWSERS.find((p) => existsSync(p));
  if (!exe) throw new Error('Chrome も Edge も見つからない');
  mkdirSync(OUT, { recursive: true });

  const chrome = spawn(exe, [
    '--headless=new',
    '--disable-gpu',
    '--hide-scrollbars',
    `--remote-debugging-port=${PORT}`,
    '--user-data-dir=' + resolve(HERE, '../node_modules/.cache/shoot-profile'),
    'about:blank',
  ]);
  chrome.on('error', (e) => console.error(e));

  try {
    const { ready, send, close } = connect(await browserWs());
    await ready;

    for (const shot of targets) {
      const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
      const { sessionId: sid } = await send('Target.attachToTarget', { targetId, flatten: true });
      await send('Page.enable', {}, sid);
      await send('Emulation.setDeviceMetricsOverride', {
        width: shot.width,
        height: 900,
        deviceScaleFactor: 1,
        mobile: false,
      }, sid);

      await send('Page.navigate', { url: BASE + shot.path }, sid);
      await sleep(2500);
      if (shot.click) {
        await send('Runtime.evaluate', { expression: `(${shot.click})?.click()` }, sid);
      }
      const cond = shot.ready ? `(${shot.ready}) && (${NO_SKELETON})` : NO_SKELETON;
      if (!(await waitFor(send, sid, cond))) {
        console.warn(`  ! ${shot.name}: 準備の条件が満たされないまま撮る`);
      }

      // 中身が揃ってから実高を測り、**リサイズしてから落ち着かせる**。
      // 逆順にすると recharts の ResponsiveContainer が測り直す途中を撮ってしまい、
      // グラフが axes ごと消えた絵になる。
      //
      // **スクロールするのは `<main>`** — App が `h-screen overflow-hidden` なので
      // `documentElement.scrollHeight` は常に viewport の高さになる。Topbar を足す。
      const h = await send('Runtime.evaluate', {
        expression:
          'Math.ceil((document.querySelector("main")?.scrollHeight ?? 0)' +
          ' + (document.querySelector("header")?.offsetHeight ?? 0))',
        returnByValue: true,
      }, sid);
      const height = Math.min(Math.max(h.result?.value ?? 900, 600), shot.maxHeight ?? MAX_HEIGHT);
      await send('Emulation.setDeviceMetricsOverride', {
        width: shot.width,
        height,
        deviceScaleFactor: 1,
        mobile: false,
      }, sid);
      // recharts のアニメーション (既定 1.5s) と再測定が終わるまで待つ。
      await sleep(shot.settle ?? 2500);

      const { data } = await send('Page.captureScreenshot', {
        format: 'png',
        captureBeyondViewport: true,
        optimizeForSpeed: false,
      }, sid);
      const file = resolve(OUT, `${shot.name}.png`);
      writeFileSync(file, Buffer.from(data, 'base64'));
      console.log(`  ${shot.name}.png  ${shot.width}x${height}  ${(Buffer.from(data, 'base64').length / 1024) | 0} KB`);
      await send('Target.closeTarget', { targetId });
    }
    close();
  } finally {
    chrome.kill();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
