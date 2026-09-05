import type { Config } from 'tailwindcss';

export default {
  darkMode: ['class'],
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    // fontSize は extend ではなく **丸ごと差し替え**。規定の 5 段
    // (docs/design.md「字の尺度」) 以外を書けなくするための仕掛けで、
    // text-base / text-xl / text-2xl はクラスごと生成されなくなる。
    // 5 段目の 26px は .text-kpi (globals.css) — 等幅・太さ・字間まで含む型なので
    // サイズだけのユーティリティにはしない。
    // borderRadius も extend ではなく差し替え。角丸は 2px の 1 値だけで、
    // ピル (バッジ) の rounded-full が唯一の例外 (docs/design.md「角丸」)。
    // extend に置くと Tailwind 既定の rounded-lg (8px) 等が生き残り、
    // 名前だけ違う中間値が生えてくる。DEFAULT も 2px に寄せて、
    // 素の `rounded` が別の値を指す抜け道を塞ぐ。
    // shadcn 既定の引き算スケール (sm = radius - 4px) を使わないのは、
    // --radius が 2px だと calc(2px - 4px) が負値になり指定ごと無効になるため。
    borderRadius: {
      none: '0',
      DEFAULT: 'var(--radius)',
      sm: 'var(--radius)',
      md: 'var(--radius)',
      full: '9999px',
    },
    fontSize: {
      '2xs': ['0.6875rem', { lineHeight: '1.35' }], // 11px ラベル・単位・バッジ
      xs: ['0.75rem', { lineHeight: '1.5' }],       // 12px 表のセル
      sm: ['0.875rem', { lineHeight: '1.5' }],      // 14px 本文 (body 既定)
      lg: ['1.125rem', { lineHeight: '1.3' }],      // 18px 見出し
    },
    extend: {
      colors: {
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        // --subtle-foreground は globals.css に定義済みだが Tailwind 側に
        // 露出していなかったため text-subtle-foreground が効いていなかった。
        subtle: {
          foreground: 'hsl(var(--subtle-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
          elevated: 'hsl(var(--card-elevated))',
        },
        border: {
          DEFAULT: 'hsl(var(--border))',
          strong: 'hsl(var(--border-strong))',
        },
        ring: 'hsl(var(--ring))',
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
        success: {
          DEFAULT: 'hsl(var(--success))',
          foreground: 'hsl(var(--success-foreground))',
        },
        warning: {
          DEFAULT: 'hsl(var(--warning))',
          foreground: 'hsl(var(--warning-foreground))',
        },
        info: {
          DEFAULT: 'hsl(var(--info))',
          foreground: 'hsl(var(--info-foreground))',
        },
      },
      fontFamily: {
        sans: ['var(--font-sans)'],
        mono: ['var(--font-mono)'],
      },
      keyframes: {
        // Skeleton 用の柔らかい fade — Tailwind デフォルトの animate-pulse は
        // 0.5 → 1.0 → 0.5 で目立つので、0.6 → 1.0 → 0.6 に範囲を狭めて
        // duration も少し長めに (1.8s) する。
        'skeleton-shimmer': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.6' },
        },
      },
      animation: {
        'skeleton-shimmer': 'skeleton-shimmer 1.8s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
} satisfies Config;
