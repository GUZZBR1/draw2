export type ThemeChoice = 'system' | 'light' | 'dark';

export const THEME_KEY = 'draw-cad-theme';
export const THEME_ORDER: ThemeChoice[] = ['system', 'light', 'dark'];
export const THEME_LABEL: Record<ThemeChoice, string> = { system: 'Sistema', light: 'Claro', dark: 'Escuro' };

export function nextTheme(t: ThemeChoice): ThemeChoice {
  return THEME_ORDER[(THEME_ORDER.indexOf(t) + 1) % THEME_ORDER.length];
}

export function parseTheme(value: string | null): ThemeChoice {
  return value === 'light' || value === 'dark' ? value : 'system';
}

/** Luminância relativa (0 = preto, 1 = branco) de uma cor #rrggbb. */
export function luminance(hex: string): number | null {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return null;
  const v = parseInt(m[1], 16);
  const lin = (c: number) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * lin((v >> 16) & 255) + 0.7152 * lin((v >> 8) & 255) + 0.0722 * lin(v & 255);
}

/**
 * Cor de exibição da camada no tema atual, como a cor 7 do AutoCAD:
 * traço quase preto vira quase branco na folha escura (e o inverso
 * na clara). Só a tela muda; o documento e a exportação mantêm a cor.
 */
export function displayColor(hex: string, dark: boolean): string {
  const l = luminance(hex);
  if (l === null) return hex;
  if (dark && l < 0.04) return '#e9ecef';
  if (!dark && l > 0.85) return '#1f2933';
  return hex;
}
