import { useEffect, useRef } from 'react';

const SOURCE_GLYPHS = Array.from("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ");
const SAMPLE_SIZE = 5;

function cssColor(element: Element, token: string, fallback: string) {
  return getComputedStyle(element).getPropertyValue(token).trim() || fallback;
}

function parseRgb(color: string) {
  const hex = color.match(/^#([\da-f]{6})$/i)?.[1];
  if (hex) return [0, 2, 4].map((offset) => Number.parseInt(hex.slice(offset, offset + 2), 16));
  const values = color.match(/[\d.]+/g)?.slice(0, 3).map(Number);
  return values?.length === 3 ? values : [34, 35, 31];
}

function luminance([red, green, blue]: number[]) {
  const channel = (value: number) => {
    const normalized = value / 255;
    return normalized <= 0.04045 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4;
  };
  return channel(red) * 0.2126 + channel(green) * 0.7152 + channel(blue) * 0.0722;
}

function smoothstep(edge0: number, edge1: number, value: number) {
  const progress = Math.max(0, Math.min(1, (value - edge0) / (edge1 - edge0)));
  return progress * progress * (3 - 2 * progress);
}

function eyeMask(x: number, y: number, centerX: number) {
  const distance = Math.hypot((x - centerX) / 0.052, (y - 0.432) / 0.067);
  return 1 - smoothstep(0.72, 1.08, distance);
}

function motionPulse(elapsed: number, start: number, duration: number, period: number) {
  const phase = ((elapsed - start) % period + period) % period;
  if (elapsed < start || phase >= duration) return 0;
  return smoothstep(0, 0.13, phase) * (1 - smoothstep(duration - 0.13, duration, phase));
}

function rankGlyphs(font: string, cellW: number, cellH: number) {
  const measure = document.createElement('canvas');
  measure.width = Math.ceil(cellW);
  measure.height = Math.ceil(cellH);
  const context = measure.getContext('2d', { willReadFrequently: true });
  if (!context) return SOURCE_GLYPHS.map((character, index) => ({ character, darkness: 1 - index / SOURCE_GLYPHS.length }));

  context.font = font;
  context.textBaseline = 'top';
  return SOURCE_GLYPHS.map((character) => {
    context.fillStyle = '#fff';
    context.fillRect(0, 0, measure.width, measure.height);
    context.fillStyle = '#000';
    context.fillText(character, 0, 0);
    const pixels = context.getImageData(0, 0, measure.width, measure.height).data;
    let darkness = 0;
    for (let index = 0; index < pixels.length; index += 4) darkness += 255 - pixels[index];
    return { character, darkness: darkness / (measure.width * measure.height * 255) };
  }).sort((a, b) => b.darkness - a.darkness);
}

function chooseGlyph(glyphs: Array<{ character: string; darkness: number }>, target: number) {
  let low = 0;
  let high = glyphs.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (glyphs[middle].darkness > target) low = middle + 1;
    else high = middle;
  }
  if (low === 0) return glyphs[0];
  const before = glyphs[low - 1];
  const after = glyphs[low];
  return Math.abs(before.darkness - target) < Math.abs(after.darkness - target) ? before : after;
}

function interpolatePixel(pixels: Uint8ClampedArray, width: number, height: number, x: number, y: number) {
  const x0 = Math.max(0, Math.min(width - 1, Math.floor(x)));
  const y0 = Math.max(0, Math.min(height - 1, Math.floor(y)));
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);
  const tx = x - Math.floor(x);
  const ty = y - Math.floor(y);
  const output = [0, 0, 0, 0];
  for (let channel = 0; channel < 4; channel += 1) {
    const top = pixels[(y0 * width + x0) * 4 + channel] * (1 - tx) + pixels[(y0 * width + x1) * 4 + channel] * tx;
    const bottom = pixels[(y1 * width + x0) * 4 + channel] * (1 - tx) + pixels[(y1 * width + x1) * 4 + channel] * tx;
    output[channel] = top * (1 - ty) + bottom * ty;
  }
  return output;
}

function textColor(red: number, green: number, blue: number, ink: string, muted: string, paper: number[], eyeDetail: boolean) {
  const pixel = [red, green, blue];
  const lightBackground = luminance(paper) > 0.5;
  const sourceLightness = luminance(pixel);
  const goldHair = red > green * 1.08 && green > blue * 1.25;
  const blueEye = blue > red * 1.1 && blue > green * 1.08;
  const warmSkin = red > green * 1.06 && green > blue * 1.04 && !goldHair;

  if (blueEye && eyeDetail) return lightBackground ? 'rgb(42, 83, 142)' : 'rgb(136, 181, 241)';
  if (blueEye) return lightBackground ? muted : `rgb(${Math.round(red)}, ${Math.round(green)}, ${Math.round(blue)})`;
  if (sourceLightness < 0.16) return ink;
  if (goldHair && lightBackground) {
    const shade = (value: number) => Math.round(value * 0.68);
    return `rgb(${shade(red)}, ${shade(green)}, ${shade(blue)})`;
  }
  if (warmSkin && lightBackground) return muted;
  if (lightBackground && sourceLightness > 0.88) return muted;
  return `rgb(${Math.round(red)}, ${Math.round(green)}, ${Math.round(blue)})`;
}

export default function VisionAsciiPortrait() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const host = canvas?.parentElement;
    const output = canvas?.getContext('2d');
    if (!canvas || !host || !output) return;

    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    const closedImage = new Image();
    const openImage = new Image();
    closedImage.src = '/images/vision/sailor-moon-eyes-closed.png';
    openImage.src = '/images/vision/sailor-moon-eyes-open.png';
    const source = document.createElement('canvas');
    const sourceContext = source.getContext('2d', { willReadFrequently: true });
    if (!sourceContext) return;

    let ink = '';
    let muted = '';
    let paper = [247, 247, 244];
    let paperLightness = luminance(paper);
    let closedPixels: Uint8ClampedArray | null = null;
    let openPixels: Uint8ClampedArray | null = null;
    let glyphs: Array<{ character: string; darkness: number }> = [];
    let frame = 0;
    let lastPaint = 0;
    let startTime = performance.now();
    let inView = true;
    let disposed = false;
    let cols = 0;
    let rows = 0;
    let cellW = 0;
    let cellH = 0;
    let outputWidth = 0;
    let outputHeight = 0;
    let fontSize = 0;

    const updatePalette = () => {
      ink = cssColor(host, '--ink', '#22231f');
      muted = cssColor(host, '--muted', '#777870');
      paper = parseRgb(cssColor(host, '--paper', '#f7f7f4'));
      paperLightness = luminance(paper);
    };

    const layout = () => {
      const box = canvas.getBoundingClientRect();
      if (!box.width) return;
      fontSize = Math.max(6, Math.min(10, box.width * 0.02));
      output.font = `${fontSize}px "DM Mono", monospace`;
      cellW = output.measureText('M').width;
      cellH = fontSize * 1.12;
      cols = Math.floor(box.width / cellW);
      const aspect = closedImage.naturalWidth && closedImage.naturalHeight
        ? closedImage.naturalWidth / closedImage.naturalHeight
        : 1.1;
      rows = Math.max(1, Math.round((cols * cellW) / (aspect * cellH)));
      outputWidth = box.width;
      outputHeight = rows * cellH;
      canvas.style.height = `${outputHeight}px`;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(outputWidth * dpr);
      canvas.height = Math.round(outputHeight * dpr);
      output.setTransform(dpr, 0, 0, dpr, 0, 0);
      output.font = `${fontSize}px "DM Mono", monospace`;
      output.textBaseline = 'top';
      glyphs = rankGlyphs(output.font, cellW, cellH);

      source.width = cols * SAMPLE_SIZE;
      source.height = rows * SAMPLE_SIZE;
      if (closedImage.complete && openImage.complete && closedImage.naturalWidth && openImage.naturalWidth) {
        sourceContext.clearRect(0, 0, source.width, source.height);
        sourceContext.drawImage(closedImage, 0, 0, source.width, source.height);
        closedPixels = sourceContext.getImageData(0, 0, source.width, source.height).data;
        sourceContext.clearRect(0, 0, source.width, source.height);
        sourceContext.drawImage(openImage, 0, 0, source.width, source.height);
        openPixels = sourceContext.getImageData(0, 0, source.width, source.height).data;
      }
    };

    const draw = (timestamp: number) => {
      frame = 0;
      if (!closedPixels || !openPixels || !glyphs.length) return;
      if (!reducedMotion.matches && timestamp - lastPaint < 45) {
        frame = requestAnimationFrame(draw);
        return;
      }
      lastPaint = timestamp;
      const elapsed = (timestamp - startTime) / 1000;
      const opening = reducedMotion.matches ? 1 : Math.max(0, Math.min(1, (elapsed - 0.35) / 1.05));
      const eyeOpen = opening * opening * (3 - 2 * opening);
      const hairFlow = reducedMotion.matches ? 0 : Math.min(1, smoothstep(1.1, 2.5, elapsed));
      const blink = reducedMotion.matches ? 0 : motionPulse(elapsed, 2.8, 0.34, 5.4);
      const wink = reducedMotion.matches ? 0 : motionPulse(elapsed, 4.2, 0.58, 8.2);
      const errors = new Float32Array(cols * rows);
      const densities = new Float32Array(cols * rows);
      const colors = new Array<string>(cols * rows);
      const paperLum = paperLightness;
      let previousFill = '';

      for (let row = 0; row < rows; row += 1) {
        for (let col = 0; col < cols; col += 1) {
          const horizontal = (col + 0.5) / cols;
          const vertical = (row + 0.5) / rows;
          let red = 0;
          let green = 0;
          let blue = 0;
          let alpha = 0;
          let samples = 0;

          for (let sy = 0; sy < SAMPLE_SIZE; sy += 1) {
            for (let sx = 0; sx < SAMPLE_SIZE; sx += 1) {
              const x = col * SAMPLE_SIZE + sx;
              const y = row * SAMPLE_SIZE + sy;
              const sourceX = (x + 0.5) / source.width;
              const sourceY = (y + 0.5) / source.height;
              const side = sourceX < 0.5 ? -1 : 1;
              const hairBand = Math.max(
                1 - smoothstep(0.16, 0.39, sourceX),
                smoothstep(0.61, 0.84, sourceX),
              ) * smoothstep(0.34, 0.53, sourceY) * (1 - smoothstep(0.94, 1, sourceY));
              const wave = Math.sin(elapsed * 0.78 - sourceY * 17 + side * 0.6) * 0.5
                + Math.sin(elapsed * 1.13 - sourceY * 29 - side * 0.9) * 0.3
                + Math.sin(elapsed * 0.49 - sourceY * 9 + side * 1.8) * 0.2;
              const hairShift = reducedMotion.matches ? 0 : wave * hairBand * hairFlow * source.width * 0.03;
              const sampleX = Math.max(0, Math.min(source.width - 1, x + hairShift));
              const closed = interpolatePixel(closedPixels, source.width, source.height, sampleX, y);
              const open = interpolatePixel(openPixels, source.width, source.height, sampleX, y);
              const closedAlpha = closed[3] / 255;
              const openAlpha = open[3] / 255;
              const localX = (x + 0.5) / source.width;
              const localY = (y + 0.5) / source.height;
              const leftEye = eyeMask(localX, localY, 0.416);
              const rightEye = eyeMask(localX, localY, 0.579);
              const leftClose = leftEye * Math.max(1 - eyeOpen, blink, wink);
              const rightClose = rightEye * Math.max(1 - eyeOpen, blink);
              const closedBlend = Math.max(leftClose, rightClose);
              const bodyCurve = smoothstep(0.77, 1, localY);
              const halfWidth = 0.48 * Math.sqrt(1 - bodyCurve);
              const sideFade = 1 - smoothstep(halfWidth - 0.025, halfWidth + 0.025, Math.abs(localX - 0.5));
              const roundedEnd = 1 - smoothstep(0.985, 1, localY);
              const shapeFade = 1 - bodyCurve * (1 - sideFade);
              const alphaMask = shapeFade * roundedEnd;
              alpha += (closedAlpha * closedBlend + openAlpha * (1 - closedBlend)) * alphaMask;
              red += (closed[0] * closedAlpha * closedBlend + open[0] * openAlpha * (1 - closedBlend)) * alphaMask;
              green += (closed[1] * closedAlpha * closedBlend + open[1] * openAlpha * (1 - closedBlend)) * alphaMask;
              blue += (closed[2] * closedAlpha * closedBlend + open[2] * openAlpha * (1 - closedBlend)) * alphaMask;
              samples += 1;
            }
          }

          alpha /= samples;
          if (alpha < 0.035) continue;
          red /= Math.max(0.001, alpha * samples);
          green /= Math.max(0.001, alpha * samples);
          blue /= Math.max(0.001, alpha * samples);
          const sourceLum = luminance([red, green, blue]);
          const compositeLum = sourceLum * alpha + paperLum * (1 - alpha);
          let darkness = paperLum > 0.5
            ? Math.max(0, (paperLum - compositeLum) / paperLum)
            : Math.max(0, (compositeLum - paperLum) / (1 - paperLum));
          if (paperLum <= 0.5 && sourceLum < 0.2) darkness = Math.max(darkness, 0.72 * alpha);
          const index = row * cols + col;
          densities[index] = Math.min(1, darkness * 1.38);
          const eyeDetail = horizontal > 0.36 && horizontal < 0.64 && vertical > 0.36 && vertical < 0.55;
          if (eyeDetail && blue > red * 1.1 && blue > green * 1.08) darkness = Math.max(darkness, 0.62);
          colors[index] = textColor(red, green, blue, ink, muted, paper, eyeDetail);
        }
      }

      output.clearRect(0, 0, outputWidth, outputHeight);
      output.font = `${fontSize}px "DM Mono", monospace`;
      output.textBaseline = 'top';
      for (let row = 0; row < rows; row += 1) {
        for (let col = 0; col < cols; col += 1) {
          const index = row * cols + col;
          const target = Math.max(0, Math.min(1, densities[index] + errors[index]));
          const glyph = chooseGlyph(glyphs, target);
          const error = target - glyph.darkness;
          if (col + 1 < cols) errors[index + 1] += error * (7 / 16);
          if (row + 1 < rows) {
            if (col > 0) errors[index + cols - 1] += error * (3 / 16);
            errors[index + cols] += error * (5 / 16);
            if (col + 1 < cols) errors[index + cols + 1] += error * (1 / 16);
          }
          const fill = colors[index];
          if (!fill || glyph.darkness < 0.012) continue;
          if (fill !== previousFill) {
            output.fillStyle = fill;
            previousFill = fill;
          }
          output.fillText(glyph.character, col * cellW, row * cellH);
        }
      }

      output.save();
      output.font = `bold ${fontSize * 1.6}px "DM Mono", monospace`;
      output.textBaseline = 'middle';
      const eyeBlue = paperLum > 0.5 ? 'rgb(42, 83, 142)' : 'rgb(136, 181, 241)';
      [
        { x: 0.409, close: Math.max(1 - eyeOpen, blink, wink) },
        { x: 0.579, close: Math.max(1 - eyeOpen, blink) },
      ].forEach(({ x, close }) => {
        const centerX = x * outputWidth;
        const centerY = 0.432 * outputHeight;
        if (close > 0.52) {
          output.fillStyle = ink;
          output.fillText('-', centerX - fontSize * 0.34, centerY);
          return;
        }
        output.fillStyle = eyeBlue;
        output.fillText('@', centerX - fontSize * 0.45, centerY);
        if (!reducedMotion.matches && close < 0.12) {
          output.fillStyle = paperLum > 0.5 ? '#fff' : ink;
          output.fillText('.', centerX + fontSize * 0.12, centerY - fontSize * 0.38);
        }
      });
      output.restore();

      if (!reducedMotion.matches) frame = requestAnimationFrame(draw);
    };

    const start = () => {
      if (disposed) return;
      updatePalette();
      layout();
      startTime = performance.now();
      if (inView) frame = requestAnimationFrame(draw);
      else draw(startTime + 50);
    };

    const observer = new IntersectionObserver(([entry]) => {
      inView = entry.isIntersecting;
      if (inView && !frame && !reducedMotion.matches) frame = requestAnimationFrame(draw);
      if (!inView && frame) {
        cancelAnimationFrame(frame);
        frame = 0;
      }
    }, { threshold: 0.01 });
    observer.observe(host);

    const resizeObserver = new ResizeObserver(() => {
      layout();
      if (!frame && inView && !reducedMotion.matches) frame = requestAnimationFrame(draw);
      else if (reducedMotion.matches) draw(performance.now());
    });
    resizeObserver.observe(host);

    const themeObserver = new MutationObserver(() => {
      updatePalette();
      if (reducedMotion.matches) draw(performance.now());
    });
    themeObserver.observe(document.body, { attributes: true, attributeFilter: ['class'] });

    const onMotionPreference = () => {
      cancelAnimationFrame(frame);
      frame = 0;
      startTime = performance.now();
      if (reducedMotion.matches) draw(startTime + 50);
      else if (inView) frame = requestAnimationFrame(draw);
    };
    reducedMotion.addEventListener('change', onMotionPreference);

    layout();
    void Promise.all([document.fonts.ready, closedImage.decode(), openImage.decode()]).then(start).catch(() => undefined);

    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      observer.disconnect();
      resizeObserver.disconnect();
      themeObserver.disconnect();
      reducedMotion.removeEventListener('change', onMotionPreference);
    };
  }, []);

  return (
    <div className="vision-portrait" role="img" aria-label="Retrato em ASCII da Sailor Moon; ela abre e pisca os olhos, dá uma piscadela, e as mechas do cabelo se movem.">
      <canvas aria-hidden="true" className="vision-portrait-canvas" ref={canvasRef} />
    </div>
  );
}
