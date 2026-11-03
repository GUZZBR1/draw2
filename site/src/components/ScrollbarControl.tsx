import { useState } from 'react';
import { motion, useMotionValueEvent, useScroll, useTransform } from 'motion/react';

export default function ScrollbarControl() {
  const { scrollYProgress } = useScroll();
  const [scrollPercent, setScrollPercent] = useState(0);
  const thumbPosition = useTransform(scrollYProgress, [0, 1], [0, 48]);

  useMotionValueEvent(scrollYProgress, 'change', (value) =>
    setScrollPercent(Math.round(value * 100)),
  );

  function seekScroll(clientY: number, rail: HTMLDivElement) {
    const bounds = rail.getBoundingClientRect();
    const trackTop = bounds.top + 16;
    const trackHeight = bounds.height - 32;
    const pointerRatio = Math.max(0, Math.min(1, (clientY - trackTop) / trackHeight));
    const thumbRatio = 48 / trackHeight;
    const position = Math.max(0, Math.min(1, (pointerRatio - thumbRatio / 2) / (1 - thumbRatio)));
    const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
    window.scrollTo({ top: position * maxScroll, behavior: 'auto' });
  }

  function moveByKeyboard(key: string) {
    const steps: Record<string, number> = {
      ArrowDown: 5,
      PageDown: 20,
      ArrowUp: -5,
      PageUp: -20,
      Home: -100,
      End: 100,
    };
    if (!(key in steps)) return;
    const next = Math.max(0, Math.min(100, scrollPercent + steps[key]));
    const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
    window.scrollTo({ top: (next / 100) * maxScroll, behavior: 'auto' });
  }

  return (
    <div
      className="scrollbar-control fixed right-2 top-1/2 z-[45] flex h-[128px] w-8 -translate-y-1/2 touch-none items-center justify-center rounded-full outline-none focus-visible:ring-2 focus-visible:ring-accent"
      role="scrollbar"
      aria-label="Barra de rolagem da página"
      aria-controls="page-content"
      aria-orientation="vertical"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={scrollPercent}
      tabIndex={0}
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId);
        seekScroll(event.clientY, event.currentTarget);
      }}
      onPointerMove={(event) => {
        if (event.buttons === 1) seekScroll(event.clientY, event.currentTarget);
      }}
      onKeyDown={(event) => {
        if (['ArrowDown', 'PageDown', 'ArrowUp', 'PageUp', 'Home', 'End'].includes(event.key)) {
          event.preventDefault();
          moveByKeyboard(event.key);
        }
      }}
    >
      <span className="scrollbar-track absolute inset-y-4 left-1/2 w-[2px] -translate-x-1/2 rounded-full bg-ink/10" />
      <motion.span
        className="scrollbar-thumb absolute left-1/2 top-4 h-12 w-[6px] -translate-x-1/2 rounded-full bg-ink/45 transition-[background-color,width] duration-200 hover:bg-accent focus:bg-accent"
        style={{ y: thumbPosition, x: '-50%' }}
      />
    </div>
  );
}
