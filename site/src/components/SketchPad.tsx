import { useCallback, useEffect, useRef, useState } from 'react';

type Point = { x: number; y: number; pressure: number };
type Tool = 'pen' | 'eraser';
type Stroke = { color: string; width: number; tool: Tool; points: Point[] };

const COLORS = [
  { name: 'Grafite', value: '#22231f' },
  { name: 'Terracota', value: '#c65d45' },
  { name: 'Azul', value: '#527c91' },
  { name: 'Verde', value: '#78885a' },
  { name: 'Ocre', value: '#c49a3a' },
  { name: 'Violeta', value: '#8c719a' },
];
const SIZES = [
  { name: 'Fino', value: 2 },
  { name: 'Médio', value: 4 },
  { name: 'Grosso', value: 8 },
];
const POINT_LIMIT = 5000;

function pressureWidth(stroke: Stroke, pressure: number) {
  return stroke.width * (0.55 + Math.max(0.1, pressure) * 0.9);
}

function ToolIcon({ tool }: { tool: Tool }) {
  return tool === 'pen' ? (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="m4 20 4.2-1 10.9-10.9a2.1 2.1 0 0 0-3-3L5.2 16 4 20Z" />
      <path d="m14.8 6.5 3 3" />
    </svg>
  ) : (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="m4 15 8.8-9a2 2 0 0 1 2.9 0l4.1 4.2a2 2 0 0 1 0 2.8L13 20H8l-4-4a.7.7 0 0 1 0-1Z" />
      <path d="m8 20 8-8" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 7h16M10 11v6m4-6v6M6 7l1 13h10l1-13M9 7V4h6v3" />
    </svg>
  );
}

export default function SketchPad() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const strokesRef = useRef<Stroke[]>([]);
  const activeStrokeRef = useRef<Stroke | null>(null);
  const activePointerRef = useRef<number | null>(null);
  const activeBoundsRef = useRef<DOMRect | null>(null);
  const [color, setColor] = useState(COLORS[0].value);
  const [brushSize, setBrushSize] = useState(4);
  const [tool, setTool] = useState<Tool>('pen');

  const paintStroke = useCallback((
    context: CanvasRenderingContext2D,
    stroke: Stroke,
    width: number,
    height: number,
  ) => {
    const points = stroke.points;
    if (!points.length) return;
    context.globalCompositeOperation = stroke.tool === 'eraser' ? 'destination-out' : 'source-over';
    context.strokeStyle = stroke.color;
    context.fillStyle = stroke.color;
    context.lineCap = 'round';
    context.lineJoin = 'round';

    if (points.length === 1) {
      const point = points[0];
      const radius = pressureWidth(stroke, point.pressure) / 2;
      context.beginPath();
      context.arc(point.x * width, point.y * height, radius, 0, Math.PI * 2);
      context.fill();
    } else {
      for (let index = 1; index < points.length; index += 1) {
        const from = points[index - 1];
        const to = points[index];
        context.lineWidth = pressureWidth(stroke, (from.pressure + to.pressure) / 2);
        context.beginPath();
        context.moveTo(from.x * width, from.y * height);
        context.lineTo(to.x * width, to.y * height);
        context.stroke();
      }
    }
    context.globalCompositeOperation = 'source-over';
  }, []);

  const repaint = useCallback(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) return;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    context.clearRect(0, 0, width, height);
    for (const stroke of strokesRef.current) paintStroke(context, stroke, width, height);
    if (activeStrokeRef.current) paintStroke(context, activeStrokeRef.current, width, height);
  }, [paintStroke]);

  const resizeCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { width, height } = canvas.getBoundingClientRect();
    if (!width || !height) return;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    const pixelWidth = Math.round(width * pixelRatio);
    const pixelHeight = Math.round(height * pixelRatio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    canvas.getContext('2d')?.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    repaint();
  }, [repaint]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(resizeCanvas);
    observer.observe(canvas);
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', resizeCanvas);
    };
  }, [resizeCanvas]);

  function makePoint(clientX: number, clientY: number, pressure: number, bounds: DOMRect): Point {
    return {
      x: Math.max(0, Math.min(1, (clientX - bounds.left) / bounds.width)),
      y: Math.max(0, Math.min(1, (clientY - bounds.top) / bounds.height)),
      pressure: pressure || 0.5,
    };
  }

  function beginStroke(event: React.PointerEvent<HTMLCanvasElement>) {
    if (!event.isPrimary || event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const bounds = event.currentTarget.getBoundingClientRect();
    activeBoundsRef.current = bounds;
    const point = makePoint(event.clientX, event.clientY, event.pressure, bounds);
    activePointerRef.current = event.pointerId;
    activeStrokeRef.current = { color, width: brushSize, tool, points: [point] };
    paintStroke(event.currentTarget.getContext('2d')!, activeStrokeRef.current, event.currentTarget.clientWidth, event.currentTarget.clientHeight);
  }

  function continueStroke(event: React.PointerEvent<HTMLCanvasElement>) {
    const stroke = activeStrokeRef.current;
    if (!stroke || activePointerRef.current !== event.pointerId) return;
    event.preventDefault();
    const samples = event.nativeEvent.getCoalescedEvents?.() ?? [event.nativeEvent];
    const context = event.currentTarget.getContext('2d');
    const { clientWidth: width, clientHeight: height } = event.currentTarget;
    const bounds = activeBoundsRef.current;
    if (!bounds) return;
    for (const sample of samples) {
      const point = makePoint(sample.clientX, sample.clientY, sample.pressure, bounds);
      const previous = stroke.points[stroke.points.length - 1];
      if (stroke.points.length < POINT_LIMIT) stroke.points.push(point);
      if (context) paintStroke(context, { ...stroke, points: [previous, point] }, width, height);
    }
  }

  function finishStroke(event?: React.PointerEvent<HTMLCanvasElement>) {
    if (event && activePointerRef.current !== event.pointerId) return;
    const stroke = activeStrokeRef.current;
    if (!stroke) return;
    activeStrokeRef.current = null;
    activePointerRef.current = null;
    activeBoundsRef.current = null;
    strokesRef.current.push(stroke);
  }

  function clearAll() {
    activeStrokeRef.current = null;
    activePointerRef.current = null;
    activeBoundsRef.current = null;
    strokesRef.current = [];
    repaint();
  }

  return (
    <section className="sketch-section" aria-label="Folha para desenhar">
      <div className="sketch-toolbar" aria-label="Ferramentas de desenho">
        <div className="sketch-tool-switch" role="group" aria-label="Ferramenta">
          {(['pen', 'eraser'] as const).map((option) => (
            <button
              className={`sketch-tool-button${tool === option ? ' is-selected' : ''}`}
              key={option}
              type="button"
              aria-label={option === 'pen' ? 'Caneta' : 'Borracha'}
              aria-pressed={tool === option}
              title={option === 'pen' ? 'Caneta' : 'Borracha'}
              onClick={() => setTool(option)}
            >
              <ToolIcon tool={option} />
            </button>
          ))}
        </div>
        <span className="sketch-toolbar-divider" aria-hidden="true" />
        <div className="sketch-palette" role="group" aria-label="Cor do traço">
          {COLORS.map((swatch) => (
            <button
              className={`sketch-swatch${color === swatch.value ? ' is-selected' : ''}`}
              key={swatch.value}
              type="button"
              style={{ '--swatch': swatch.value } as React.CSSProperties}
              aria-label={`${swatch.name}${color === swatch.value ? ', selecionada' : ''}`}
              aria-pressed={color === swatch.value}
              title={swatch.name}
              onClick={() => {
                setColor(swatch.value);
                setTool('pen');
              }}
            />
          ))}
        </div>
        <span className="sketch-toolbar-divider sketch-size-divider" aria-hidden="true" />
        <div className="sketch-size-options" role="group" aria-label="Espessura do traço">
          {SIZES.map((size) => (
            <button
              className={`sketch-size${brushSize === size.value ? ' is-selected' : ''}`}
              key={size.value}
              type="button"
              aria-label={`Traço ${size.name.toLowerCase()}`}
              aria-pressed={brushSize === size.value}
              title={`Traço ${size.name.toLowerCase()}`}
              onClick={() => setBrushSize(size.value)}
            >
              <span style={{ width: size.value * 1.7, height: size.value * 1.7 }} />
            </button>
          ))}
        </div>
        <span className="sketch-toolbar-divider sketch-clear-divider" aria-hidden="true" />
        <button
          className="sketch-tool-button sketch-clear-button"
          type="button"
          aria-label="Apagar tudo"
          title="Apagar tudo"
          onClick={clearAll}
        >
          <TrashIcon />
        </button>
      </div>
      <p className="sketch-hint" aria-hidden="true">Toque e arraste, solte quando terminar</p>
      <canvas
        ref={canvasRef}
        className="sketch-canvas"
        aria-label="Folha de desenho em branco. Use mouse, toque ou caneta para desenhar."
        tabIndex={0}
        onPointerDown={beginStroke}
        onPointerMove={continueStroke}
        onPointerUp={finishStroke}
        onPointerCancel={finishStroke}
        onLostPointerCapture={finishStroke}
        onContextMenu={(event) => event.preventDefault()}
      >
        Canvas indisponível neste navegador.
      </canvas>
    </section>
  );
}
