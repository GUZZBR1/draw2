import { useEffect, useRef, useState } from 'react';
import {
  motion,
  useMotionTemplate,
  useMotionValueEvent,
  useReducedMotion,
  useScroll,
  useTransform,
} from 'motion/react';
import { Link, useLocation } from 'react-router-dom';

const MORPH_DISTANCE = 180;

function scrollProgress(value: number, reduceMotion: boolean | null) {
  if (reduceMotion) return 0;
  return Math.min(1, Math.max(0, value / MORPH_DISTANCE));
}

function startFontSize() {
  if (window.matchMedia('(max-width: 640px)').matches) {
    return Math.max(92, Math.min(window.innerWidth * 0.24, 148));
  }
  return Math.max(112, Math.min(window.innerWidth * 0.19, 224));
}

function compactFontSize() {
  return window.matchMedia('(max-width: 640px)').matches ? 22 : 26;
}

function readSavedTheme(): boolean {
  try {
    return localStorage.getItem('draw-theme') === 'dark';
  } catch {
    return false;
  }
}

interface SiteHeaderProps {
  onRepeatProduct: () => void;
}

export default function SiteHeader({ onRepeatProduct }: SiteHeaderProps) {
  const { scrollY } = useScroll();
  const reduceMotion = useReducedMotion();
  const location = useLocation();
  const isHome = location.pathname === '/';
  const isVision = location.pathname === '/vision';
  const isProduct = location.pathname === '/product';
  const [isScrolled, setIsScrolled] = useState(!isHome);
  const [brandHovered, setBrandHovered] = useState(false);
  const [cardHovered, setCardHovered] = useState(false);
  const [brandFocused, setBrandFocused] = useState(false);
  const [webcamHovered, setWebcamHovered] = useState(false);
  const [webcamCardHovered, setWebcamCardHovered] = useState(false);
  const [webcamFocused, setWebcamFocused] = useState(false);
  const [webcamPinned, setWebcamPinned] = useState(false);
  const [dark, setDark] = useState(readSavedTheme);
  const [, setViewportRevision] = useState(0);
  const hoverCloseTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const webcamAnchorRef = useRef<HTMLDivElement>(null);
  const showWebcamCard = isHome && !isScrolled && (webcamHovered || webcamCardHovered || webcamFocused || webcamPinned);
  const showDefinition = (brandHovered || cardHovered || brandFocused) && !showWebcamCard;

  function cancelHoverClose() {
    if (hoverCloseTimer.current) clearTimeout(hoverCloseTimer.current);
    hoverCloseTimer.current = null;
  }

  function scheduleHoverClose() {
    cancelHoverClose();
    hoverCloseTimer.current = setTimeout(() => {
      setBrandHovered(false);
      setCardHovered(false);
      setWebcamHovered(false);
      setWebcamCardHovered(false);
    }, 160);
  }

  const progress = useTransform(scrollY, (value) =>
    isHome ? scrollProgress(value, reduceMotion) : 1,
  );
  const brandScale = useTransform(progress, (amount) => {
    const ratio = compactFontSize() / startFontSize();
    return 1 + (ratio - 1) * amount;
  });
  const brandY = useTransform(progress, (amount) => {
    const height = startFontSize() * 0.83;
    const scale = 1 + (compactFontSize() / startFontSize() - 1) * amount;
    const centered = -height / 2;
    const compact = 18 + (height * scale) / 2 - height / 2 - window.innerHeight / 2;
    return centered + (compact - centered) * amount;
  });
  const productOpacity = useTransform(progress, [0.62, 0.94], [0, 1]);
  const productX = useTransform(progress, [0.62, 0.94], [-6, 0]);
  const webcamOpacity = useTransform(progress, [0, 0.14], [1, 0]);
  const inverseBrandScale = useTransform(() => 1 / brandScale.get());
  const glassAmount = useTransform(progress, [0, 0.22], [0, 44]);
  const borderAmount = useTransform(progress, [0, 0.18], [0, 11]);
  const blurAmount = useTransform(progress, [0, 0.2], [0, 28]);
  const headerGutter = useTransform(progress, (amount) =>
    (window.matchMedia('(max-width: 640px)').matches ? 16 : 64) * amount,
  );
  const headerMaxWidth = useTransform(progress, [0, 1], [1440, 1160]);
  const headerWidth = useMotionTemplate`min(calc(100% - ${headerGutter}px), ${headerMaxWidth}px)`;
  const navHeight = useTransform(progress, (amount) => {
    const mobile = window.matchMedia('(max-width: 640px)').matches;
    const initial = mobile ? 44 : 56;
    const compact = mobile ? 38 : 44;
    return initial + (compact - initial) * amount;
  });
  const themeButtonScale = useTransform(progress, (amount) => {
    const mobile = window.matchMedia('(max-width: 640px)').matches;
    const compact = mobile ? 30 / 34 : 34 / 42;
    return 1 + (compact - 1) * amount;
  });
  const navPadding = useTransform(progress, (amount) => {
    const mobile = window.matchMedia('(max-width: 640px)').matches;
    const initial = mobile ? 18 : Math.max(22, Math.min(window.innerWidth * 0.052, 70));
    const compact = mobile ? 12 : Math.max(16, Math.min(window.innerWidth * 0.02, 28));
    return initial + (compact - initial) * amount;
  });
  const navRadius = useTransform(progress, [0, 0.2], [0, 18]);
  const navBackground = useMotionTemplate`color-mix(in srgb, var(--paper) ${glassAmount}%, transparent)`;
  const navBorder = useMotionTemplate`color-mix(in srgb, var(--ink) ${borderAmount}%, transparent)`;
  const navBlur = useMotionTemplate`blur(${blurAmount}px) saturate(138%)`;
  const navShadowAlpha = useTransform(progress, [0, 0.25], [0, 0.045]);
  const navShadow = useMotionTemplate`0 7px 25px rgba(34, 35, 31, ${navShadowAlpha}), inset 0 1px rgba(255, 255, 255, ${navShadowAlpha})`;

  useMotionValueEvent(scrollY, 'change', (value) => setIsScrolled(!isHome || value > 8));

  useEffect(() => {
    setIsScrolled(!isHome);
    setBrandHovered(false);
    setCardHovered(false);
    setBrandFocused(false);
    setWebcamHovered(false);
    setWebcamCardHovered(false);
    setWebcamFocused(false);
    setWebcamPinned(false);
  }, [isHome, location.pathname]);

  useEffect(() => {
    if (!isScrolled) return;
    setWebcamHovered(false);
    setWebcamCardHovered(false);
    setWebcamFocused(false);
    setWebcamPinned(false);
  }, [isScrolled]);

  useEffect(() => {
    if (!webcamPinned) return;
    const closeOnOutsidePointer = (event: PointerEvent) => {
      if (!webcamAnchorRef.current?.contains(event.target as Node)) setWebcamPinned(false);
    };
    document.addEventListener('pointerdown', closeOnOutsidePointer);
    return () => document.removeEventListener('pointerdown', closeOnOutsidePointer);
  }, [webcamPinned]);

  useEffect(() => {
    const updateViewport = () => setViewportRevision((revision) => revision + 1);
    window.addEventListener('resize', updateViewport);
    return () => window.removeEventListener('resize', updateViewport);
  }, []);

  useEffect(() => {
    document.body.classList.toggle('dark', dark);
    document.querySelector('meta[name="theme-color"]')?.setAttribute(
      'content',
      dark ? '#1b1c19' : '#f7f7f4',
    );
    try {
      localStorage.setItem('draw-theme', dark ? 'dark' : 'light');
    } catch {
      // The visual toggle still works when browser storage is unavailable.
    }
  }, [dark]);

  useEffect(() => () => cancelHoverClose(), []);

  useEffect(() => {
    if (!showDefinition && !showWebcamCard) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setBrandHovered(false);
        setBrandFocused(false);
        setWebcamHovered(false);
        setWebcamCardHovered(false);
        setWebcamFocused(false);
        setWebcamPinned(false);
      }
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [showDefinition, showWebcamCard]);

  return (
    <>
      <motion.header
        className={`site-header fixed left-1/2 top-2 z-50 ${!isHome || isScrolled ? 'is-scrolled' : ''}`}
        style={{ width: headerWidth, maxWidth: headerMaxWidth, x: '-50%' }}
      >
        <motion.nav
          className="nav-shell flex w-full items-center justify-between gap-4 border text-ink shadow-[0_7px_25px_rgba(34,35,31,0.04)] backdrop-blur-2xl"
          style={{
            height: navHeight,
            paddingInline: navPadding,
            borderRadius: navRadius,
            backgroundColor: navBackground,
            borderColor: navBorder,
            backdropFilter: navBlur,
            boxShadow: navShadow,
          }}
          aria-label="Navegação principal"
        >
          <Link
            className={`nav-link vision-link font-mono text-[16px] tracking-[.025em] text-muted transition-colors duration-300 hover:text-ink${isVision ? ' is-active' : ''}`}
            to="/vision"
            aria-current={isVision ? 'page' : undefined}
          >
            vision
          </Link>
          <div className="nav-right flex items-center gap-3 sm:gap-5">
            <motion.span
              className="location font-mono text-[10px] tracking-[.025em] text-muted sm:text-[16px]"
              whileHover={{ y: -1, color: 'var(--ink)' }}
              transition={{ duration: 0.2, ease: 'easeOut' }}
            >
              são paulo, brasil
            </motion.span>
            <motion.button
              className="theme-toggle min-h-[34px] min-w-[56px] rounded-full border border-ink/15 bg-paper/20 px-3 font-mono text-[10px] text-ink transition duration-300 hover:-translate-y-px hover:border-ink/40 hover:bg-ink/5 active:translate-y-0 active:scale-[.97] sm:min-h-[42px] sm:min-w-[78px] sm:px-4 sm:text-[13px]"
              style={{ scaleY: themeButtonScale, transformOrigin: 'center' }}
              type="button"
              aria-label={`Ativar tema ${dark ? 'claro' : 'escuro'}`}
              onClick={() => setDark((current) => !current)}
            >
              {dark ? 'light' : 'dark'}
            </motion.button>
          </div>
        </motion.nav>
      </motion.header>

      <motion.div
        className={`brand-lockup${!isHome || isScrolled ? ' is-scrolled' : ''}`}
        style={{ top: '50%', y: brandY, x: '-50%', scale: brandScale }}
      >
        <motion.div
          className="wordmark-trigger"
          onHoverStart={() => {
            cancelHoverClose();
            setBrandHovered(true);
          }}
          onHoverEnd={scheduleHoverClose}
          onFocus={() => {
            cancelHoverClose();
            setBrandFocused(true);
          }}
          onBlur={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
              setBrandFocused(false);
            }
          }}
        >
          <Link
            className="wordmark relative z-[1] block font-sans font-extrabold leading-[.83] tracking-[-.035em] text-ink no-underline"
            to="/"
            aria-label="Draw, início"
          >
            Draw
          </Link>
          {isHome && (
            <div className="webcam-interaction" ref={webcamAnchorRef}>
              <motion.div
                className="webcam-anchor"
                aria-hidden={isScrolled}
                style={{
                  opacity: reduceMotion ? (isScrolled ? 0 : 1) : webcamOpacity,
                  pointerEvents: isScrolled ? 'none' : 'auto',
                }}
                onHoverStart={() => {
                  cancelHoverClose();
                  setBrandHovered(false);
                  setCardHovered(false);
                  setBrandFocused(false);
                  setWebcamHovered(true);
                }}
                onHoverEnd={scheduleHoverClose}
              >
                <motion.button
                  className="webcam-trigger"
                  type="button"
                  aria-label="Sobre a webcam do Draw"
                  aria-controls="webcam-info-card"
                  aria-expanded={showWebcamCard}
                  tabIndex={isScrolled ? -1 : 0}
                  title="Sobre a webcam do Draw"
                  onFocus={() => {
                    cancelHoverClose();
                    setBrandFocused(false);
                    setWebcamFocused(true);
                  }}
                  onBlur={() => setWebcamFocused(false)}
                  onClick={() => setWebcamPinned((pinned) => !pinned)}
                  whileHover={{ y: -1 }}
                  transition={{ duration: 0.18, ease: 'easeOut' }}
                >
                  <svg className="webcam-icon" viewBox="0 0 64 40" aria-hidden="true">
                    <rect x="5" y="4" width="54" height="26" rx="9" />
                    <circle className="webcam-lens" cx="32" cy="17" r="8" />
                    <circle className="webcam-aperture" cx="32" cy="17" r="3" />
                    <circle className="webcam-status" cx="51" cy="12" r="1.5" />
                    <path d="M27 30v3c0 1.8 1.4 3 3 3h4c1.7 0 3-1.3 3-3v-3" />
                  </svg>
                </motion.button>
              </motion.div>
              <motion.aside
                id="webcam-info-card"
                className="webcam-card-anchor definition-card"
                aria-label="Sobre a webcam do Draw"
                aria-hidden={!showWebcamCard}
                initial={false}
                animate={showWebcamCard ? 'visible' : 'hidden'}
                style={{ scale: inverseBrandScale }}
                onHoverStart={() => {
                  cancelHoverClose();
                  setBrandHovered(false);
                  setCardHovered(false);
                  setBrandFocused(false);
                  setWebcamCardHovered(true);
                }}
                onHoverEnd={scheduleHoverClose}
                variants={{
                  hidden: { opacity: 0, y: 8, pointerEvents: 'none' },
                  visible: { opacity: 1, y: 0, pointerEvents: 'auto' },
                }}
                transition={{ type: 'spring', stiffness: 320, damping: 28 }}
              >
                <h1>webcam</h1>
                <p className="definition-kind"><em>instrumento</em> · captura</p>
                <p className="definition-phrase">
                  Acompanha o papel e transforma cada traço em tinta digital, em tempo real.
                </p>
                <p className="definition-note">A primeira camada do Draw.</p>
              </motion.aside>
            </div>
          )}
        </motion.div>
        <motion.div
          className={`product-label font-mono text-muted${isProduct ? ' is-active' : ''}`}
          style={{
            opacity: productOpacity,
            x: productX,
            y: '-50%',
            scale: inverseBrandScale,
            transformOrigin: 'left center',
          }}
          onHoverStart={() => {
            cancelHoverClose();
            setBrandHovered(false);
            setCardHovered(false);
          }}
          whileHover={{ filter: 'brightness(1.25)' }}
          transition={{ type: 'spring', stiffness: 380, damping: 24 }}
        >
          <Link
            to="/product"
            aria-current={isProduct ? 'page' : undefined}
            onClick={(event) => {
              if (isProduct) {
                event.preventDefault();
                onRepeatProduct();
              }
            }}
          >
            <span aria-hidden="true">·</span> product
          </Link>
        </motion.div>
        <motion.div className="definition-card-anchor" style={{ x: '-50%', scale: inverseBrandScale }}>
          <motion.aside
            className="definition-card border border-white/70 bg-paper/65 p-5 text-ink shadow-[0_18px_54px_rgba(36,37,32,.1)] backdrop-blur-2xl"
            aria-label="Definição de Draw"
            aria-hidden={!showDefinition}
            initial={false}
            animate={showDefinition ? 'visible' : 'hidden'}
            onHoverStart={() => {
              cancelHoverClose();
              setCardHovered(true);
            }}
            onHoverEnd={scheduleHoverClose}
            variants={{
              hidden: { opacity: 0, y: 9, pointerEvents: 'none' },
              visible: { opacity: 1, y: 0, pointerEvents: 'auto' },
            }}
            transition={{ type: 'spring', stiffness: 320, damping: 28 }}
          >
            <h1>draw</h1>
            <p className="definition-kind"><em>substantivo</em> · conceito</p>
            <p className="definition-phrase">
              Traçar uma ideia no papel e transformá-la em algo que pode ser compreendido,
              testado e levado ao mundo.
            </p>
            <p className="definition-note">Do papel à vida.</p>
          </motion.aside>
        </motion.div>
      </motion.div>
    </>
  );
}
