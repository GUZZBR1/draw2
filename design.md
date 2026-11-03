# Draw — Design System

This file is the visual source of truth for the Draw website. Keep new pages and changes aligned with these decisions.

## Brand

Draw is a bridge from a person’s paper sketch to digital tools that can interpret, model, and test the idea. The site should feel thoughtful, precise, calm, and ambitious: editorial typography with a light studio interface.

## Color and surfaces

- Light canvas: warm off-white `#f7f7f4`.
- Ink: soft near-black `#22231f`.
- Muted text: `#777870`.
- Accent: pale paper yellow `#e7dfa1`, used sparingly for highlights and interaction feedback.
- Dark mode uses the existing CSS variables; components must inherit them instead of hard-coding light colors.
- Glass surfaces use a translucent paper tint, a fine ink border, restrained shadow, and backdrop blur. Keep enough tint for text behind a card to become unreadable.
- Prefer rounded pills for compact controls and softly rounded cards for content. Avoid boxed rules and decorative borders that interrupt the editorial page flow.

## Typography

- Sans: Manrope, with the existing system fallbacks. Use it for headlines, body copy, and interface labels where a clean reading voice is needed.
- Mono: DM Mono, with the existing system fallbacks. Use it for navigation, metadata, and small status labels.
- Serif italics are reserved for short editorial accents already present in the Vision and Product headlines. Do not apply italics or bold to whole paragraphs.
- Definition card: a restrained medium-weight sans title; small mono part-of-speech label; readable, regular-weight sans definition; muted secondary line. Italicize only the part-of-speech word.
- Do not add a new font family without updating this system and getting an explicit design reason.

## Header and motion

- Keep `Draw` centered in the header as the page scrolls. It begins as the large home-page wordmark and settles into the compact header position without jumping.
- The navigation becomes a narrow, light glass surface while scrolling: subtle blur, fine border, soft shadow, and a shorter height.
- The header, centered wordmark, and definition card must remain legible and within the viewport at browser zoom and at 320 px CSS width.
- Hover feedback should be small and clear: a fine underline, slight color change, or a one-pixel lift. Avoid jitter, looping motion, and pointer-driven movement of content.
- The definition opens from the Draw wordmark only. It stays open while the pointer moves from the wordmark onto the card and closes when moving to Product or another control.
- The custom page scrubber is a visible, compact glass capsule with a readable track and thumb. Keep pointer dragging and keyboard navigation; do not reduce it to a tiny line.
- Respect reduced-motion preferences.
- The Product construction illustration is static. Do not animate or parallax the artwork.

## Definition card

- The Draw definition appears only when hovering or keyboard-focusing the Draw mark in the shared header, on every route.
- Keep the glass card centered beneath the mark, readable at compact and large viewport sizes, with no scaling tied to the wordmark size.
- The card copy should describe the paper-to-digital idea in plain language and retain enough detail to communicate the vision. The card does not belong in the Vision page body.

## Page composition

- Home places a still archive of consequential human drawings around the centered Draw wordmark. Keep the artifacts static, make each one link to its primary or archival source, and preserve room around the wordmark. Add a small webcam SVG perched on the final “w”; fade it as the wordmark settles into the compact header. Hovering or focusing the webcam opens a matching glass card that explains its role in live capture. Place that card to the wordmark’s right on desktop and below it on narrow screens, clear of the page scrubber. Keep the webcam card separate from the Draw definition card. On narrow screens, move the archive below the first-screen mark.
- Below the home archive, provide a full-viewport, edge-to-edge drawing sheet. Keep a compact floating glass toolbar for pen, eraser, color, stroke width, and one clear-all action. The sheet must remain free of cards, inset margins, and secondary action rows. Store strokes in normalized coordinates so resize does not distort them, use coalesced pointer events, and cap canvas pixel ratio to keep memory use predictable.
- Vision is an uninterrupted editorial page: a small mono eyebrow, a large centered headline, readable centered paragraphs, and a quiet sign-off. Do not insert a dictionary card or horizontal rules into the Vision copy.
- Product uses the shared pill style for its construction status, a centered static illustration, a clear headline, and concise explanatory copy.
- Use whitespace for hierarchy. Keep the pages minimal and avoid adding sections or labels that were not requested.

## Interaction and responsive behavior

- Navigation links and controls need visible hover and keyboard-focus states.
- The location label gets a restrained hover response on desktop. On narrow layouts, prioritize the centered wordmark and prevent overlap with the theme control.
- Theme changes must work on every route and preserve contrast.
- Test critical header/card layouts at desktop width, 640 px, and 320 px, including while the home wordmark is transitioning on scroll.
