# Repository guidance

Before changing the website, read [`design.md`](design.md). Treat it as the source of truth for the Draw brand, page composition, typography, glass surfaces, and interaction behavior.

## Website implementation

- Website source lives in `site/` and uses React, TypeScript, Tailwind CSS v4, React Router, and Motion (`motion/react`). Keep that stack unless the user asks to change it.
- Reuse the existing design tokens and shared header patterns. Do not introduce fonts, colors, page sections, decorative rules, or animations that conflict with `design.md`.
- Keep the Draw definition card in the shared header only. Do not add another definition block to Vision or Product.
- Keep the Product construction artwork static.
- Preserve keyboard access, reduced-motion support, dark mode, and responsive behavior when editing interactions.
- Open the definition card from the Draw wordmark only. Moving into the card keeps it open; moving to Product closes it.
- Keep the custom scroll scrubber large enough to read and interact with. Preserve pointer dragging and keyboard control.
- When checking visual changes, inspect the relevant route at desktop width and narrow widths, including 320 px where the layout allows it.

## Change discipline

- Prefer small, localized changes that preserve the existing page structure.
- If a visual rule needs to change, update `design.md` in the same change so later work remains consistent.
- Keep interface copy concise, natural, and aligned with the product vision in the repository.
