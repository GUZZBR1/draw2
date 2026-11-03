# Draw site

The site is a small React and TypeScript app built with Vite and Tailwind CSS 4. Motion for React handles the scroll-linked wordmark and glass header. The branded scrollbar supports pointer dragging and keyboard navigation.

## Source layout

- `src/app/` composes the router, shared shell, and route transitions.
- `src/pages/` contains one module per page: home, vision, and product.
- `src/components/` contains shared interface modules such as the header and scrollbar.
- `src/styles.css` owns the shared tokens and visual rules for those modules.
- `public/` contains static assets served directly by Vite.

The shared header owns the scroll-linked wordmark, theme switch, and Draw definition. The scrollbar owns its own drag and keyboard behavior. Pages should not reimplement either module.

Keep route-specific copy and layout in its page module. Keep behavior shared across routes in `components/`, and follow the repository's [`../design.md`](../design.md) and [`../AGENTS.md`](../AGENTS.md) when changing visual or structural conventions.

```bash
npm install
npm run dev
```

Create a production build with `npm run build`.
