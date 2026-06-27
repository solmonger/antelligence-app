# Antelligence Frontend

This frontend is the local React/Vite interface for Antelligence. It visualizes swarm simulations, tumor-focused nanobot runs, and comparison views while keeping real backend execution on the private Python service.

## Current routes

- `/` - main colony simulation interface
- `/comparison` - side-by-side simulation comparison view
- `/tumor` - tumor nanobot simulation playback and controls
- `/tumor-hunt` - wave-based tumor hunt demo

## Local development

From the repo root:

```bash
npm --prefix frontend install
npm --prefix frontend run dev
```

The Vite dev server runs on `http://127.0.0.1:8081`.

## Backend connection

The frontend expects the Antelligence API at `http://127.0.0.1:8001` by default. Override it with `VITE_API_BASE_URL` if your backend is elsewhere.

Useful runtime variables:

- `VITE_API_BASE_URL` - backend base URL for simulation requests
- `VITE_FRONTEND_MODE` - set to `preview` to disable private backend execution in the public UI
- `VITE_PREVIEW_HOSTNAME` - hostname shown in preview banners
- `VITE_BUILD_LABEL` and `VITE_GIT_SHA` - optional build metadata surfaced in the UI

## Preview mode

Preview mode is a frontend-only surface. When `VITE_FRONTEND_MODE=preview`, the app keeps backend execution private and shows banner/toast messaging instead of sending live simulation requests.

## Build

```bash
npm --prefix frontend run build
npm --prefix frontend run preview
```

## Stack

- React 18 + TypeScript
- Vite
- TanStack Query
- React Router
- Tailwind + shadcn/ui
- Recharts for simulation charts
