"""Drug field and pathing for the 3D world. Vectorised numpy/scipy, no RNG, no truth."""

from __future__ import annotations

import heapq
import math
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

Vox = Tuple[int, int, int]


class DrugField:
    """Concentration ``conc`` (mass per voxel) and cumulative exposure ``dose``.

    A release deposits ``mass`` as a 3D Gaussian (sigma ``release_sigma_vox``); mass whose kernel falls outside the
    crop is lost at once. Each :meth:`step` diffuses (Gaussian, sigma ``diffuse_sigma_vox``, zero outside the crop so
    mass crossing the boundary is lost), decays by ``decay``, then adds the new concentration to ``dose``.

    Mass balance (checked by tests): ``released == mass + decayed + lost_boundary`` to float precision.
    """

    def __init__(self, shape: Sequence[int], *, release_sigma_vox: float = 1.0, diffuse_sigma_vox: float = 1.0,
                 decay: float = 0.02) -> None:
        self.shape = tuple(int(s) for s in shape)
        self.conc = np.zeros(self.shape, dtype=np.float64)
        self.dose = np.zeros(self.shape, dtype=np.float64)
        self.diffuse_sigma_vox = float(diffuse_sigma_vox)
        self.decay = float(decay)
        self.released = 0.0
        self.decayed = 0.0
        self.lost_boundary = 0.0
        self.decayed_in_region = 0.0  # decayed mass inside the accounting region (scorer's tumour mask)
        r = max(1, int(math.ceil(3.0 * release_sigma_vox)))
        ax = np.arange(-r, r + 1, dtype=np.float64)
        g = np.exp(-0.5 * (ax / release_sigma_vox) ** 2)
        k = g[:, None, None] * g[None, :, None] * g[None, None, :]
        self._kernel = k / k.sum()
        self._r = r

    @property
    def mass(self) -> float:
        return float(self.conc.sum())

    def deposit(self, pos: Vox, mass: float) -> None:
        if mass <= 0.0:
            return
        r = self._r
        lo = [pos[i] - r for i in range(3)]
        hi = [pos[i] + r + 1 for i in range(3)]
        dst = tuple(slice(max(lo[i], 0), min(hi[i], self.shape[i])) for i in range(3))
        src = tuple(slice(dst[i].start - lo[i], dst[i].stop - lo[i]) for i in range(3))
        part = self._kernel[src]
        self.conc[dst] += mass * part
        self.released += mass
        self.lost_boundary += mass * (1.0 - float(part.sum()))

    def step(self, region: Optional[np.ndarray] = None) -> None:
        before = float(self.conc.sum())
        if before > 0.0:
            self.conc = gaussian_filter(self.conc, self.diffuse_sigma_vox, mode="constant", cval=0.0, truncate=3.0)
            np.maximum(self.conc, 0.0, out=self.conc)
            after = float(self.conc.sum())
            self.lost_boundary += before - after
            if region is not None:
                self.decayed_in_region += float(self.conc[region].sum()) * self.decay
            self.conc *= 1.0 - self.decay
            self.decayed += after * self.decay
        self.dose += self.conc

    def mass_in(self, region: np.ndarray) -> float:
        return float(self.conc[region].sum())

    def mass_balance_residual(self) -> float:
        return self.released - (self.mass + self.decayed + self.lost_boundary)


# --- A* ---------------------------------------------------------------------

_SQ2, _SQ3 = math.sqrt(2.0), math.sqrt(3.0)
_OFFSETS: List[Tuple[int, int, int, float]] = [
    (dx, dy, dz, math.sqrt(dx * dx + dy * dy + dz * dz))
    for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]


class Pathfinder:
    """Weighted A* on the brain mask, 26-neighbour moves (diagonals allowed, Euclidean step cost).

    26-neighbour was chosen over 6 so a squad's speed is one *voxel step* per tick regardless of direction (the
    straight-line shortcut is the natural path through a roughly convex brain). The heuristic is the exact octile
    distance (admissible and consistent on the open grid); ``weight`` > 1 trades optimality for speed
    (cost <= weight * optimal). Deterministic: ties broken by insertion counter.
    """

    def __init__(self, brain: np.ndarray, *, weight: float = 1.5, max_expand: int = 40000) -> None:
        self.shape = tuple(int(s) for s in brain.shape)
        padded = np.pad(brain.astype(np.uint8), 1)
        self._ps = padded.shape
        self._mask = bytes(padded.ravel())
        self.weight = float(weight)
        self.max_expand = int(max_expand)
        self._brain = brain
        self.brain = brain

    @staticmethod
    def _h(dx: int, dy: int, dz: int) -> float:
        a, b, c = sorted((abs(dx), abs(dy), abs(dz)))
        return (_SQ3 - _SQ2) * a + (_SQ2 - 1.0) * b + c

    def path(self, start: Vox, goal: Vox) -> List[Vox]:
        """Voxels from just after ``start`` to ``goal`` (or to the closest point reached if the budget runs out)."""
        if start == goal:
            return []
        _, sy, sz = self._ps
        mask = self._mask
        w = self.weight
        h = self._h
        start_p = (start[0] + 1, start[1] + 1, start[2] + 1)
        goal_p = (goal[0] + 1, goal[1] + 1, goal[2] + 1)
        h0 = h(goal_p[0] - start_p[0], goal_p[1] - start_p[1], goal_p[2] - start_p[2])
        counter = 0
        heap = [(w * h0, 0, start_p)]
        g_cost = {start_p: 0.0}
        parent = {start_p: None}
        best = (h0, start_p)
        closed = set()
        expanded = 0
        end: Optional[Tuple[int, int, int]] = None
        while heap:
            _, _, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            if cur == goal_p:
                end = cur
                break
            closed.add(cur)
            expanded += 1
            if expanded > self.max_expand:
                break
            cx, cy, cz = cur
            gc = g_cost[cur]
            for dx, dy, dz, cost in _OFFSETS:
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                if not mask[(nx * sy + ny) * sz + nz]:
                    continue
                nxt = (nx, ny, nz)
                ng = gc + cost
                if ng < g_cost.get(nxt, 1e18):
                    g_cost[nxt] = ng
                    parent[nxt] = cur
                    hh = h(goal_p[0] - nx, goal_p[1] - ny, goal_p[2] - nz)
                    if hh < best[0]:
                        best = (hh, nxt)
                    counter += 1
                    heapq.heappush(heap, (ng + w * hh, counter, nxt))
        if end is None:
            end = best[1]
        out: List[Vox] = []
        node = end
        while node is not None and node != start_p:
            out.append((node[0] - 1, node[1] - 1, node[2] - 1))
            node = parent[node]
        out.reverse()
        return out

    def nearest_brain_voxel(self, point: Sequence[float]) -> Vox:
        """The brain voxel closest to ``point`` (itself if it is in the brain)."""
        p = [min(max(int(round(c)), 0), s - 1) for c, s in zip(point, self.shape)]
        if self._brain[tuple(p)]:
            return (p[0], p[1], p[2])
        r = 1
        while True:
            lo = [max(p[i] - r, 0) for i in range(3)]
            hi = [min(p[i] + r + 1, self.shape[i]) for i in range(3)]
            sub = self._brain[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            if sub.any():
                pts = np.argwhere(sub) + np.array(lo)
                d = ((pts - np.array(p)) ** 2).sum(axis=1)
                j = int(np.argmin(d))
                return (int(pts[j][0]), int(pts[j][1]), int(pts[j][2]))
            if all(lo[i] == 0 and hi[i] == self.shape[i] for i in range(3)):
                raise ValueError("empty brain mask")
            r += 2


# --- unit-puff response, boundary-exact ------------------------------------------------------------------


class PuffResponse:
    """Cumulative dose in a ``(2W+1)^3`` window around a unit puff, in a crop with absorbing faces.

    The world's physics is separable: the release kernel is a product of 1D Gaussians, diffusion is a Gaussian filter
    applied per axis with zero outside the crop, and decay is a scalar. So one tick of a puff is
    ``(1-decay)^t * A_t(x) * B_t(y) * C_t(z)`` where each factor is a 1D process with an absorbing end, and the window dose
    is the sum of the first ``n`` ticks. That is the same arithmetic as :class:`DrugField` (tests compare them), not an
    approximation: mass that leaves the crop, at release or by diffusion, never comes back, so a puff near a face (or
    corner) delivers less than the free-space kernel says. ``retained`` is the share of the puff that survives release.

    A face farther than ``edge_r`` voxels from the puff is treated as absent (room ``kern_r`` on that side, the same
    box the free-space kernel uses); nearer faces are exact. Window cells outside the crop are 0 (the part that was lost).
    """

    def __init__(self, *, release_sigma_vox: float, diffuse_sigma_vox: float, decay: float, ticks: int,
                 win_r: int = 16, kern_r: int = 24, edge_r: int = 14, cache: int = 512) -> None:
        self.win_r, self.kern_r, self.edge_r = int(win_r), int(kern_r), int(edge_r)
        self.sigma, self.decay, self.ticks = float(diffuse_sigma_vox), float(decay), int(ticks)
        r = max(1, int(math.ceil(3.0 * release_sigma_vox)))
        ax = np.arange(-r, r + 1, dtype=np.float64)
        g = np.exp(-0.5 * (ax / release_sigma_vox) ** 2)
        self._g, self._r = g / g.sum(), r  # 3D kernel / its sum == product of per-axis normalised 1D kernels
        self._weights = (1.0 - self.decay) ** np.arange(1, self.ticks + 1, dtype=np.float64)
        self._axes: dict = {}
        self._cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._cache_max = int(cache)

    def _axis_key(self, length: int, p: int) -> Tuple[int, int]:
        left, right = int(p), int(length) - 1 - int(p)
        return (left if left < self.edge_r else self.kern_r, right if right < self.edge_r else self.kern_r)

    def _axis(self, key: Tuple[int, int]):
        """(profiles (ticks+1, 2W+1), retained fraction) for a 1D puff with ``key`` voxels of room left and right."""
        hit = self._axes.get(key)
        if hit is not None:
            return hit
        lk, rk = key
        m = lk + rk + 1
        a = np.zeros(m, dtype=np.float64)
        for off in range(-self._r, self._r + 1):
            if 0 <= lk + off < m:
                a[lk + off] = self._g[off + self._r]
        retained = float(a.sum())
        w = self.win_r
        prof = np.zeros((self.ticks + 1, 2 * w + 1), dtype=np.float64)
        lo, hi = max(lk - w, 0), min(lk + w + 1, m)  # window cells that exist
        cur = a
        for t in range(self.ticks + 1):
            if t:
                cur = np.maximum(gaussian_filter1d(cur, self.sigma, mode="constant", cval=0.0, truncate=3.0), 0.0)
            prof[t, lo - (lk - w):hi - (lk - w)] = cur[lo:hi]
        self._axes[key] = (prof, retained)
        return self._axes[key]

    def retained(self, shape: Sequence[int], v: Vox) -> float:
        """Share of a unit puff released at ``v`` that stays inside the crop (the rest is lost at once)."""
        out = 1.0
        for ax in range(3):
            out *= self._axis(self._axis_key(shape[ax], v[ax]))[1]
        return out

    def window(self, shape: Sequence[int], v: Vox, n: int) -> np.ndarray:
        """Dose in the window around ``v`` from a unit puff released ``n`` ticks before the end (float32)."""
        n = int(min(max(n, 0), self.ticks))
        w = 2 * self.win_r + 1
        if n < 1:
            return np.zeros((w, w, w), dtype=np.float32)
        keys = tuple(self._axis_key(shape[ax], v[ax]) for ax in range(3))
        ck = (keys, n)
        hit = self._cache.get(ck)
        if hit is not None:
            self._cache.move_to_end(ck)
            return hit
        a, b, c = (self._axis(k)[0][1:n + 1] for k in keys)
        bc = (b[:, :, None] * c[:, None, :]).reshape(n, w * w)
        out = ((a * self._weights[:n, None]).T @ bc).reshape(w, w, w).astype(np.float32)
        for ax in range(3):  # window cells beyond a face are outside the crop: lost, not delivered
            lo, hi = max(self.win_r - int(v[ax]), 0), min(w, self.win_r + int(shape[ax]) - int(v[ax]))
            idx = [slice(None)] * 3
            idx[ax] = slice(0, lo)
            out[tuple(idx)] = 0.0
            idx[ax] = slice(hi, w)
            out[tuple(idx)] = 0.0
        self._cache[ck] = out
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return out
