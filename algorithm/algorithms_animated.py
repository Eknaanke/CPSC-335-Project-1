import random
import time
from dataclasses import dataclass
from typing import List, Generator, Iterable, Optional, Tuple, Dict

# --- Matplotlib import ---
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ========================== Stats & Array Proxy ===========================

@dataclass
class SortStats:
    comparisons: int = 0
    array_accesses: int = 0
    start_time: float = 0.0

class ArrayProxy:
    """
    Wraps a list and counts ALL element reads/writes/iteration as "array accesses".
    Use .snapshot() to copy for drawing WITHOUT incrementing counts.
    """
    def __init__(self, data, stats: SortStats):
        self._a = list(data)
        self._s = stats

    def __len__(self):
        return len(self._a)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            rng = range(*idx.indices(len(self._a)))
            self._s.array_accesses += len(rng)
            return self._a[idx]
        else:
            self._s.array_accesses += 1
            return self._a[idx]

    def __setitem__(self, idx, val):
        if isinstance(idx, slice):
            if hasattr(val, "__len__"):
                self._s.array_accesses += len(val)
            else:
                self._s.array_accesses += 1
            self._a[idx] = val
        else:
            self._s.array_accesses += 1
            self._a[idx] = val

    def __iter__(self):
        for v in self._a:
            self._s.array_accesses += 1
            yield v

    def snapshot(self) -> List[int]:
        return list(self._a)

# ============================= Animation Core =============================

Highlight = Dict[int, str]  # index -> color (e.g., 'r', 'g', 'y')

def yield_state(a, highlight: Optional[Highlight] = None):
    """Yield a SAFE copy for drawing without counting accesses."""
    snap = a.snapshot() if hasattr(a, "snapshot") else a[:]
    yield (snap, highlight or {})

def animate_sort(
    data,
    frames: Iterable[Tuple[List[int], Highlight]],
    title: str,
    interval_ms: int = 10,
    final_sorted_curve: bool = True,
    stats: Optional[SortStats] = None,
    speed: float = 1.0,     # e.g., 5.0 for 5×
    frame_skip: int = 1,    # e.g., 2, 3, 5 for faster playback
    blit: bool = False,
    use_line: bool = False, # True = MUCH faster (no per-bar colors)
):
    arr0 = data.snapshot() if hasattr(data, "snapshot") else list(data)
    n = len(arr0)

    # Effective playback controls
    eff_interval = max(1, int(interval_ms / max(0.1, speed)))
    eff_skip = max(1, int(frame_skip))

    fig, ax = plt.subplots()
    if use_line:
        line, = ax.plot(range(n), arr0, lw=2)
    else:
        bars = ax.bar(range(n), arr0, align="edge")
        default_color = bars[0].get_facecolor() if n else (0, 0, 1, 1)
    ax.set_xlim(0, n)
    ax.set_ylim(0, max(arr0) * 1.1 if arr0 else 1)
    ax.set_title(title)

    hud = ax.text(
        0.01, 0.99,
        "",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=10, family="monospace",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=4),
    )

    if stats is None:
        stats = SortStats()
    stats.start_time = time.time()

    def _apply_colors(h: Highlight):
        if use_line:
            return
        # reset colors
        for bar in bars:
            bar.set_color(default_color)
        # apply
        for i, c in h.items():
            if 0 <= i < n:
                bars[i].set_color(c)

    def frame_generator():
        for idx, frame in enumerate(frames):
            if (idx % eff_skip) != 0:
                continue
            yield frame

    def update(frame: Tuple[List[int], Highlight]):
        heights, highlight = frame
        if use_line:
            line.set_ydata(heights)
        else:
            for bar, h in zip(bars, heights):
                bar.set_height(h)
            _apply_colors(highlight)

        hud.set_text(
            f"comparisons:   {stats.comparisons}\n"
            f"array accesses: {stats.array_accesses}"
        )
        if blit:
            return (line, hud) if use_line else (*bars, hud)
        return (line,) if use_line else bars

    def on_finish(*_):
        if final_sorted_curve:
            full = data.snapshot() if hasattr(data, "snapshot") else list(data)
            plt.figure()
            plt.plot(sorted(full), linewidth=2)
            plt.title(f"Final Sorted Curve — {title}")
            plt.xlabel("Index"); plt.ylabel("Value")
            plt.tight_layout(); plt.show()

    ani = FuncAnimation(
        fig, update, frames=frame_generator(),
        interval=eff_interval, blit=blit, repeat=False
    )
    ani._stop = (lambda f=ani._stop: (on_finish(), f()))
    plt.tight_layout()
    plt.show()

# ============================ Visual Generators ===========================

# 1) Bubble Sort (stable)
def bubble_sort_visual(a, stats: Optional[SortStats] = None) -> Generator[Tuple[List[int], Highlight], None, None]:
    if stats is None: stats = SortStats()
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            stats.comparisons += 1
            yield from yield_state(a, {j: "y", j + 1: "y"})
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                yield from yield_state(a, {j: "r", j + 1: "r"})
                swapped = True
        if not swapped:
            break
    yield from yield_state(a, {})

# 2) Insertion Sort (stable)
def insertion_sort_visual(a, stats: Optional[SortStats] = None) -> Generator[Tuple[List[int], Highlight], None, None]:
    if stats is None: stats = SortStats()
    for i in range(1, len(a)):
        key = a[i]
        yield from yield_state(a, {i: "g"})
        j = i - 1
        while j >= 0:
            stats.comparisons += 1
            if a[j] > key:
                a[j + 1] = a[j]
                yield from yield_state(a, {j + 1: "r"})
                j -= 1
            else:
                break
        a[j + 1] = key
        yield from yield_state(a, {j + 1: "g"})
    yield from yield_state(a, {})

# 3) Merge Sort (stable)
def merge_sort_visual(a, stats: Optional[SortStats] = None) -> Generator[Tuple[List[int], Highlight], None, None]:
    if stats is None: stats = SortStats()
    tmp = a.snapshot() if hasattr(a, "snapshot") else a[:]

    def _merge(l: int, m: int, r: int):
        i, j, k = l, m + 1, l
        while i <= m and j <= r:
            stats.comparisons += 1
            if tmp[i] <= tmp[j]:
                a[k] = tmp[i]; i += 1
            else:
                a[k] = tmp[j]; j += 1
            yield from yield_state(a, {k: "r"}); k += 1
        while i <= m:
            a[k] = tmp[i]; i += 1
            yield from yield_state(a, {k: "r"}); k += 1
        while j <= r:
            a[k] = tmp[j]; j += 1
            yield from yield_state(a, {k: "r"}); k += 1

    def _sort(l: int, r: int):
        if l >= r: return
        m = (l + r) // 2
        yield from _sort(l, m)
        yield from _sort(m + 1, r)
        # copy current range to tmp without counting accesses
        block = (a.snapshot() if hasattr(a, "snapshot") else a[:])[l:r+1]
        tmp[l:r+1] = block
        yield from yield_state(a, {i: "y" for i in range(l, r + 1)})
        yield from _merge(l, m, r)

    yield from _sort(0, len(a) - 1)
    yield from yield_state(a, {})

# 4) Heap Sort (not stable)
def heap_sort_visual(a, stats: Optional[SortStats] = None) -> Generator[Tuple[List[int], Highlight], None, None]:
    if stats is None: stats = SortStats()

    def sift_down(start: int, end: int):
        root = start
        while (left := 2 * root + 1) <= end:
            right = left + 1
            largest = root
            stats.comparisons += 1
            if a[left] > a[largest]: largest = left
            if right <= end:
                stats.comparisons += 1
                if a[right] > a[largest]: largest = right
            if largest == root: return
            a[root], a[largest] = a[largest], a[root]
            yield from yield_state(a, {root: "r", largest: "r"})
            root = largest

    n = len(a)
    for i in range(n // 2 - 1, -1, -1):
        yield from sift_down(i, n - 1)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        yield from yield_state(a, {0: "g", end: "g"})
        yield from sift_down(0, end - 1)
    yield from yield_state(a, {})

# 5) Quick Sort (not stable)
def quick_sort_visual(a, stats: Optional[SortStats] = None) -> Generator[Tuple[List[int], Highlight], None, None]:
    if stats is None: stats = SortStats()

    def partition(lo: int, hi: int) -> Generator[Tuple[List[int], Highlight], Tuple[int, int], None]:
        pivot = a[(lo + hi) // 2]
        i, j = lo, hi
        while i <= j:
            while True:
                stats.comparisons += 1
                if not (a[i] < pivot): break
                i += 1
                yield from yield_state(a, {i: "y"})
            while True:
                stats.comparisons += 1
                if not (a[j] > pivot): break
                j -= 1
                yield from yield_state(a, {j: "y"})
            if i <= j:
                a[i], a[j] = a[j], a[i]
                yield from yield_state(a, {i: "r", j: "r"})
                i += 1; j -= 1
        return i, j

    def sort(lo: int, hi: int):
        if lo >= hi: return
        i, j = yield from partition(lo, hi)
        if lo < j: yield from sort(lo, j)
        if i < hi: yield from sort(i, hi)

    yield from sort(0, len(a) - 1)
    yield from yield_state(a, {})

# 6) Counting Sort (stable; distribution sort → comparisons = 0)
def counting_sort_visual(a, stats: Optional[SortStats] = None) -> Generator[Tuple[List[int], Highlight], None, None]:
    if stats is None: stats = SortStats()
    if not a:
        yield from yield_state(a, {}); return
    # find bounds without inflating access count
    snap = a.snapshot() if hasattr(a, "snapshot") else list(a)
    mn, mx = min(snap), max(snap)
    k = mx - mn + 1
    count = [0] * k
    out = [0] * len(snap)

    for v in a:
        count[v - mn] += 1
        yield from yield_state(a, {})
    for i in range(1, k):
        count[i] += count[i - 1]
        yield from yield_state(a, {})
    for i in range(len(snap) - 1, -1, -1):
        v = snap[i]
        count[v - mn] -= 1
        pos = count[v - mn]
        out[pos] = v
        a[pos] = v
        yield from yield_state(a, {pos: "r"})
    yield from yield_state(a, {})

# 7) Radix Sort (LSD) — supports negatives; comparisons ~0
def radix_sort_visual(a, base: int = 10, stats: Optional[SortStats] = None) -> Generator[Tuple[List[int], Highlight], None, None]:
    if stats is None: stats = SortStats()
    if not a:
        yield from yield_state(a, {}); return

    def counting_digit_pass(arr: List[int], exp: int):
        n = len(arr)
        out = [0] * n
        count = [0] * base
        for v in arr:
            d = (v // exp) % base
            count[d] += 1
            yield from yield_state(a, {})
        for i in range(1, base):
            count[i] += count[i - 1]
            yield from yield_state(a, {})
        for i in range(n - 1, -1, -1):
            v = arr[i]
            d = (v // exp) % base
            count[d] -= 1
            out[count[d]] = v
            a[count[d]] = v
            yield from yield_state(a, {count[d]: "r"})
        arr[:] = out

    snap = a.snapshot() if hasattr(a, "snapshot") else list(a)
    neg = [-x for x in snap if x < 0]
    pos = [x for x in snap if x >= 0]

    if neg:
        m = max(neg); exp = 1
        while m // exp > 0:
            yield from counting_digit_pass(neg, exp); exp *= base
    if pos:
        m = max(pos) if pos else 0; exp = 1
        while m // exp > 0:
            yield from counting_digit_pass(pos, exp); exp *= base

    neg_sorted = [-x for x in reversed(neg)]
    idx = 0
    for v in neg_sorted + pos:
        a[idx] = v
        yield from yield_state(a, {idx: "r"})
        idx += 1
    yield from yield_state(a, {})

# 8) Bucket Sort (range-agnostic; comparisons ~0)
def bucket_sort_visual(a, stats: Optional[SortStats] = None) -> Generator[Tuple[List[int], Highlight], None, None]:
    if stats is None: stats = SortStats()
    n = len(a)
    if n <= 1:
        yield from yield_state(a, {}); return

    snap = a.snapshot() if hasattr(a, "snapshot") else list(a)
    mn, mx = min(snap), max(snap)
    if mn == mx:
        yield from yield_state(a, {}); return

    span = mx - mn
    buckets: List[List[float]] = [[] for _ in range(n)]
    for x in snap:
        idx = int((x - mn) / span * n)
        if idx == n: idx = n - 1
        buckets[idx].append(x)
        yield from yield_state(a, {})
    for b in buckets:
        b.sort()
    i = 0
    for b in buckets:
        for v in b:
            a[i] = v
            yield from yield_state(a, {i: "r"})
            i += 1
    yield from yield_state(a, {})

# 9) QuickSelect (visualize partitions + k)
def quickselect_visual(a, k: int, stats: Optional[SortStats] = None) -> Generator[Tuple[List[int], Highlight], None, None]:
    if stats is None: stats = SortStats()
    if not a:
        yield from yield_state(a, {}); return

    def partition(lo: int, hi: int) -> Generator[Tuple[List[int], Highlight], int, None]:
        pivot = a[hi]
        i = lo
        for j in range(lo, hi):
            stats.comparisons += 1
            yield from yield_state(a, {hi: "y", j: "y", k: "g"})
            if a[j] <= pivot:
                a[i], a[j] = a[j], a[i]
                yield from yield_state(a, {i: "r", j: "r", hi: "y", k: "g"})
                i += 1
        a[i], a[hi] = a[hi], a[i]
        yield from yield_state(a, {i: "r", hi: "r", k: "g"})
        return i

    lo, hi = 0, len(a) - 1
    while lo <= hi:
        p = yield from partition(lo, hi)
        if p == k:
            yield from yield_state(a, {p: "g"})
            break
        elif p > k:
            hi = p - 1
        else:
            lo = p + 1
    yield from yield_state(a, {k: "g"})

# =========================== Registry & Driver ============================

VISUAL_FUNCS = {
    "bubble": bubble_sort_visual,
    "insertion": insertion_sort_visual,
    "merge": merge_sort_visual,
    "heap": heap_sort_visual,
    "quick": quick_sort_visual,
    "counting": counting_sort_visual,
    "radix": radix_sort_visual,
    "bucket": bucket_sort_visual,
    "quickselect": quickselect_visual,  # needs k
}

def main():
    print("\nAlgorithms:")
    print(", ".join(VISUAL_FUNCS.keys()))
    algo = input("\nChoose algorithm: ").strip().lower()

    if algo not in VISUAL_FUNCS:
        print("Unknown algorithm.")
        return

    n_in = input("Array size (e.g., 64): ").strip()
    n = int(n_in) if n_in.isdigit() and int(n_in) > 0 else 64

    stats = SortStats()

    # Generate data
    if algo == "bucket":
        raw = [random.random() * 100 for _ in range(n)]
    else:
        raw = [random.randint(1, 999) for _ in range(n)]

    data = ArrayProxy(raw, stats)

    # Speed knobs (change here or prompt)
    SPEED = 5.0        # 5× playback
    FRAME_SKIP = 2     # set to 2/3/5 for even faster
    USE_LINE = False   # True = faster, no per-bar highlighting
    BLIT = False       # True if backend supports it

    if algo == "quickselect":
        k_in = input(f"k index (0..{n-1}, default median): ").strip()
        k = int(k_in) if k_in.isdigit() and 0 <= int(k_in) < n else n // 2
        frames = VISUAL_FUNCS[algo](data, k, stats=stats)
        title = f"QuickSelect (k={k})"
        animate_sort(
            data, frames, title=title,
            interval_ms=10, final_sorted_curve=True,
            stats=stats, speed=SPEED, frame_skip=FRAME_SKIP, blit=BLIT, use_line=USE_LINE
        )
    else:
        frames = VISUAL_FUNCS[algo](data, stats=stats)
        title = algo.title() + " Sort"
        animate_sort(
            data, frames, title=title,
            interval_ms=10, final_sorted_curve=True,
            stats=stats, speed=SPEED, frame_skip=FRAME_SKIP, blit=BLIT, use_line=USE_LINE
        )

if __name__ == "__main__":
    while True:
        main()
        again = input("\nRun again? (y/n): ").strip().lower()
        if again != "y":
            print("Goodbye!")
            break
