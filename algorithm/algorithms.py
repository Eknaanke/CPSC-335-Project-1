import time
import random
import matplotlib.pyplot as plt
from typing import List

# ------------------------------ Bubble Sort ------------------------------
def bubble_sort(arr: List[int]) -> List[int]:
    """
    Bubble Sort: repeatedly swaps adjacent out-of-order pairs.
    Time: O(n^2) worst/avg, O(n) best (already sorted).
    Space: O(1). Stable: Yes.
    """
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


# ------------------------------ Bucket Sort ------------------------------
def bucket_sort(arr: List[float]) -> List[float]:
    """
    Bucket Sort: distribute elements into buckets, sort each, then concatenate.
    Time: O(n + k) average, O(n^2) worst (all in one bucket).
    Space: O(n + k). Stable if stable sub-sorts are used.
    """
    n = len(arr)
    if n == 0:
        return arr
    buckets = [[] for _ in range(n)]
    for x in arr:
        idx = int(n * x)
        if idx == n:
            idx = n - 1
        buckets[idx].append(x)
    for i in range(n):
        buckets[i].sort()
    result = []
    for b in buckets:
        result.extend(b)
    return result


# ------------------------------ Counting Sort ------------------------------
def counting_sort(arr: List[int]) -> List[int]:
    """
    Counting Sort: counts occurrences, then reconstructs output.
    Time: O(n + k), Space: O(k). Stable if implemented carefully.
    Assumes integer input within a known small range.
    """
    if not arr:
        return []
    max_val = max(arr)
    min_val = min(arr)
    k = max_val - min_val + 1
    count = [0] * k
    for num in arr:
        count[num - min_val] += 1
    output = []
    for i, freq in enumerate(count):
        value = i + min_val
        output.extend([value] * freq)
    return output


# ------------------------------ Heap Sort ------------------------------
def heap_sort(arr: List[int]) -> List[int]:
    """
    Heap Sort: builds a max heap, then extracts max elements.
    Time: O(n log n) worst/avg/best. Space: O(1). Not stable.
    """
    def sift_down(a, start, end):
        root = start
        while (left := 2 * root + 1) <= end:
            right = left + 1
            largest = root
            if a[left] > a[largest]:
                largest = left
            if right <= end and a[right] > a[largest]:
                largest = right
            if largest == root:
                break
            a[root], a[largest] = a[largest], a[root]
            root = largest

    def build_max_heap(a):
        n = len(a)
        for i in range(n // 2 - 1, -1, -1):
            sift_down(a, i, n - 1)

    a = arr
    n = len(a)
    build_max_heap(a)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift_down(a, 0, end - 1)
    return a


# ------------------------------ Insertion Sort ------------------------------
def insertion_sort(arr: List[int]) -> List[int]:
    """
    Insertion Sort: builds sorted portion by inserting elements one by one.
    Time: O(n^2) worst/avg, O(n) best (already sorted).
    Space: O(1). Stable: Yes.
    """
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


# ------------------------------ Merge Sort ------------------------------
def merge_sort(arr: List[int]) -> List[int]:
    """
    Merge Sort: divide-and-conquer, recursively sort halves and merge.
    Time: O(n log n) worst/avg/best. Space: O(n). Stable: Yes.
    """
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left_half = merge_sort(arr[:mid])
    right_half = merge_sort(arr[mid:])
    return merge(left_half, right_half)

def merge(left: List[int], right: List[int]) -> List[int]:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# ------------------------------ QuickSelect ------------------------------
def partition(arr: List[int], low: int, high: int) -> int:
    pivot = arr[high]
    i = low
    for j in range(low, high):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[high] = arr[high], arr[i]
    return i

def quick_select(arr: List[int], low: int, high: int, k: int) -> int:
    """
    QuickSelect: partition-based selection to find kth element.
    Time: O(n) average, O(n^2) worst. Space: O(1).
    """
    if low <= high:
        pi = partition(arr, low, high)
        if pi == k:
            return arr[pi]
        elif pi > k:
            return quick_select(arr, low, pi - 1, k)
        else:
            return quick_select(arr, pi + 1, high, k)


# ------------------------------ Quick Sort ------------------------------
def quick_sort(arr: List[int]) -> List[int]:
    """
    Quick Sort: partition around pivot, then recursively sort parts.
    Time: O(n log n) average, O(n^2) worst. Space: O(log n) recursion.
    Not stable.
    """
    def partition(low, high):
        pivot = arr[(low + high) // 2]
        i, j = low, high
        while i <= j:
            while arr[i] < pivot:
                i += 1
            while arr[j] > pivot:
                j -= 1
            if i <= j:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
                j -= 1
        return i, j

    def sort(low, high):
        if low < high:
            i, j = partition(low, high)
            sort(low, j)
            sort(i, high)

    sort(0, len(arr) - 1)
    return arr


# ------------------------------ Radix Sort (LSD) ------------------------------
def counting_sort_by_digit(a: List[int], exp: int, base: int = 10):
    n = len(a)
    output = [0] * n
    count = [0] * base
    for i in range(n):
        index = (a[i] // exp) % base
        count[index] += 1
    for d in range(1, base):
        count[d] += count[d - 1]
    for i in range(n - 1, -1, -1):
        digit = (a[i] // exp) % base
        pos = count[digit] - 1
        output[pos] = a[i]
        count[digit] -= 1
    for i in range(n):
        a[i] = output[i]

def radix_sort_lsd_nonneg(a: List[int], base: int = 10) -> List[int]:
    if not a:
        return a
    max_val = max(a)
    exp = 1
    while max_val // exp > 0:
        counting_sort_by_digit(a, exp, base)
        exp *= base
    return a

def radix_sort_lsd(a: List[int], base: int = 10) -> List[int]:
    """
    Radix Sort (LSD): sorts by digits using counting sort as subroutine.
    Handles negatives by separating, sorting, and merging.
    Time: O(nk), Space: O(n + k). Stable.
    """
    if not a:
        return a
    neg = [-x for x in a if x < 0]
    pos = [x for x in a if x >= 0]
    if neg:
        radix_sort_lsd_nonneg(neg, base)
    if pos:
        radix_sort_lsd_nonneg(pos, base)
    neg_sorted = [-x for x in reversed(neg)]
    return neg_sorted + pos


# ------------------------------ Registry ------------------------------
PURE_FUNCS = {
    "bubble": bubble_sort,
    "bucket": bucket_sort,
    "counting": counting_sort,
    "heap": heap_sort,
    "insertion": insertion_sort,
    "merge": merge_sort,
    "quick": quick_sort,
    "radix": radix_sort_lsd,
}

# ------------------------------ Runtime Comparison ------------------------------
def compare_algorithms(algos, data):
    results = {}
    for name in algos:
        func = PURE_FUNCS.get(name)
        if not func:
            print(f"Unknown algorithm: {name}")
            continue
        test_data = data.copy()
        start = time.time()
        func(test_data)
        end = time.time()
        runtime = end - start
        results[name] = runtime
        print(f"{name.title()} Sort finished in {runtime:.6f} seconds")
    return results

def plot_results(results):
    names = list(results.keys())
    times = list(results.values())
    plt.bar(names, times)
    plt.ylabel("Runtime (seconds)")
    plt.title("Sorting Algorithm Comparison")
    plt.show()

# ------------------------------ Main ------------------------------
if __name__ == "__main__":
    algos = input("Enter algorithms (comma separated): ").strip().lower().split(",")
    choice = input("Enter numbers separated by space (or type 'random'): ").strip()
    if choice == "random":
        data = [random.randint(1, 1000) for _ in range(5000)]
    else:
        data = list(map(int, choice.split()))
    results = compare_algorithms([a.strip() for a in algos], data)
    plot_results(results)
