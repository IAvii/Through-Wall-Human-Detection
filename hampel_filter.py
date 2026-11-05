import numpy as np

from config import HAMPEL_WINDOW, HAMPEL_THRESHOLD

def hampel_filter(matrix, k=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD):
    if matrix.size == 0:
        return matrix.copy(), 0

    T, N = matrix.shape
    filtered = matrix.copy().astype(np.float64)
    outlier_count = 0

    # scale factor: 1.4826 makes mad consistent with std for normal distribution
    SCALE = 1.4826

    for col in range(N):
        column = filtered[:, col]

        for i in range(T):
            # determine window boundaries (handle edges)
            start = max(0, i - k)
            end = min(T, i + k + 1)

            window = column[start:end]

            # compute local median
            med = np.median(window)

            # compute mad (median absolute deviation)
            mad = np.median(np.abs(window - med))

            # estimate standard deviation from mad
            sigma = SCALE * mad

            # detect outlier: if sigma is 0, only exact duplicates pass
            if sigma < 1e-10:
                # all values in window are identical — no outlier possible
                # unless the value differs from median
                if abs(column[i] - med) > 1e-10:
                    column[i] = med
                    outlier_count += 1
            elif abs(column[i] - med) > threshold * sigma:
                column[i] = med
                outlier_count += 1

        filtered[:, col] = column

    return filtered, outlier_count

def hampel_filter_vectorized(matrix, k=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD):
    if matrix.size == 0:
        return matrix.copy(), 0

    T, N = matrix.shape
    # use original data for statistics, write replacements to output
    original = matrix.astype(np.float64)
    filtered = original.copy()
    outlier_count = 0
    SCALE = 1.4826

    for i in range(T):
        start = max(0, i - k)
        end = min(T, i + k + 1)

        # window from original data — prevents cascading
        window = original[start:end, :]

        # median across time axis for each column: shape (n,)
        med = np.median(window, axis=0)

        # mad for each column: shape (n,)
        mad = np.median(np.abs(window - med), axis=0)

        # sigma estimate
        sigma = SCALE * mad

        # current row values (from original)
        values = original[i, :]

        # detect outliers (handle sigma == 0 case)
        deviation = np.abs(values - med)
        is_outlier = np.where(
            sigma < 1e-10,
            deviation > 1e-10,         # sigma~0: outlier if not equal to median
            deviation > threshold * sigma  # normal case
        )

        # replace outliers with median in output only
        n_outliers = np.sum(is_outlier)
        if n_outliers > 0:
            filtered[i, is_outlier] = med[is_outlier]
            outlier_count += int(n_outliers)

    return filtered, outlier_count

if __name__ == "__main__":
    # self-test with synthetic data
    print("=== Hampel Filter Self-Test ===\n")

    # create clean data with known spikes
    np.random.seed(42)
    data = np.random.normal(50, 5, (500, 52)).astype(np.float64)
    original = data.copy()

    # inject spikes
    spikes = [(50, 0, 200), (150, 10, 200), (300, 25, 200)]
    for row, col, val in spikes:
        data[row, col] = val

    # test standard version
    cleaned, n_out = hampel_filter(data.copy())
    print(f"Standard filter — Outliers removed: {n_out}")
    for row, col, val in spikes:
        print(f"  Spike at [{row},{col}]: {val} → {cleaned[row, col]:.1f}  "
              f"{'✓ REMOVED' if cleaned[row, col] < 100 else '✗ MISSED'}")

    # test vectorized version
    cleaned_v, n_out_v = hampel_filter_vectorized(data.copy())
    print(f"\nVectorized filter — Outliers removed: {n_out_v}")
    for row, col, val in spikes:
        print(f"  Spike at [{row},{col}]: {val} → {cleaned_v[row, col]:.1f}  "
              f"{'✓ REMOVED' if cleaned_v[row, col] < 100 else '✗ MISSED'}")

    # verify non-spike values unchanged
    mask = np.ones_like(data, dtype=bool)
    for row, col, val in spikes:
        mask[row, col] = False
    max_diff = np.max(np.abs(original[mask] - cleaned[mask]))
    print(f"\nMax deviation in non-spike values: {max_diff:.6f}")
    print(f"Non-spike values preserved: {'✓ PASS' if max_diff < 0.1 else '✗ FAIL'}")

    print("\n=== Self-Test Complete ===")
