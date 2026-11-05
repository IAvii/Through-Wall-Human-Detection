import numpy as np

def preprocess(matrix):
        # handle edge case: all-zero or empty matrix
    if matrix.size == 0:
        return np.zeros(0), 0.0

    T, N = matrix.shape

    if T == 0 or N == 0:
        return np.zeros(T), 0.0

    # check for degenerate case: matrix is all zeros or constant
    if np.allclose(matrix, 0) or np.std(matrix) < 1e-10:
        return np.zeros(T), 0.0

    # step 1: remove static component (column-wise mean subtraction)
    # the mean across time for each subcarrier represents the constant
    # path — direct wall reflection, multipath from static objects
    column_means = matrix.mean(axis=0)  # shape (n,)
    H_dyn = matrix - column_means       # shape (t, n)

    # check if dynamic component is essentially zero
    if np.std(H_dyn) < 1e-10:
        return np.zeros(T), 0.0

    # step 2: svd decomposition
    # h_dyn = u @ diag(s) @ vt
    # u[:,0] is the temporal pattern of the dominant signal
    # s[0] is its magnitude
    # vt[0,:] is the subcarrier weight pattern
    try:
        U, S, Vt = np.linalg.svd(H_dyn, full_matrices=False)
    except np.linalg.LinAlgError:
        # svd failed to converge — return zero signal
        return np.zeros(T), 0.0

    # step 3: extract dominant signal
    # the first principal component captures the most correlated variation
    # across subcarriers — typically chest wall displacement from breathing
    dominant_signal = U[:, 0] * S[0]  # shape (t,)

    # step 4: compute explained variance percentage
    total_variance = np.sum(S ** 2)
    if total_variance < 1e-10:
        explained_variance = 0.0
    else:
        explained_variance = (S[0] ** 2 / total_variance) * 100.0

    return dominant_signal, explained_variance

if __name__ == "__main__":
    # self-test
    print("=== Preprocessor Self-Test ===\n")

    # test 1: static room (no breathing)
    print("Test 1: Static room (no motion)")
    np.random.seed(42)
    static = np.ones((500, 52)) * 50 + np.random.normal(0, 0.5, (500, 52))
    signal, var = preprocess(static)
    print(f"  Signal std: {np.std(signal):.4f} (expect < 5.0)")
    print(f"  Explained variance: {var:.2f}%")
    print(f"  Result: {'✓ PASS' if np.std(signal) < 5.0 else '✗ FAIL'}\n")

    # test 2: breathing signal
    print("Test 2: Breathing at 0.25 Hz (15 BPM)")
    t = np.linspace(0, 10, 500)
    breathing = static.copy()
    breathing[:, 0] += np.sin(2 * np.pi * 0.25 * t) * 20
    signal, var = preprocess(breathing)
    print(f"  Signal std: {np.std(signal):.4f}")
    print(f"  Explained variance: {var:.2f}%")
    print(f"  Result: {'✓ PASS' if var > 30.0 else '✗ FAIL'}\n")

    # test 3: all zeros
    print("Test 3: All-zero matrix (edge case)")
    zeros = np.zeros((500, 52))
    signal, var = preprocess(zeros)
    print(f"  Signal norm: {np.linalg.norm(signal):.4f}")
    print(f"  Result: {'✓ PASS' if np.allclose(signal, 0) else '✗ FAIL'}\n")

    # test 4: empty matrix
    print("Test 4: Empty matrix (edge case)")
    empty = np.zeros((0, 52))
    signal, var = preprocess(empty)
    print(f"  Signal length: {len(signal)}")
    print(f"  Result: {'✓ PASS' if len(signal) == 0 else '✗ FAIL'}")

    print("\n=== Self-Test Complete ===")
