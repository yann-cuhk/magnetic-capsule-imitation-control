import argparse

from Control_Package.ADP import (
    DEFAULT_DATA_PATH,
    DEFAULT_POLICY_PATH,
    load_adp_samples,
    save_policy,
    solve_adp_policy,
)


def _weights(value, expected):
    if value is None:
        return None
    weights = [float(item) for item in value.split(",")]
    if len(weights) != expected:
        raise argparse.ArgumentTypeError(f"expected {expected} comma-separated weights")
    return weights


def main():
    parser = argparse.ArgumentParser(description="Solve the capsule ADP policy from offline samples.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--policy", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--nu-grid", type=int, default=3)
    parser.add_argument("--policy-samples", type=int, default=800)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--q-weights", type=lambda v: _weights(v, 12), default=None)
    parser.add_argument("--r-weights", type=lambda v: _weights(v, 6), default=None)
    args = parser.parse_args()

    t, x, u = load_adp_samples(args.data)
    result = solve_adp_policy(
        t,
        x,
        u,
        q_weights=args.q_weights,
        r_weights=args.r_weights,
        max_iter=args.max_iter,
        tol=args.tol,
        nu_grid=args.nu_grid,
        policy_sample_count=args.policy_samples,
        ridge=args.ridge,
    )
    save_policy(args.policy, result)
    print(
        "[MagRobotADP] "
        f"samples={result['sample_count']} "
        f"iterations={result['iterations']} "
        f"converged={bool(result['converged'])} "
        f"policy={args.policy}"
    )


if __name__ == "__main__":
    main()
