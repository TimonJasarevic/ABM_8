"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from misinfo_abm.config import ModelConfig
from misinfo_abm.experiments import run_factorial, save_factorial_results
from misinfo_abm.model import MisinformationModel
from misinfo_abm.plotting import plot_factorial_results
from misinfo_abm.sensitivity import run_morris


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agents", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seed-mode", choices=["hub", "random", "peripheral"], default="random"
    )
    parser.add_argument(
        "--hub-strategy",
        choices=["cooperative", "non_verifying", "adaptive"],
        default="cooperative",
    )
    parser.add_argument(
        "--payoff-mode", choices=["normalized", "accumulated"], default="normalized"
    )
    parser.add_argument(
        "--network", choices=["scale_free", "erdos_renyi"], default="scale_free"
    )


def _config_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        n_agents=args.agents,
        n_episodes=args.episodes,
        seed=args.seed,
        seed_mode=args.seed_mode,
        hub_strategy=args.hub_strategy,
        payoff_mode=args.payoff_mode,
        network_type=args.network,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="misinfo-abm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one model configuration")
    _add_common_arguments(run_parser)
    run_parser.add_argument("--output", default="results/single_run_episodes.csv")
    run_parser.add_argument("--ticks-output", default="results/single_run_ticks.csv")

    factorial_parser = subparsers.add_parser(
        "factorial", help="Run the core hub-seeding factorial experiment"
    )
    _add_common_arguments(factorial_parser)
    factorial_parser.add_argument("--replications", type=int, default=20)
    factorial_parser.add_argument("--output", default="results/factorial.csv")
    factorial_parser.add_argument("--no-payoff-comparison", action="store_true")

    plot_parser = subparsers.add_parser("plot", help="Plot factorial output")
    plot_parser.add_argument("--input", default="results/factorial.csv")
    plot_parser.add_argument("--output-dir", default="results/plots")

    morris_parser = subparsers.add_parser("morris", help="Run Morris sensitivity screening")
    _add_common_arguments(morris_parser)
    morris_parser.add_argument("--trajectories", type=int, default=20)
    morris_parser.add_argument("--replications", type=int, default=5)
    morris_parser.add_argument("--output", default="results/morris_indices.csv")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "plot":
        paths = plot_factorial_results(args.input, args.output_dir)
        for path in paths:
            print(path)
        return

    config = _config_from_args(args)

    if args.command == "run":
        model = MisinformationModel(config)
        model.run_model()
        episode_path = Path(args.output)
        tick_path = Path(args.ticks_output)
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        tick_path.parent.mkdir(parents=True, exist_ok=True)
        model.episode_dataframe().to_csv(episode_path, index=False)
        model.tick_dataframe().to_csv(tick_path, index=False)
        print(episode_path)
        print(tick_path)
        return

    if args.command == "factorial":
        data = run_factorial(
            config,
            replications=args.replications,
            include_payoff_comparison=not args.no_payoff_comparison,
        )
        path = save_factorial_results(data, args.output)
        print(path)
        return

    if args.command == "morris":
        result = run_morris(
            config,
            trajectories=args.trajectories,
            replications=args.replications,
            output=args.output,
        )
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
