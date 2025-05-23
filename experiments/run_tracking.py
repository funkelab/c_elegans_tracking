import argparse

import toml

from c_elegans_utils.experiment import Experiment
from c_elegans_utils.tracking import create_cand_graph, evaluate, run_motile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument(
        "--dataset",
        choices=[
            "post_twitching_neurons",
            "lin_26_0208_Pos4",
            "lin_26_0213_Pos3",
            "lin_26_0315_Pos4",
        ],
    )
    parser.add_argument("--time-range", nargs=2, type=int, default=None)
    parser.add_argument("-c", "--cluster", action="store_true")
    parser.add_argument("-n", "--name")
    args = parser.parse_args()
    config = toml.load(args.config)
    if args.dataset is not None:
        config["dataset"]["name"] = args.dataset
    if args.name is not None:
        config["name"] = args.name
    if args.time_range is not None:
        config["dataset"]["time_range"] = args.time_range
    # this makes an experiment directory with timestamp uuid and saves config
    name = config["name"]
    experiment = Experiment(name, config, cluster=args.cluster)
    ds = experiment.dataset
    solver_params = experiment.solver_params
    cand_graph_params = experiment.cand_graph_params

    cand_graph, conflict_sets = create_cand_graph(cand_graph_params, ds)
    print(
        f"cand graph has {len(cand_graph.nodes)} nodes, {len(cand_graph.edges)} edges, "
        f"and {len(conflict_sets)} conflict sets"
    )
    experiment.candidate_graph = cand_graph

    solver, soln_graph = run_motile(cand_graph, conflict_sets, solver_params)
    print(
        f"soln graph has {soln_graph.number_of_nodes()} nodes and "
        f"{soln_graph.number_of_edges()} edges"
    )

    experiment.solution_graph = soln_graph
    results = evaluate(experiment)
    if results is not None:
        experiment.results = results
    print(results)
    print(experiment.exp_base_dir)
