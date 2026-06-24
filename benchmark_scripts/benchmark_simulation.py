import argparse
import json
import multiprocessing as mp
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import psutil
from qiskit_aer import AerSimulator

import numpy as np
from troma import (
    CombinatorialProblem,
    ConstraintSketchMap,
    matching_pursuit,
    get_optimizer,
    bind_optimizer,
)
from easy_dnr import DNR_Network

#########################################################
#                        data                           #
#########################################################

benchmark_list = ["9_bus","12_bus","15_bus", "33_bus"]

ITERATION_NUMBER = 5
N_SAMPLES = 1000

MAX_NUMBER_OF_RUNS = 30 #per method, per network, chosen to keep total runtime reasonable (see estimate_simulation_times)

DIGITAL_ANNEALING_MAXITER = 1000

number_shots = {
    "9_bus": 4096,
    "12_bus": 4096,
    "15_bus": 4096,
    "33_bus": 512,
}

number_layers = {
    "9_bus": 4,
    "12_bus": 4,
    "15_bus": 4,
    "33_bus": 2,
}

simulator_mode = {
    "9_bus": "statevector",
    "12_bus": "statevector",
    "15_bus": "statevector",
    "33_bus": "matrix_product_state",
}

sim_types = {
    "9_bus" : {
        "state-of-art": [
            "genetic",
            "simulated_annealing_pp" #simulated annealing with custom transition function (preserves hamming weight)
        ],
        "classical": [
            "classical_NN_2", 
        ],
        "quantum": [
            "quantum_NN_2", 
            "quantum_NN_2_pp", 
            "quantum_NN_3", 
            "quantum_NN_3_pp", 
            "quantum_NN_4", 
            "quantum_NN_4_pp", 
            "quantum_all_2", 
            "quantum_all_2_pp", 
            "quantum_all_3", 
            "quantum_all_3_pp",
        ]
    },
    "12_bus": {
        "state-of-art": [
            "genetic",
            "simulated_annealing_pp"
        ],
        "classical": [
            "classical_NN_2", 
        ],
        "quantum": [
            "quantum_NN_2", 
            "quantum_NN_2_pp", 
            "quantum_NN_3", 
            "quantum_NN_3_pp", 
            "quantum_NN_4", 
            "quantum_NN_4_pp", 
            "quantum_all_2", 
            "quantum_all_2_pp", 
            "quantum_all_3", 
            "quantum_all_3_pp",
        ]
    },
    "15_bus": {
        "state-of-art": [
            "genetic",
            "simulated_annealing_pp"
        ],
        "classical": [
            "classical_NN_2"
        ],
        "quantum": [
            "quantum_NN_2",
            "quantum_NN_2_pp",
            "quantum_NN_3",
            "quantum_NN_3_pp",
            "quantum_NN_4",
            "quantum_NN_4_pp",
            "quantum_all_2",
            "quantum_all_2_pp",
            "quantum_all_3",
            "quantum_all_3_pp",
        ]
    },
    "33_bus": {
        "state-of-art": [
            "genetic",
            "simulated_annealing_pp"
        ],
        "classical": [
            "classical_NN_2", 
            "classical_NN_3", 
            "classical_NN_4", 
            "classical_all_2"
        ],
        "quantum": [
            "quantum_NN_2", 
            "quantum_NN_2_pp"
        ]
    }
}

interaction_size_per_method = {
    "classical_NN_2" : 2,
    "classical_NN_3" : 3,
    "classical_NN_4" : 4,
    "classical_all_2" : 2,
    "quantum_NN_2" : 2,
    "quantum_NN_2_pp" : 2,
    "quantum_NN_3" : 3,
    "quantum_NN_3_pp" : 3,
    "quantum_NN_4" : 4,
    "quantum_NN_4_pp" : 4,
    "quantum_all_2" : 2,
    "quantum_all_2_pp" : 2,
    "quantum_all_3" : 3,
    "quantum_all_3_pp" : 3,
}

constraint_types_per_method = {
    "classical_NN_2" : "nearest_neighbors",
    "classical_NN_3" : "nearest_neighbors",
    "classical_NN_4" : "nearest_neighbors",
    "classical_all_2" : "all_interactions",
    "quantum_NN_2" : "nearest_neighbors",
    "quantum_NN_2_pp" : "nearest_neighbors",
    "quantum_NN_3" : "nearest_neighbors",
    "quantum_NN_3_pp" : "nearest_neighbors",
    "quantum_NN_4" : "nearest_neighbors",
    "quantum_NN_4_pp" : "nearest_neighbors",
    "quantum_all_2" : "all_interactions",
    "quantum_all_2_pp" : "all_interactions",
    "quantum_all_3" : "all_interactions",
    "quantum_all_3_pp" : "all_interactions",
}

optimizer_per_method = {
    "classical_NN_2" : "spin_chain_nn_max",
    "classical_NN_3" : "spin_chain_nn_max",
    "classical_NN_4" : "spin_chain_nn_max",
    "classical_all_2" : "dual_annealing",
    "quantum_NN_2" : "aoa",
    "quantum_NN_2_pp" : "aoa",
    "quantum_NN_3" : "aoa",
    "quantum_NN_3_pp" : "aoa",
    "quantum_NN_4" : "aoa",
    "quantum_NN_4_pp" : "aoa",
    "quantum_all_2" : "aoa",
    "quantum_all_2_pp" : "aoa",
    "quantum_all_3" : "aoa",
    "quantum_all_3_pp" : "aoa",
}

is_post_process_method = {
    "classical_NN_2" : False,
    "classical_NN_3" : False,
    "classical_NN_4" : False,
    "classical_all_2" : False,
    "quantum_NN_2" : False,
    "quantum_NN_2_pp" : True,
    "quantum_NN_3" : False,
    "quantum_NN_3_pp" : True,
    "quantum_NN_4" : False,
    "quantum_NN_4_pp" : True,
    "quantum_all_2" : False,
    "quantum_all_2_pp" : True,
    "quantum_all_3" : False,
    "quantum_all_3_pp" : True,
}

sim_budget = 7200 #seconds, i.e. 2 hours

#Simulations time for one run (in seconds), loaded from sim_time.json
#(used to estimate number of runs we can do in sim_budget)
sim_time = json.loads((Path(__file__).resolve().parent / "sim_time.json").read_text())

#########################################################
#                 Benchmark History                     #
#########################################################

def _to_jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        return _to_jsonable(value.__dict__)
    return str(value)


def _extract_aoa_summary(optimizer_metadata):
    if not optimizer_metadata:
        return None

    valid_items = [item for item in optimizer_metadata if item is not None]
    if not valid_items:
        return None

    last_item = valid_items[-1]
    truth_evals_per_iter = [
        int(item.get("truth_objective_evaluations", 0)) for item in valid_items
    ]
    return {
        "number_layers": int(last_item.get("number_layers", 0)),
        "final_parameters": _to_jsonable(last_item.get("final_parameters")),
        "final_parameters": _to_jsonable(last_item.get("final_parameters")),
        "gammas": _to_jsonable(last_item.get("gammas")),
        "betas": _to_jsonable(last_item.get("betas")),
        "solver_steps_per_iteration": [
            int(item.get("solver_steps", 0)) for item in valid_items
        ],
        "circuit_depth_per_iteration": [
            int(item.get("circuit_depth", 0)) for item in valid_items
        ],
        "objective_evaluations_per_iteration": [
            int(item.get("objective_evaluations", 0)) for item in valid_items
        ],
        "truth_objective_evaluations_per_iteration": truth_evals_per_iter,
        "total_truth_objective_evaluations": sum(truth_evals_per_iter),
    }


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


RESULTS_FILE = None  # set at the start of main() with a run timestamp


def _append_run_record(record):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_to_jsonable(record), ensure_ascii=True) + "\n")


#########################################################
#                   Sanity Check                        #
#########################################################

def run_sanity_check(nb_cores, max_memory_mb, networks=None, categories=None):
    #Check every method works on a minimal 1-shot run before committing to the full benchmark.
    #Uses tiny parameters so this completes in seconds.

    networks = networks or benchmark_list
    categories = categories or {"state-of-art", "classical", "quantum"}

    _SC_NETWORK = networks[0]
    _SC_SAMPLES = 50
    _SC_ITERATIONS = 1
    _SC_SHOTS = 64
    _SC_LAYERS = 1

    print(f"Running sanity checks on {_SC_NETWORK} for categories: {', '.join(sorted(categories))}...")
    _sc_net = DNR_Network(_SC_NETWORK)
    _sc_n_ones = _sc_net.n_switches - _sc_net.n_dits

    if "classical" in categories:
        # --- classical: spin_chain_nn_max ---
        _sc_prob = CombinatorialProblem(_sc_net.evaluate_mcco,
                                        problem_size=_sc_net.n_dits,
                                        problem_dimension=_sc_net.n_switches)
        _sc_sketch_map = ConstraintSketchMap(sketch_length=_sc_net.n_dits,
                                             sketch_dimension=_sc_net.n_switches,
                                             interaction_size=2,
                                             constraints="nearest_neighbors")
        _sc_prob.sampling(n_samples=_SC_SAMPLES, threshold_parameter="Auto",
                          sampling_function=_sc_net.sample_radial_configuration,
                          seed=0, n_jobs=-1)
        _sc_result = matching_pursuit(_sc_prob.sketching(_sc_sketch_map),
                                       iteration_number=_SC_ITERATIONS,
                                       optimizer=get_optimizer("spin_chain_nn_max"))
        assert len(_sc_result.positions) > 0, "Sanity: spin_chain_nn_max returned no positions."
        assert all(np.isfinite(v) for v in _sc_result.values), "Sanity: spin_chain_nn_max returned non-finite values."
        _sc_net.analyse_results(_sc_result.positions, representation="dit",print_results=False)
        print("Spin chain NN sanity check passed.")

        # --- classical: dual_annealing ---
        _sc_prob.sampling(n_samples=_SC_SAMPLES, threshold_parameter="Auto",
                          sampling_function=_sc_net.sample_radial_configuration,
                          seed=1, n_jobs=-1)
        _sc_sketch_map_all = ConstraintSketchMap(sketch_length=_sc_net.n_dits,
                                                 sketch_dimension=_sc_net.n_switches,
                                                 interaction_size=2,
                                                 constraints="all_interactions")
        _sc_result_da = matching_pursuit(_sc_prob.sketching(_sc_sketch_map_all),
                                          iteration_number=_SC_ITERATIONS,
                                          optimizer=bind_optimizer("dual_annealing", maxiter=10, seed=1))
        assert len(_sc_result_da.positions) > 0, "Sanity: dual_annealing returned no positions."
        assert all(np.isfinite(v) for v in _sc_result_da.values), "Sanity: dual_annealing returned non-finite values."
        _sc_net.analyse_results(_sc_result_da.positions, representation="dit", print_results=False)
        print("Dual annealing sanity check passed.")

    if "quantum" in categories:
        # --- quantum: aoa ---
        _sc_q_prob = CombinatorialProblem(_sc_net.evaluate_mcco,
                                           problem_size=_sc_net.n_switches,
                                           problem_dimension=2)
        _sc_q_sketch_map = ConstraintSketchMap(sketch_length=_sc_net.n_switches,
                                                sketch_dimension=2,
                                                interaction_size=2,
                                                constraints="nearest_neighbors")
        _sc_q_prob.sampling(n_samples=_SC_SAMPLES, threshold_parameter="Auto",
                            sampling_function=_sc_net.sample_fix_ham_weight,
                            sampling_args={"number_of_one": _sc_n_ones},
                            seed=2, n_jobs=-1)
        _sc_backend = AerSimulator(method=simulator_mode[_SC_NETWORK], device="GPU",
                                    max_parallel_threads=nb_cores,
                                    max_parallel_experiments=0,
                                    max_memory_mb=max_memory_mb,
                                    seed_simulator=2)
        _sc_backend.set_max_qubits(_sc_net.n_switches)
        _sc_q_result = matching_pursuit(
            _sc_q_prob.sketching(_sc_q_sketch_map),
            iteration_number=_SC_ITERATIONS,
            optimizer=bind_optimizer("aoa", backend=_sc_backend,
                                     number_shots=_SC_SHOTS, number_layers=_SC_LAYERS,
                                     mixer="ring", hamming_weight=_sc_n_ones),
            return_optimizer_metadata=True,
        )
        assert len(_sc_q_result.positions) > 0, "Sanity: aoa returned no positions."
        assert all(np.isfinite(v) for v in _sc_q_result.values), "Sanity: aoa returned non-finite values."
        assert _sc_q_result.optimizer_metadata is not None, "Sanity: aoa returned no optimizer metadata."
        _sc_net.analyse_results(_sc_q_result.positions, representation="bit", print_results=False)
        print("AOA sanity check passed.")

    if "state-of-art" in categories:
        # --- state-of-art: genetic (jakus) ---
        _sc_genetic_result = _sc_net.solve_with_dnrlib("jakus")
        assert isinstance(_sc_genetic_result, list) and len(_sc_genetic_result) == _sc_net.n_switches, \
            "Sanity: genetic returned unexpected result shape."
        assert _sc_net.num_pf > 0, "Sanity: genetic made no power flow calls."
        print("Genetic sanity check passed.")

        # --- state-of-art: simulated_annealing_pp ---
        _sc_sa_result = _sc_net.solve_via_simulated_annealing(max_iter=20, seed=0)
        assert isinstance(_sc_sa_result, list) and len(_sc_sa_result) == _sc_net.n_switches, \
            "Sanity: simulated_annealing returned unexpected result shape."
        assert _sc_net.num_pf > 0, "Sanity: simulated_annealing made no power flow calls."
        print("Simulated annealing sanity check passed.")

        # --- state-of-art: simulated_annealing_dnr_pp ---
        _sc_sa_dnr_result = _sc_net.solve_via_simulated_annealing(max_iter=5, seed=1,
                                                                   local_search_fn=_sc_net.local_search)
        assert isinstance(_sc_sa_dnr_result, list) and len(_sc_sa_dnr_result) == _sc_net.n_switches, \
            "Sanity: simulated_annealing_dnr returned unexpected result shape."
        assert _sc_net.num_pf > 0, "Sanity: simulated_annealing_dnr made no power flow calls."
        print("Simulated annealing DNR sanity check passed.")

    print("Sanity checks passed.")


#########################################################
#             Estimate Simulation Times                 #
#########################################################

def estimate_simulation_times(networks=None, categories=None):
    #Print a table of estimated run counts and total times derived from sim_time.

    networks = networks or benchmark_list
    categories = categories or {"state-of-art", "classical", "quantum"}

    col = 66
    print(f"{'network':<10} {'sim_type':<22} {'time/run':>9} {'n_runs':>7} {'est. total':>12}")
    print("-" * col)

    grand_total_sec = 0

    for network in networks:
        for category in ("state-of-art", "classical", "quantum"):
            if category not in categories:
                continue
            for sim_type in sim_types[network][category]:
                t = sim_time[network][sim_type]
                n_runs = min(MAX_NUMBER_OF_RUNS, sim_budget // t)
                total_sec = t * n_runs
                grand_total_sec += total_sec
                print(f"{network:<10} {sim_type:<22} {t:>8}s {n_runs:>7} {total_sec/3600:>11.2f}h")

    print("-" * col)
    print(f"{'Grand total':<41} {grand_total_sec/3600:>11.2f}h")


#########################################################
#               Update Simulation Times                 #
#########################################################

def update_sim_times(nb_cores, max_memory_mb, networks=None, categories=None):
    #Run each method once with iteration_number=1 and full benchmark parameters.
    #Estimated time per run = measured_time * ITERATION_NUMBER, written to sim_time.json.

    networks = networks or benchmark_list
    categories = categories or {"state-of-art", "classical", "quantum"}

    print("Updating simulation times...")
    seed = 0

    sim_time_path = Path(__file__).resolve().parent / "sim_time.json"
    new_sim_time = json.loads(sim_time_path.read_text()) if sim_time_path.exists() else {}

    for network in networks:
        dnr_net = DNR_Network(network)
        new_sim_time.setdefault(network, {})

        #state-of-art
        for sim_type in (sim_types[network]["state-of-art"] if "state-of-art" in categories else []):
            print(f"Updating time for {network}/{sim_type}...")
            t0 = time.perf_counter()

            if sim_type == "genetic":
                dnr_net.solve_with_dnrlib('jakus')

            if sim_type == "simulated_annealing_pp":
                dnr_net.solve_via_simulated_annealing(max_iter=1000)

            if sim_type == "simulated_annealing_dnr_pp":
                dnr_net.solve_via_simulated_annealing(max_iter=1000, local_search_fn=dnr_net.local_search)

            elapsed = time.perf_counter() - t0
            estimated = max(1, round(elapsed))
            new_sim_time[network][sim_type] = estimated
            print(f"  {network}/{sim_type}: {elapsed:.1f}s per run -> {estimated}s")

        #classical
        for sim_type in (sim_types[network]["classical"] if "classical" in categories else []):
            print(f"Updating time for {network}/{sim_type}...")
            interaction_size = interaction_size_per_method[sim_type]
            constraints = constraint_types_per_method[sim_type]
            optimizer_name = optimizer_per_method[sim_type]
            is_pp = is_post_process_method[sim_type]

            problem = CombinatorialProblem(dnr_net.evaluate_mcco,
                                        problem_size=dnr_net.n_dits,
                                        problem_dimension=dnr_net.n_switches,
                                        feasibility_function=lambda x: int(np.sum(x)) == number_of_ones,
            )
            sketch_map = ConstraintSketchMap(sketch_length=dnr_net.n_dits,
                                            sketch_dimension=dnr_net.n_switches,
                                            interaction_size=interaction_size,
                                            constraints=constraints)
            problem.sampling(n_samples=N_SAMPLES, threshold_parameter="Auto",
                             sampling_function=dnr_net.sample_radial_configuration,
                             seed=seed, n_jobs=-1)
            if optimizer_name == "spin_chain_nn_max":
                optimizer = get_optimizer(optimizer_name)
            else:
                optimizer = bind_optimizer(optimizer_name, maxiter=DIGITAL_ANNEALING_MAXITER, seed=seed)

            t0 = time.perf_counter()
            matching_pursuit(problem.sketching(sketch_map),
                             iteration_number=1,
                             optimizer=optimizer,
                             post_processing="2_bit_swap" if is_pp else None)
            elapsed = time.perf_counter() - t0

            estimated = max(1, round(elapsed * ITERATION_NUMBER))
            new_sim_time[network][sim_type] = estimated
            print(f"  {network}/{sim_type}: {elapsed:.1f}s x {ITERATION_NUMBER} -> {estimated}s")

        #quantum
        number_shots_sim = number_shots[network]
        number_layers_sim = number_layers[network]
        number_of_ones = dnr_net.n_switches - dnr_net.n_dits
        mode = simulator_mode[network]

        for sim_type in (sim_types[network]["quantum"] if "quantum" in categories else []):
            print(f"Updating time for {network}/{sim_type}...")
            interaction_size = interaction_size_per_method[sim_type]
            constraints = constraint_types_per_method[sim_type]
            optimizer_name = optimizer_per_method[sim_type]
            is_pp = is_post_process_method[sim_type]

            problem = CombinatorialProblem(dnr_net.evaluate_mcco,
                                        problem_size=dnr_net.n_switches,
                                        problem_dimension=2,
                                        feasibility_function=lambda x: int(np.sum(x)) == number_of_ones,
            )
            sketch_map = ConstraintSketchMap(sketch_length=dnr_net.n_switches,
                                            sketch_dimension=2,
                                            interaction_size=interaction_size,
                                            constraints=constraints)
            backend = AerSimulator(method=mode,
                                   device="GPU",
                                   max_parallel_threads=nb_cores,
                                   max_parallel_experiments=0,
                                   matrix_product_state_truncation_threshold=1e-10,
                                   max_memory_mb=max_memory_mb,
                                   seed_simulator=seed)
            backend.set_max_qubits(dnr_net.n_switches)
            problem.sampling(n_samples=N_SAMPLES, threshold_parameter="Auto",
                             sampling_function=dnr_net.sample_fix_ham_weight,
                             sampling_args={"number_of_one": number_of_ones},
                             seed=seed, n_jobs=-1)
            opti = bind_optimizer(optimizer_name, backend=backend,
                                   number_shots=number_shots_sim,
                                   number_layers=number_layers_sim,
                                   mixer="ring", hamming_weight=number_of_ones)

            t0 = time.perf_counter()
            matching_pursuit(problem.sketching(sketch_map),
                             iteration_number=1,
                             optimizer=opti, return_optimizer_metadata=True,
                             post_processing="2_bit_swap" if is_pp else None)
            elapsed = time.perf_counter() - t0

            estimated = max(1, round(elapsed * ITERATION_NUMBER))
            new_sim_time[network][sim_type] = estimated
            print(f"  {network}/{sim_type}: {elapsed:.1f}s x {ITERATION_NUMBER} -> {estimated}s")

    sim_time_path.write_text(json.dumps(new_sim_time, indent=4))
    print("sim_time.json updated.")


#########################################################
#                Run the benchmark                     #
#########################################################

MASTER_SEED = 0


def _run_quantum_case(network, sim_type, run, n_runs, seed, nb_cores, max_memory_mb,
                       interaction_size, constraints, optimizer_name, is_pp,
                       number_shots_sim, number_layers_sim, number_of_ones, mode,
                       results_file):
    #Run a single quantum benchmark case in its own process.
    #Each run gets a fresh AerSimulator/CUDA context, and a hard crash here
    #(e.g. a GPU/CUDA segfault, which Python can't catch) only kills this
    #subprocess instead of the whole multi-hour benchmark.
    global RESULTS_FILE
    RESULTS_FILE = results_file

    run_start = time.perf_counter()

    dnr_net = DNR_Network(network)

    try:
        problem = CombinatorialProblem(dnr_net.evaluate_mcco,
                                    problem_size=dnr_net.n_switches,
                                    problem_dimension=2,
                                    feasibility_function=lambda x: int(np.sum(x)) == number_of_ones,
        )
        sketch_map = ConstraintSketchMap(sketch_length=dnr_net.n_switches,
                                            sketch_dimension=2,
                                        interaction_size=interaction_size,
                                        constraints=constraints
        )
        backend = AerSimulator(
            method=mode,
            device="GPU",
            max_parallel_threads=nb_cores,
            max_parallel_experiments=0,
            matrix_product_state_truncation_threshold=1e-10,
            max_memory_mb=max_memory_mb,
            seed_simulator=seed,
        )
        backend.set_max_qubits(dnr_net.n_switches)

        problem.sampling(
            n_samples=N_SAMPLES,
            threshold_parameter="Auto",
            sampling_function=dnr_net.sample_fix_ham_weight,
            sampling_args={"number_of_one": number_of_ones},
            seed=seed,
            n_jobs=-1,
        )

        problem_sketch = problem.sketching(sketch_map)

        opti = bind_optimizer(optimizer_name,
                              backend=backend,
                              number_shots=number_shots_sim,
                              number_layers=number_layers_sim,
                              mixer="ring",
                              hamming_weight=number_of_ones
        )
        result = matching_pursuit(problem_sketch,
                                iteration_number=ITERATION_NUMBER,
                                optimizer=opti,
                                return_optimizer_metadata=True,
                                post_processing="2_bit_swap" if is_pp else None,
        )

        details = dnr_net.analyse_results(result.positions, representation="bit", print_results=False)

        run_record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "network": network,
            "category": "quantum",
            "sim_type": sim_type,
            "run_index": int(run),
            "n_runs_budgeted": int(n_runs),
            "seed": seed,
            "duration_sec": float(time.perf_counter() - run_start),
            "problem": {
                "problem_size": int(dnr_net.n_switches),
                "problem_dimension": 2,
                "n_samples": int(N_SAMPLES),
                "iteration_number": int(ITERATION_NUMBER),
                "number_of_ones": int(number_of_ones),
            },
            "method": {
                "interaction_size": int(interaction_size),
                "constraints": constraints,
                "optimizer_name": optimizer_name,
                "post_processing": "2_bit_swap" if is_pp else None,
                "number_shots": int(number_shots_sim),
                "number_layers": int(number_layers_sim),
                "mixer": "ring",
                "backend": {
                    "method": mode,
                    "device": "GPU",
                    "seed_simulator": seed,
                    "max_parallel_threads": int(nb_cores),
                    "max_memory_mb": max_memory_mb,
                },
            },
            "matching_pursuit": {
                "positions": [int(x) for x in result.positions],
                "values": [float(x) for x in result.values],
                "n_lines": int(result.n_lines),
            },
            "optimizer_metadata": _to_jsonable(result.optimizer_metadata),
            "aoa": _extract_aoa_summary(result.optimizer_metadata),
            "details": _to_jsonable(details),
        }
        _append_run_record(run_record)

    except Exception as e:
        print(f"ERROR [{network}/{sim_type} run {run} seed {seed}]: {e}")
        _append_run_record({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "network": network,
            "category": "quantum",
            "sim_type": sim_type,
            "run_index": int(run),
            "seed": seed,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        })


def main():
    parser = argparse.ArgumentParser(description="DNR full benchmark.")
    parser.add_argument("--sanity-only", action="store_true",
                        help="Run sanity check only and exit.")
    parser.add_argument("--estimate", action="store_true",
                        help="Print estimated run counts and total times from sim_time and exit.")
    parser.add_argument("--sim-time-update", action="store_true",
                        help="Benchmark each method (iteration_number=1, x5) and update sim_time.json.")
    parser.add_argument("--networks", nargs="+", choices=benchmark_list, default=None,
                        metavar="NETWORK",
                        help="Networks to run. Choices: %(choices)s. Default: all.")
    parser.add_argument("--methods", nargs="+", choices=["state-of-art", "classical", "quantum"],
                        default=None, metavar="CATEGORY",
                        help="Method categories to run. Choices: %(choices)s. Default: all.")
    parser.add_argument("--run-from-seeds", type=str, default=None, metavar="SEEDS_FILE",
                        help="Path to a JSONL file of seeds. Each line has {network, category, sim_type, n_runs, seeds}. "
                             "Replays exactly those runs with the given seeds instead of drawing new ones.")
    args = parser.parse_args()

    active_networks = args.networks or benchmark_list
    active_categories = set(args.methods or ["state-of-art", "classical", "quantum"])

    # Load seed schedule if provided
    seed_schedule = {}
    if args.run_from_seeds:
        seed_path = Path(args.run_from_seeds)
        assert seed_path.exists(), f"Seed file not found: {seed_path}"
        with seed_path.open() as f:
            for line in f:
                entry = json.loads(line)
                key = (entry["network"], entry["sim_type"])
                seed_schedule[key] = entry["seeds"]

    if args.estimate:
        estimate_simulation_times(networks=active_networks, categories=active_categories)
        return

    #########################################################
    #                Check Installation                     #
    #########################################################

    #check GPU works (only needed for quantum)
    if "quantum" in active_categories:
        import subprocess, shutil
        if shutil.which("nvidia-smi"):
            result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
            assert result.returncode == 0, "GPU is not available."
        sim = AerSimulator()
        assert 'GPU' in sim.available_devices(), "Qiskit Aer does not have GPU support available."

    #check memory amount
    total_gb = psutil.virtual_memory().total / (1024**3)
    assert total_gb >= 15, f"Not enough memory: {total_gb:.2f} GB available, but at least 15 GB is required."

    max_memory_mb = int(total_gb * 1024 * 0.9) #use at most 90% of available memory
    nb_cores = psutil.cpu_count(logical=True) #number of logical cores

    run_sanity_check(nb_cores, max_memory_mb, networks=active_networks, categories=active_categories)

    if args.sanity_only:
        return

    if args.sim_time_update:
        update_sim_times(nb_cores, max_memory_mb, networks=active_networks, categories=active_categories)
        return

    global RESULTS_FILE
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    RESULTS_FILE = RESULTS_DIR / f"benchmark_runs_{run_ts}.jsonl"

    rng = np.random.default_rng(seed=MASTER_SEED)

    for network in active_networks:

        dnr_net = DNR_Network(network)

        #state-of-art
        for sim_type in (sim_types[network]["state-of-art"] if "state-of-art" in active_categories else []):

            #decide number of runs and seeds
            sched_key = (network, sim_type)
            if sched_key in seed_schedule:
                run_seeds = seed_schedule[sched_key]
                n_runs = len(run_seeds)
            else:
                n_runs = min(MAX_NUMBER_OF_RUNS, sim_budget // sim_time[network][sim_type])
                run_seeds = [int(rng.integers(0, 1_000_000)) for _ in range(n_runs)]
            print(f"Running {n_runs} runs for {sim_type} on {network} benchmark.")

            for run, seed in enumerate(run_seeds):
                run_start = time.perf_counter()
                try:
                    if sim_type == "genetic":
                        # jakus does not expose a seed parameter — runs are not reproducible
                        solver_output = dnr_net.solve_with_dnrlib("jakus")
                        objective_function_calls = int(dnr_net.num_pf)
                        method_params = {
                            "solver": "jakus",
                        }

                    elif sim_type == "simulated_annealing_pp":
                        solver_output = dnr_net.solve_via_simulated_annealing(max_iter=1000, seed=seed)
                        objective_function_calls = int(dnr_net.num_pf)
                        method_params = {
                            "max_iter": 1000,
                            "post_processing": "2_bit_swap",
                        }

                    elif sim_type == "simulated_annealing_dnr_pp":
                        solver_output = dnr_net.solve_via_simulated_annealing(
                            max_iter=1000,
                            local_search_fn=dnr_net.local_search,
                            seed=seed
                        )
                        objective_function_calls = int(dnr_net.num_pf)
                        method_params = {
                            "max_iter": 1000,
                            "post_processing": "dnr_local_search",
                        }

                    else:
                        raise ValueError(f"Unknown state-of-art sim_type: {sim_type}")

                    details = {
                        "switch_vector": [int(x) for x in solver_output],
                        "dit_config": list(dnr_net.switch_vector_to_dit_representation(solver_output)),
                        "objective": float(dnr_net.evaluate(solver_output)),
                        "is_radial": bool(dnr_net.check_radiality(solver_output)),
                        "is_connected": bool(dnr_net.check_connectivity(solver_output)),
                    }

                    _append_run_record({
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "network": network,
                        "category": "state-of-art",
                        "sim_type": sim_type,
                        "run_index": int(run),
                        "n_runs_budgeted": int(n_runs),
                        "seed": seed,
                        "duration_sec": float(time.perf_counter() - run_start),
                        "method": method_params,
                        "objective_function_calls": objective_function_calls,
                        "details": details,
                    })

                except Exception as e:
                    print(f"ERROR [{network}/{sim_type} run {run} seed {seed}]: {e}")
                    _append_run_record({
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "network": network,
                        "category": "state-of-art",
                        "sim_type": sim_type,
                        "run_index": int(run),
                        "n_runs_budgeted": int(n_runs),
                        "seed": seed,
                        "status": "error",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    })


        #classical
        for sim_type in (sim_types[network]["classical"] if "classical" in active_categories else []):

            #Resolve simulation method parameters
            interaction_size = interaction_size_per_method[sim_type]
            constraints = constraint_types_per_method[sim_type]
            optimizer_name = optimizer_per_method[sim_type]
            is_pp = is_post_process_method[sim_type]

            #decide number of runs and seeds
            sched_key = (network, sim_type)
            if sched_key in seed_schedule:
                run_seeds = seed_schedule[sched_key]
                n_runs = len(run_seeds)
            else:
                n_runs = min(MAX_NUMBER_OF_RUNS, sim_budget // sim_time[network][sim_type])
                run_seeds = [int(rng.integers(0, 1_000_000)) for _ in range(n_runs)]
            print(f"Running {n_runs} runs for {sim_type} on {network} benchmark.")

            problem = CombinatorialProblem(dnr_net.evaluate_mcco,
                                        problem_size=dnr_net.n_dits,
                                        problem_dimension=dnr_net.n_switches,
                                        feasibility_function=lambda x: int(np.sum(x)) == number_of_ones,
            )
            sketch_map = ConstraintSketchMap(sketch_length=dnr_net.n_dits,
                                            sketch_dimension=dnr_net.n_switches,
                                            interaction_size=interaction_size,
                                            constraints=constraints
            )

            for run, seed in enumerate(run_seeds):
                run_start = time.perf_counter()

                try:
                    #Start the simulation run
                    #----------------------------------------------------

                    problem.sampling(
                        n_samples=N_SAMPLES,
                        threshold_parameter="Auto",
                        sampling_function=dnr_net.sample_radial_configuration,
                        seed=seed,
                        n_jobs=-1,
                    )

                    problem_sketch = problem.sketching(sketch_map)

                    if optimizer_name == "spin_chain_nn_max":
                        optimizer = get_optimizer(optimizer_name)
                    else:
                        optimizer = bind_optimizer(optimizer_name, maxiter=DIGITAL_ANNEALING_MAXITER, seed=seed)

                    result = matching_pursuit(problem_sketch,
                                            iteration_number=ITERATION_NUMBER,
                                            optimizer=optimizer,
                                            post_processing="2_bit_swap" if is_pp else None
                    )

                    details = dnr_net.analyse_results(result.positions, representation="dit",print_results=False)

                    run_record = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "network": network,
                        "category": "classical",
                        "sim_type": sim_type,
                        "run_index": int(run),
                        "n_runs_budgeted": int(n_runs),
                        "seed": seed,
                        "duration_sec": float(time.perf_counter() - run_start),
                        "problem": {
                            "problem_size": int(dnr_net.n_dits),
                            "problem_dimension": int(dnr_net.n_switches),
                            "n_samples": int(N_SAMPLES),
                            "iteration_number": int(ITERATION_NUMBER),
                        },
                        "method": {
                            "interaction_size": int(interaction_size),
                            "constraints": constraints,
                            "optimizer_name": optimizer_name,
                            "post_processing": "2_bit_swap" if is_pp else None,
                        },
                        "matching_pursuit": {
                            "positions": [int(x) for x in result.positions],
                            "values": [float(x) for x in result.values],
                            "n_lines": int(result.n_lines),
                        },
                        "details": _to_jsonable(details),
                    }
                    _append_run_record(run_record)

                    #----------------------------------------------------
                    #End of simulation run

                except Exception as e:
                    print(f"ERROR [{network}/{sim_type} run {run} seed {seed}]: {e}")
                    _append_run_record({
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "network": network,
                        "category": "classical",
                        "sim_type": sim_type,
                        "run_index": int(run),
                        "seed": seed,
                        "status": "error",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    })

        #quantum
        number_shots_sim = number_shots[network]
        number_layers_sim = number_layers[network]
        number_of_ones = dnr_net.n_switches - dnr_net.n_dits #number of closed switches
        mode = simulator_mode[network]
        for sim_type in (sim_types[network]["quantum"] if "quantum" in active_categories else []):

            #Resolve simulation method parameters
            interaction_size = interaction_size_per_method[sim_type]
            constraints = constraint_types_per_method[sim_type]
            optimizer_name = optimizer_per_method[sim_type]
            is_pp = is_post_process_method[sim_type]

            #decide number of runs and seeds
            sched_key = (network, sim_type)
            if sched_key in seed_schedule:
                run_seeds = seed_schedule[sched_key]
                n_runs = len(run_seeds)
            else:
                n_runs = min(MAX_NUMBER_OF_RUNS, sim_budget // sim_time[network][sim_type])
                run_seeds = [int(rng.integers(0, 1_000_000)) for _ in range(n_runs)]
            print(f"Running {n_runs} runs for {sim_type} on {network} benchmark.")

            for run, seed in enumerate(run_seeds):

                #Run this case in its own process: a fresh AerSimulator/CUDA context per
                #run, isolated so a crash here can't take down the whole benchmark.
                ctx = mp.get_context("spawn")
                proc = ctx.Process(
                    target=_run_quantum_case,
                    kwargs=dict(
                        network=network, sim_type=sim_type, run=run, n_runs=n_runs, seed=seed,
                        nb_cores=nb_cores, max_memory_mb=max_memory_mb,
                        interaction_size=interaction_size, constraints=constraints,
                        optimizer_name=optimizer_name, is_pp=is_pp,
                        number_shots_sim=number_shots_sim, number_layers_sim=number_layers_sim,
                        number_of_ones=number_of_ones, mode=mode,
                        results_file=RESULTS_FILE,
                    ),
                )
                proc.start()
                proc.join()

                if proc.exitcode != 0:
                    print(f"CRASHED [{network}/{sim_type} run {run} seed {seed}]: subprocess exited with code {proc.exitcode}")
                    _append_run_record({
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "network": network,
                        "category": "quantum",
                        "sim_type": sim_type,
                        "run_index": int(run),
                        "n_runs_budgeted": int(n_runs),
                        "seed": seed,
                        "status": "crashed",
                        "exitcode": int(proc.exitcode),
                    })

    print("Benchmark completed.")


if __name__ == "__main__":
    main()