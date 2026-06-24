import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone

from pathlib import Path
import psutil
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

import numpy as np

IBM_QUANTUM_TOKEN = os.environ.get("IBM_QUANTUM_TOKEN")
from troma import (
    CombinatorialProblem,
    ConstraintSketchMap,
    matching_pursuit,
    bind_optimizer,
)
from easy_dnr import DNR_Network

#########################################################
#                        data                           #
#########################################################

benchmark_list = [
    "9_bus", 
    "12_bus", 
    "15_bus"
    ]

USE_GPU = True       # use GPU device in Aer simulator (sanity check)
USE_PARALLEL = True  # use all CPU cores for sampling and simulation
FAKE = False          # use FakeFezV2 noisy simulator instead of real QPU (no credentials needed)

QPU_BACKEND_NAME = "ibm_fez"

N_SAMPLES = 1000

NUMBER_SHOTS = 4096 if not FAKE else 1024  # noisy fake sim doesn't need as many shots
POST_PROCESS = "2_bit_swap" 
QPU_MAX_ITER = 10
GRID_POINTS = 20
PRETRAIN_MAX_ITER = 60 #for 15_bus only


ITERATION_NUMBER = 1

number_layers = {
    "9_bus": 1,
    "12_bus": 1,
    "15_bus": 1
}

seeds_per_network = {
    "9_bus" : 123,
    "12_bus" : 456,
    "15_bus" : 840311
}

sim_types = {
    "9_bus": [
        "NN_2_aoa_native",
        "NN_2_qaoa_native",
    ],
    "12_bus": [
        "NN_2_aoa_native",
        "NN_2_qaoa_native",
    ],
    "15_bus": [
        "NN_2_aoa_native",
        "NN_2_qaoa_native",
    ]
}

optimizer_per_sim_type = {
    "NN_2_aoa_native" : "aoa_native",
    "NN_2_qaoa_native" : "qaoa_native",
}

interaction_size_per_method = {
    "NN_2_aoa_native" : 2,
    "NN_2_qaoa_native" : 2,
}

constraint_types_per_method = {
    "NN_2_aoa_native" : "nearest_neighbors",
    "NN_2_qaoa_native" : "nearest_neighbors",
}

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
    transpiled_depth_per_iter = [
        int(item.get("transpiled_circuit_depth", item.get("circuit_depth", 0)))
        for item in valid_items
    ]
    transpiled_gate_count_per_iter = [
        (
            None
            if item.get("transpiled_gate_count") is None
            else int(item.get("transpiled_gate_count"))
        )
        for item in valid_items
    ]
    job_ids_per_iter = [
        (None if item.get("job_id") is None else str(item.get("job_id")))
        for item in valid_items
    ]
    last_job_id = next((jid for jid in reversed(job_ids_per_iter) if jid), None)
    return {
        "number_layers": int(last_item.get("number_layers", 0)),
        "final_parameters": _to_jsonable(last_item.get("final_parameters")),
        "gammas": _to_jsonable(last_item.get("gammas")),
        "betas": _to_jsonable(last_item.get("betas")),
        "final_sample_distribution": _to_jsonable(last_item.get("final_sample_distribution")),
        "solver_steps_per_iteration": [
            int(item.get("solver_steps", 0)) for item in valid_items
        ],
        "circuit_depth_per_iteration": [
            int(item.get("circuit_depth", 0)) for item in valid_items
        ],
        "transpiled_circuit_depth_per_iteration": transpiled_depth_per_iter,
        "transpiled_gate_count_per_iteration": transpiled_gate_count_per_iter,
        "job_ids_per_iteration": job_ids_per_iter,
        "last_job_id": last_job_id,
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

def run_sanity_check(nb_cores, max_memory_mb, networks=None):
    #Check every method works on a minimal 1-shot run before committing to the full benchmark.
    #Uses tiny parameters so this completes in seconds.

    networks = networks or benchmark_list

    _SC_NETWORK = networks[0]
    _SC_SAMPLES = 50
    _SC_ITERATIONS = 1
    _SC_SHOTS = 64
    _SC_LAYERS = 1

    print(f"Running sanity checks on {_SC_NETWORK}...")
    _sc_net = DNR_Network(_SC_NETWORK)
    _sc_n_ones = _sc_net.n_switches - _sc_net.n_dits

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
                        seed=2, n_jobs=-1 if USE_PARALLEL else 1)
    _sc_backend = AerSimulator(method="statevector",
                                device="GPU" if USE_GPU else "CPU",
                                max_parallel_threads=nb_cores if USE_PARALLEL else 1,
                                max_parallel_experiments=0 if USE_PARALLEL else 1,
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

    print("Sanity checks passed.")


#########################################################
#                Run the benchmark                     #
#########################################################

MASTER_SEED = 0


def main():
    parser = argparse.ArgumentParser(description="DNR full benchmark.")
    parser.add_argument("--sanity-only", action="store_true",
                        help="Run sanity check only and exit.")
    args = parser.parse_args()

    #########################################################
    #                Check Installation                     #
    #########################################################

    #check GPU works

    import subprocess, shutil
    if USE_GPU:
        if shutil.which("nvidia-smi"):
            result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
            assert result.returncode == 0, "GPU is not available."
        sim = AerSimulator()
        assert 'GPU' in sim.available_devices(), "Qiskit Aer does not have GPU support available."

    #check memory amount
    total_gb = psutil.virtual_memory().total / (1024**3)
    assert total_gb >= 15, f"Not enough memory: {total_gb:.2f} GB available, but at least 15 GB is required."

    max_memory_mb = int(total_gb * 1024 * 0.95) #use at most 95% of available memory
    nb_cores = psutil.cpu_count(logical=True) #number of logical cores

    print(f"System has {total_gb:.2f} GB RAM and {nb_cores} CPU cores. Using up to {max_memory_mb/1024:.2f} GB RAM for simulations.\n")

    if not FAKE:
        run_sanity_check(nb_cores, max_memory_mb, networks=benchmark_list)
    else:
        print("[FAKE MODE] Skipping sanity checks.\n")

    if args.sanity_only:
        return

    global RESULTS_FILE
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if not FAKE:
        RESULTS_FILE = RESULTS_DIR / f"qpu_runs_{run_ts}.jsonl"
    else:
        RESULTS_FILE = RESULTS_DIR / f"qpu_runs_{run_ts}_FAKE.jsonl"

    rng = np.random.default_rng(seed=MASTER_SEED)

    PRETRAIN_OPTS = {
        "num_grid_points": GRID_POINTS,         # 20² = 400 grid points for p=1 scan
        "number_shots": 512,                    # shots per COBYLA evaluation
        "sim_method": "matrix_product_state",
        "mps_truncation_threshold": 1e-6,       # truncate small singular values
        "mps_max_bond_dimension": 64,           # cap bond dim — enough for warm-starting
        "progressive_depth_refinement": True,   # refine at p=1 first, then INTERP to p
        "max_sim_iter_p1": 40,                  # p=1 budget: cheap circuit, spend more
        "max_sim_iter": 0,                      # skip full-depth sim — p=1 refined + INTERP is enough
        "force_simulator": True,
        "device": "GPU" if USE_GPU else "CPU",
        "num_threads": nb_cores if USE_PARALLEL else 1,
        "max_sim_iter": PRETRAIN_MAX_ITER,
    }

    if FAKE:
        try:
            from qiskit_ibm_runtime.fake_provider import FakeFezV2
            fake_hw = FakeFezV2()
        except ImportError:
            from qiskit_ibm_runtime.fake_provider import FakeFez
            fake_hw = FakeFez()
        from qiskit_aer.noise import NoiseModel
        noise_model = NoiseModel.from_backend(fake_hw)
        # MPS is much faster than statevector/density_matrix for QAOA with ring mixer:
        # the ring topology creates 1D-like entanglement that MPS handles efficiently.
        backend = AerSimulator(
            method="statevector",
            noise_model=noise_model,
            device="GPU" if USE_GPU else "CPU",
            max_parallel_threads=nb_cores if USE_PARALLEL else 1,
            max_parallel_experiments=0 if USE_PARALLEL else 1,
            max_memory_mb=max_memory_mb,
            # matrix_product_state_max_bond_dimension=64,
            # matrix_product_state_truncation_threshold=1e-6,
        )
        print(f"[FAKE MODE] Using noisy Aer MPS simulator with {QPU_BACKEND_NAME} noise model\n")
    else:
        if not IBM_QUANTUM_TOKEN:
            raise RuntimeError(
                "IBM_QUANTUM_TOKEN environment variable is not set.\n"
                "Get your token from https://quantum.ibm.com/ and run:\n"
                "  export IBM_QUANTUM_TOKEN='your-token-here'"
            )
        QiskitRuntimeService.delete_account()
        QiskitRuntimeService.save_account(
            channel="ibm_quantum_platform",
            token=IBM_QUANTUM_TOKEN,
            overwrite=True,
        )

        service = QiskitRuntimeService()
        backend = service.backend(QPU_BACKEND_NAME)

    total_runs = sum(len(sim_types[n]) for n in benchmark_list)
    run_idx = 0
    print(f"\nBenchmark start — {len(benchmark_list)} networks, {total_runs} runs total.")
    print(f"Results → {RESULTS_FILE}\n")

    for network in benchmark_list:

        dnr_net = DNR_Network(network)

        #resolve method parameters that depend on the network
        number_layers_sim = number_layers[network]
        number_of_ones = dnr_net.n_switches - dnr_net.n_dits #number of closed switches

        print(f"{'='*60}")
        print(f"Network: {network}  |  {dnr_net.n_switches} switches  |  {len(sim_types[network])} methods")

        problem = CombinatorialProblem(dnr_net.evaluate_mcco,
                                    problem_size=dnr_net.n_switches,
                                    problem_dimension=2,
                                    feasibility_function=lambda x: int(np.sum(x)) == number_of_ones,
        )

        seed = seeds_per_network[network]
        print(f"  Sampling {N_SAMPLES} points (seed={seed})...")
        problem.sampling(
            n_samples=N_SAMPLES,
            threshold_parameter="Auto",
            sampling_function=dnr_net.sample_fix_ham_weight,
            sampling_args={"number_of_one": number_of_ones},
            seed=seed,
            n_jobs=-1 if USE_PARALLEL else 1,
        )
        print(f"  Sampling done.\n")

        for sim_type in sim_types[network]:

            #Resolve simulation method parameters
            interaction_size = interaction_size_per_method[sim_type]
            constraints = constraint_types_per_method[sim_type]
            optimizer = optimizer_per_sim_type[sim_type]

            run_idx += 1
            print(f"  [{run_idx}/{total_runs}] {sim_type}  "
                  f"layers={number_layers_sim}  interaction_size={interaction_size}  "
                  f"iterations={ITERATION_NUMBER}")

            sketch_map = ConstraintSketchMap(sketch_length=dnr_net.n_switches,
                                                sketch_dimension=2,
                                            interaction_size=interaction_size,
                                            constraints=constraints
            )

            try:

                run_start = time.perf_counter()
                print(f"  Started at {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")

                #Start the simulation run
                #----------------------------------------------------

                problem_sketch = problem.sketching(sketch_map)

                if optimizer == "aoa_native" or optimizer == "aoa":
                    opti = bind_optimizer(optimizer,
                                            backend=backend,
                                            number_shots=NUMBER_SHOTS,
                                            number_layers=number_layers_sim,
                                            mixer="ring",
                                            hamming_weight=number_of_ones,
                                            initial_state = "dicke",
                                            optimizer_options={"maxiter": QPU_MAX_ITER},
                                            pretrain=True,
                                            pretrain_options=PRETRAIN_OPTS,
                                            sampler_options=None if FAKE else {
                                                "dynamical_decoupling": {"enable": True, "sequence_type": "XpXm"},
                                                "twirling": {"enable_gates": True, "num_randomizations": 300},
                                            },
                    )
                else:
                    opti = bind_optimizer(optimizer,
                                            backend=backend,
                                            number_shots=NUMBER_SHOTS,
                                            number_layers=number_layers_sim,
                                            mixer="ring",
                                            hamming_weight=number_of_ones,
                                            optimizer_options={"maxiter": QPU_MAX_ITER},
                                            pretrain=True,
                                            pretrain_options=PRETRAIN_OPTS,
                                            sampler_options=None if FAKE else {
                                                "dynamical_decoupling": {"enable": True, "sequence_type": "XpXm"},
                                                "twirling": {"enable_gates": True, "num_randomizations": 300},
                                            },
                    )
                result = matching_pursuit(problem_sketch,
                                        iteration_number=ITERATION_NUMBER,
                                        optimizer=opti,
                                        return_optimizer_metadata=True,
                                        post_processing=POST_PROCESS,
                                        verbose=True,
                )
                #------------------------------------------------
                #End of simulation run

                elapsed = time.perf_counter() - run_start
                print(f"  Done in {elapsed:.1f}s  |  best value: {min(result.values):.4f}  |  {len(result.positions)} positions")

                details = dnr_net.analyse_results(result.positions, representation="bit", print_results=False)
                aoa_summary = _extract_aoa_summary(result.optimizer_metadata)

                run_record = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "network": network,
                    "category": "quantum",
                    "sim_type": sim_type,
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
                        "optimizer_name": optimizer,
                        "post_processing": POST_PROCESS,
                        "number_shots": int(NUMBER_SHOTS),
                        "number_layers": int(number_layers_sim),
                        "mixer": "ring",
                        "backend": {
                            "name": backend.name,
                        },
                    },
                    "matching_pursuit": {
                        "positions": [int(x) for x in result.positions],
                        "values": [float(x) for x in result.values],
                        "n_lines": int(result.n_lines),
                    },
                    "optimizer_metadata": _to_jsonable(result.optimizer_metadata),
                    "aoa": aoa_summary,
                    "compiled_circuit": {
                        "transpiled_circuit_depth_per_iteration": None if aoa_summary is None else aoa_summary.get("transpiled_circuit_depth_per_iteration"),
                        "transpiled_gate_count_per_iteration": None if aoa_summary is None else aoa_summary.get("transpiled_gate_count_per_iteration"),
                        "job_ids_per_iteration": None if aoa_summary is None else aoa_summary.get("job_ids_per_iteration"),
                        "last_job_id": None if aoa_summary is None else aoa_summary.get("last_job_id"),
                    },
                    "details": _to_jsonable(details),
                }
                _append_run_record(run_record)
                print(f"  Saved.\n")

                

            except Exception as e:
                print(f"ERROR [{network}/{sim_type} seed {seed}]: {e}")
                _append_run_record({
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "network": network,
                    "category": "quantum",
                    "sim_type": sim_type,
                    "seed": seed,
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
            
            #return # TEMP - run only the first method of the first network for quick testing
                    
    print("Benchmark completed.")


if __name__ == "__main__":
    main()
