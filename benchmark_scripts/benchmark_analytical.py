import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from easy_dnr import DNR_Network

#########################################################
#                        data                           #
#########################################################


sim_types = {
    '9_bus': ['baran', 'merlin','taylor'],
    '12_bus': ['baran', 'merlin','taylor'],
    '15_bus': ['baran', 'merlin','taylor'],
    '33_bus': ['baran', 'merlin','taylor'],
}

benchmark_list = ['9_bus', '12_bus','15_bus', '33_bus']

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
RESULTS_FILE = None


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
    if hasattr(value, 'to_dict') and callable(value.to_dict):
        return _to_jsonable(value.to_dict())
    if hasattr(value, '__dict__'):
        return _to_jsonable(value.__dict__)
    return str(value)


def _append_run_record(record):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open('a', encoding='utf-8') as f:
        f.write(json.dumps(_to_jsonable(record), ensure_ascii=True) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Analytical DNR benchmark (deterministic methods).')
    parser.add_argument('--networks', nargs='+', choices=benchmark_list, default=None,
                        metavar='NETWORK',
                        help='Networks to run. Choices: %(choices)s. Default: all.')
    parser.add_argument('--methods', nargs='+', choices=['baran', 'merlin', 'taylor'], default=None,
                        metavar='METHOD',
                        help='Analytical methods to run. Choices: %(choices)s. Default: all per network.')
    args = parser.parse_args()

    active_networks = args.networks or benchmark_list

    global RESULTS_FILE
    run_ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    RESULTS_FILE = RESULTS_DIR / f'analytical_runs_{run_ts}.jsonl'

    for network in active_networks:
        dnr_net = DNR_Network(network)
        active_methods = args.methods or sim_types[network]

        for method in active_methods:
            if method not in sim_types[network]:
                continue

            print(f'Running {method} on {network}...')
            run_start = time.perf_counter()

            try:
                solver_output = dnr_net.solve_with_dnrlib(method=method)
                objective_function_calls = int(dnr_net.num_pf)

                details = {
                    'switch_vector': [int(x) for x in solver_output],
                    'dit_config': list(dnr_net.switch_vector_to_dit_representation(solver_output)),
                    'objective': float(dnr_net.evaluate(solver_output)),
                    'is_radial': bool(dnr_net.check_radiality(solver_output)),
                    'is_connected': bool(dnr_net.check_connectivity(solver_output)),
                }

                _append_run_record({
                    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                    'network': network,
                    'category': 'analytical',
                    'sim_type': method,
                    'run_index': 0,
                    'n_runs_budgeted': 1,
                    'seed': None,
                    'duration_sec': float(time.perf_counter() - run_start),
                    'method': {
                        'solver': method,
                        'deterministic': True,
                    },
                    'objective_function_calls': objective_function_calls,
                    'details': details,
                })

            except Exception as e:
                print(f'ERROR [{network}/{method}]: {e}')
                _append_run_record({
                    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                    'network': network,
                    'category': 'analytical',
                    'sim_type': method,
                    'run_index': 0,
                    'n_runs_budgeted': 1,
                    'seed': None,
                    'status': 'error',
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                })

    print('Analytical benchmark completed.')


if __name__ == '__main__':
    main()
        
