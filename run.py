import os
import json
import argparse
from typing import Any, Dict

from tot.tasks import get_task
from tot.methods.bfs import solve, naive_solve
from tot.models import configure_llm_backend, gpt_usage, reset_gpt_usage


TASK_DEFAULTS = {
    'game24': {
        'task_start_index': 900,
        'task_end_index': 1000,
        'naive_run': False,
        'prompt_sample': None,
        'method_generate': 'propose',
        'method_evaluate': 'value',
        'method_select': 'greedy',
        'n_generate_sample': 1,
        'n_evaluate_sample': 3,
        'n_select_sample': 5,
        'temperature': 0.7,
        'evaluator_temperature': 0.7,
    },
    'text': {
        'task_start_index': 0,
        'task_end_index': 100,
        'naive_run': False,
        'prompt_sample': 'cot',
        'method_generate': 'sample',
        'method_evaluate': 'vote',
        'method_select': 'greedy',
        'n_generate_sample': 5,
        'n_evaluate_sample': 5,
        'n_select_sample': 1,
        'temperature': 1.0,
        'evaluator_temperature': 0.7,
    },
    'crosswords': {
        'task_start_index': 0,
        'task_end_index': 20,
        'naive_run': True,
        'prompt_sample': 'cot',
        'method_generate': 'sample',
        'method_evaluate': 'vote',
        'method_select': 'greedy',
        'n_generate_sample': 10,
        'n_evaluate_sample': 1,
        'n_select_sample': 1,
        'temperature': 0.7,
        'evaluator_temperature': 0.7,
    },
}


def safe_name(value):
    return ''.join(char if char.isalnum() or char in '._-' else '_' for char in value)


def apply_task_defaults(args, task_length):
    defaults = TASK_DEFAULTS.get(args.task, {})
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if args.task_start_index is None:
        args.task_start_index = defaults.get('task_start_index', 0)
    if args.task_end_index is None:
        args.task_end_index = min(defaults.get('task_end_index', task_length), task_length)

    if args.task_start_index < 0 or args.task_start_index >= task_length:
        raise ValueError(f'task_start_index {args.task_start_index} is out of range for task {args.task} with length {task_length}')
    if args.task_end_index <= args.task_start_index:
        raise ValueError(
            f'task_end_index {args.task_end_index} must be greater than task_start_index {args.task_start_index}'
        )

    args.task_end_index = min(args.task_end_index, task_length)

    if args.evaluator_model is None:
        args.evaluator_model = args.backend
    if args.evaluator_temperature is None:
        args.evaluator_temperature = defaults.get('evaluator_temperature', 0.7)

def run(args):
    task = get_task(args.task)
    apply_task_defaults(args, len(task))

    configure_llm_backend(
        provider=args.provider,
        default_model=args.backend,
        default_temperature=args.temperature,
        api_key=args.api_key,
        api_base=args.api_base,
        hf_device_map=args.hf_device_map,
        hf_torch_dtype=args.hf_torch_dtype,
        hf_trust_remote_code=args.hf_trust_remote_code,
    )
    reset_gpt_usage()

    if hasattr(task, 'configure_evaluator'):
        task.configure_evaluator(args.evaluator_model, args.evaluator_temperature)

    logs, cnt_avg, cnt_any = [], 0, 0
    backend_name = safe_name(args.backend)
    if args.naive_run:
        file = f'./logs/{args.task}/{backend_name}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    else:
        file = f'./logs/{args.task}/{backend_name}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    os.makedirs(os.path.dirname(file), exist_ok=True)

    for i in range(args.task_start_index, args.task_end_index):
        # solve
        if args.naive_run:
            ys, raw_info = naive_solve(args, task, i) 
        else:
            ys, raw_info = solve(args, task, i)
        info: Dict[str, Any] = dict(raw_info)

        # log
        infos = [task.test_output(i, y) for y in ys]
        info['idx'] = i
        info['ys'] = ys
        info['infos'] = infos
        info['usage_so_far'] = gpt_usage(args.backend)
        logs.append(info)
        with open(file, 'w') as f:
            json.dump(logs, f, indent=4)
        
        # log main metric
        accs = [info['r'] for info in infos]
        cnt_avg += sum(accs) / len(accs)
        cnt_any += any(accs)
        print(i, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg, 'cnt_any', cnt_any, '\n')
    
    n = args.task_end_index - args.task_start_index
    print(cnt_avg / n, cnt_any / n)
    print('usage_so_far', gpt_usage(args.backend))


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--provider', type=str, choices=['openai', 'openai-compatible', 'transformers'], default='openai')
    args.add_argument('--backend', '--model', dest='backend', type=str, default='gpt-4')
    args.add_argument('--temperature', type=float, default=None)
    args.add_argument('--evaluator_model', type=str, default=None)
    args.add_argument('--evaluator_temperature', type=float, default=None)

    args.add_argument('--api_key', type=str, default=None)
    args.add_argument('--api_base', type=str, default=None)
    args.add_argument('--hf_device_map', type=str, default='auto')
    args.add_argument('--hf_torch_dtype', type=str, default='auto')
    args.add_argument('--hf_trust_remote_code', action='store_true')

    args.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords'])
    args.add_argument('--task_start_index', type=int, default=None)
    args.add_argument('--task_end_index', type=int, default=None)

    args.add_argument('--naive_run', action='store_true', default=None)
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'], default=None)  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'], default=None)
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'], default=None)
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default=None)
    args.add_argument('--n_generate_sample', type=int, default=None)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=None)
    args.add_argument('--n_select_sample', type=int, default=None)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)