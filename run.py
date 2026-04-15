import os
import json
import argparse

from tot.tasks import get_task
from tot.methods.bfs import solve, naive_solve
from tot.models import configure_llm_backend, gpt_usage, reset_gpt_usage

def run(args):
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

    task = get_task(args.task)
    if hasattr(task, 'configure_evaluator'):
        task.configure_evaluator(args.evaluator_model, args.evaluator_temperature)

    logs, cnt_avg, cnt_any = [], 0, 0
    if args.naive_run:
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    else:
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    os.makedirs(os.path.dirname(file), exist_ok=True)

    for i in range(args.task_start_index, args.task_end_index):
        # solve
        if args.naive_run:
            ys, info = naive_solve(args, task, i) 
        else:
            ys, info = solve(args, task, i)

        # log
        infos = [task.test_output(i, y) for y in ys]
        info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far': gpt_usage(args.backend)})
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
    args.add_argument('--temperature', type=float, default=0.7)
    args.add_argument('--evaluator_model', type=str, default=None)
    args.add_argument('--evaluator_temperature', type=float, default=0.0)

    args.add_argument('--api_key', type=str, default=None)
    args.add_argument('--api_base', type=str, default=None)
    args.add_argument('--hf_device_map', type=str, default='auto')
    args.add_argument('--hf_torch_dtype', type=str, default='auto')
    args.add_argument('--hf_trust_remote_code', action='store_true')

    args.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords'])
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy')
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)

    args = args.parse_args()
    if args.evaluator_model is None:
        args.evaluator_model = args.backend
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)