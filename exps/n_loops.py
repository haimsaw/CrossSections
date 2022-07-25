import os
import sys

exps = [1, 10, 20, 5, 15]

for i, exp in enumerate(exps):
    out = f'artifacts/n_loops/{exp}/'
    os.makedirs(out, exist_ok=True)
    cmd = f'python3 main.py {out}' \
          f' --gpu {sys.argv[1]}' \
          f' --model Elephant' \
          f' --n_loops {exp}' \
          f' | tee {out}/out.txt'

    print(f'\n\nrunning exp {i+1}/{len(exps)}:\n{cmd}', flush=True)
    os.system(cmd)
