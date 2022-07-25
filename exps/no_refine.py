import os
import sys

exps = [True, False]

for i, exp in enumerate(exps):
    out = f'artifacts/no_refine/{exp}/'
    os.makedirs(out, exist_ok=True)
    cmd = f'python3 main.py {out}' \
          f' --gpu {sys.argv[1]}' \
          f' --model balloondog' \
          f' { "-no_refine" if exp else "" }' \
          f' | tee {out}/out.txt'

    print(f'\n\nrunning exp {i+1}/{len(exps)}:\n{cmd}', flush=True)
    os.system(cmd)
