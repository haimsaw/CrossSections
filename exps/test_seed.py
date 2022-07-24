import os
import sys

for i in range(5):
    out = f'artifacts/test_random/{i}/'
    os.makedirs(out, exist_ok=True)
    cmd = f'python3 main.py {out}' \
          f' --gpu {sys.argv[1]}' \
          f' --model armadillo' \
          f' --seed -1' \
          f' | tee {out}/out.txt'
    print(f'\n\nrunning exp {i+1}/{5}:\n{cmd}', flush=True)
    os.system(cmd)
