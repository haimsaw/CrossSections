import os
import sys

exps = [32, 1, 64, 16, 4,128]

for i, exp in enumerate(exps):
    out = f'artifacts/hidden_state_size/{exp}/'
    os.makedirs(out, exist_ok=True)
    cmd = f'\n\npython3 main.py {out} --model dice --gpu {sys.argv[1]} --hidden_state_size {exp} | tee {out}/out.txt'
    print(f'running exp {i+1}/{len(exps)}:\n{cmd}', flush=True)
    os.system(cmd)
