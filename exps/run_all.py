import os
import sys

out = f'artifacts/base_from_mesh/'
os.makedirs(out, exist_ok=True)
cmd = f'python3 main.py {out}' \
      f' --gpu {sys.argv[1]}' \
      f' | tee {out}/out.txt'
print(f'\n\nrunning exp:\n{cmd}', flush=True)
os.system(cmd)

out = f'artifacts/base_from_mri/'
os.makedirs(out, exist_ok=True)
cmd = f'python3 main.py {out}' \
      f' --gpu {sys.argv[1]}' \
      f' -run_mri' \
      f' | tee {out}/out.txt'
print(f'\n\nrunning exp:\n{cmd}', flush=True)
os.system(cmd)