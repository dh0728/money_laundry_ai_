# 2.10.0+cu126 12.6 2.8.1 0.8.0+pt210cu126 2.0.0 4.7.0 3.4.1
# nvidia-smi; nvcc --version 2>/dev/null | tail -1
python -V; pip freeze > /workspace/money_laundry_ai_/docs/binary_classification/requirements_old_server.txt
python -c "import torch, torch_geometric, pyg_lib, duckdb, lightgbm, xgboost; print(torch.__version__, torch.version.cuda, torch_geometric.__version__, pyg_lib.__version__, duckdb.__version__, lightgbm.__version__, xgboost.__version__)"
nproc; free -g; cat /sys/fs/cgroup/memory.max; df -h /workspace; ls -la /dev/shm | head -280
               total        used        free      shared  buff/cache   available
Mem:             503          45         178           0         283         458
Swap:              0           0           0
29360128000
Filesystem      Size  Used Avail Use% Mounted on
/dev/rbd1       503G  355G  149G  71% /workspace
head: invalid trailing option -- n
Try 'head --help' for more information.
# Python 3.12.3
# 2.10.0+cu126 12.6 2.8.1 0.8.0+pt210cu126 2.0.0 4.7.0 3.4.1