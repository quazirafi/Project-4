# Project-4

1. run the following command to install Virtual Environment ''python3 -m venv {{environment_name}}''
2. Activate the virtual environment using the following command ''source {{environment_name}}/bin/activate''
3. Install the dependencies using the following command. ''pip install -r requirements.txt''

sft-ft.jsonl contains the CUDAPerf Dataset

struct_ranker-64-min-speedup-1-epoch-150.json contains the learned feature weights.

1. Run python QiMeng-MUPA.py to generate the results on BabelTower Dataset with CUDAPerf
2. Run python cuda-perf-main.py to generate the results on CUDAPerf Dataset with CUDAPerf
3. Run python cuda-perf-kernelbench.py to generate the results on KernelBench Dataset with CUDAPerf.
4. Run qwen-2.5-exp.py to run experiments with Qwen-2.5 model.
5. Run codellama-exp.py to run experiments with CodeLlama model.
