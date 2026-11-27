# Running on a cluster

1. Copy the project to the cluster, including the current `train.py`,
   `data_loader.py`, and `Models/` files. Download or copy the data into the
   paths shown in the [README](README.md#data-folder-layout). The full run
   needs all four bigWigs.
2. From the project directory, create a Python environment using your
   cluster's instructions. Install a CUDA-enabled PyTorch build for the
   cluster's GPU using the [PyTorch installation guide](https://pytorch.org/get-started/locally/),
   then run `python -m pip install -r requirements.txt` in that environment.
   The compute node also needs a compatible NVIDIA driver.
3. Request a GPU compute node using the cluster's instructions, activate your
   Python environment, and run these commands from the project directory:

   ```bash
   nvidia-smi
   python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'"
   ```

   For a short smoke test, run:

   ```bash
   python -u -c "from train import run_fold; run_fold('GM12878', 0, epochs=1, batch_size=2, max_batches=2, ckpt_dir='checkpoints_smoke')"
   ```

   The smoke test still prepares the chromosome datasets. Check for errors and
   `Using device: cuda`, then start the full run:

   ```bash
   python -u train.py
   ```

The full command trains 20 models sequentially (four cell lines, five folds
each), using up to 10 epochs with early stopping. Checkpoints are saved in
`checkpoints/`; back up existing files first because matching names can be
overwritten. A stopped job does not automatically resume training.
