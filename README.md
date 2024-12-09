# ai4con-wildfire

put numpy objs into data folder
```
data/
  - LC09_CU_011002_20241108_20241113_02_mask_block_7.pt.npz
  - LC09_CU_011002_20241108_20241113_02_mask_block_8.pt.npz
  - LC09_CU_011002_20241108_20241113_02_mask_block_10.pt.npz
```

run the data_preprocessor.ipynb and you should end up with
data_processed/
  train/
    - LC09_CU_011002_20241108_20241113_02_mask_block_8.pt.npz
  val/
    - LC09_CU_011002_20241108_20241113_02_mask_block_9.pt.npz
  test/
    - LC09_CU_011002_20241108_20241113_02_mask_block_10.pt.npz


they are stored as floats from 0 to 1 representing the radiance observed

# training
start visdom server in a seperate terminal
```bash
python -m visdom.server
```

another terminal
```bash
python main.py --config /path/to/config.yml
python main.py --config /path/to/config.yml --name Something
```