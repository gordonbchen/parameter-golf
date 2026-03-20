import os
from pathlib import Path
import numpy as np

data_path = "/workspace/parameter-golf/data/datasets/fineweb10B_sp1024"
train_files = os.path.join(data_path, "fineweb_train_*.bin")

for file in sorted(Path(data_path).glob("fineweb_train_*.bin")):
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize

    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")

    num_tokens = int(header[2])
    print(file.name, num_tokens)

# 100_000_000, so 100 M tokens per shard. each token is 2 bytes (int16), so each shard is 200 Mb.
# so 80 shards = 80 * 200Mb = 16000 Mb = 16Gb.
# in total there are 80 * 100M = 8 B tokens.
# Sota sees 7,055,769,600 tokens, so 7 B tokens.
