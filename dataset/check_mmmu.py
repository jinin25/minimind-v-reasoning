heck_mmmu.pyimport os

path = r"D:\daily\study\ai\LLM\minimind-v\minimind-v\dataset\sft_i2t.parquet"
print(os.path.getsize(path) / 1024 / 1024, "MB")

with open(path, "rb") as f:
    print(f.read(100))