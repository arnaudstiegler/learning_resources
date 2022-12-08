# CLIP training

Simply testing a contrastive learning for text/image pairs (LAION dataset).

Data can be found here: https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet

Steps:
- Download data locally using provided link to DATA_PATH
- Run `python clip_training/datagen.py --filepath {DATA_PATH} --dest {IMAGE_PATH}`
- Run training using `python clip_training/run_training.py --filepath {DATA_PATH} --images_path {IMAGE_PATH}`