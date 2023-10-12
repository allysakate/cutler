import os
import json
import argparse
from detectron2.utils.file_io import PathManager

parser = argparse.ArgumentParser()
parser.add_argument(
    "-b",
    "--basedir",
    help="/home/kate.brillantes/thesis/cutler/cutler/output_cvat/inference",
)
args = parser.parse_args()

base_dir = args.basedir
annotations = []
num_chunk = 25
for chunk_id in range(num_chunk):
    with open(
        os.path.join(base_dir, f"coco_instances_{chunk_id}.json"), "r", encoding="utf8"
    ) as f:
        chunk_data = json.load(f)
        annotations.extend(chunk_data)
# coco_dict["annotations"] = annotations

file_path = os.path.join(base_dir, "coco_instances_results.json")
print(f"Saving results to {file_path}, Length: {len(annotations)}")
with PathManager.open(file_path, "w") as f:
    f.write(json.dumps(annotations))
    f.flush()
