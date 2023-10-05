import os
import json
from detectron2.utils.file_io import PathManager

BASE_DIR = "/home/kate.brillantes/thesis/cutler/cutler/output/self-train-r1/inference"

# Load info.json

# with open(os.path.join(BASE_DIR, "info.json"), "r", encoding="utf8") as f:
#     data = json.load(f)
#     coco_dict = {
#         "info": data["info"],
#         "licenses": None,
#         "categories": data["categories"],
#         "images": data["images"],
#     }

annotations = []
num_chunk = 25
for chunk_id in range(num_chunk):
    with open(
        os.path.join(BASE_DIR, f"coco_instances_{chunk_id}.json"), "r", encoding="utf8"
    ) as f:
        chunk_data = json.load(f)
        annotations.extend(chunk_data)
# coco_dict["annotations"] = annotations

file_path = os.path.join(BASE_DIR, "coco_instances_results.json")
print(f"Saving results to {file_path}, Length: {len(annotations)}")
with PathManager.open(file_path, "w") as f:
    f.write(json.dumps(annotations))
    f.flush()
