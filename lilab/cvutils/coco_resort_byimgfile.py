# python -m lilab.cvutils.coco_resort_byimgfile coco.json
# %%
import json
import argparse


def convert(cocojson_file):
    # load coco json
    with open(cocojson_file, "r") as f:
        anno_data = json.load(f)

    # get image id to new image id
    anno_images = anno_data["images"]
    anno_annos = anno_data["annotations"]
    fun_filename = lambda x: x["file_name"]
    anno_images.sort(key=fun_filename)
    oldids = [x["id"] for x in anno_images]
    oldid_to_newid = {oldid: iframe + 1 for iframe, oldid in enumerate(oldids)}

    # revise image id
    for anno in anno_annos:
        anno["image_id"] = oldid_to_newid[anno["image_id"]]

    for anno in anno_images:
        anno["id"] = oldid_to_newid[anno["id"]]

    # save file
    out_coco_file = cocojson_file.replace(".json", "_resort.json")
    with open(out_coco_file, "w") as f:
        json.dump(anno_data, f, indent=2)


# %% main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resort COCO annotations file.")
    parser.add_argument(
        "cocojson_file", type=str, help="Path to COCO annotations file."
    )
    args = parser.parse_args()
    convert(args.cocojson_file)
