#
# python labelme2deeplabcut.py --yaml /home/chenxinfeng/deeplabcut-project/abc/config.yaml --json /home/chenxinfeng/deeplabcut-project/abc/rat_keypoints
# %% imports
import os
import json
import pandas as pd
import yaml
import shutil
import argparse

# %% read the config.yaml file
def main(yamlfile="config.yaml", json_folder="rat_keypoints"):
    with open(yamlfile, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    bodyparts = cfg["bodyparts"]
    scorer = cfg["scorer"]

    # %% list all json files in the folder
    json_files = [
        pos_json for pos_json in os.listdir(json_folder) if pos_json.endswith(".json")
    ]
    json_fullfiles = [os.path.join(json_folder, pos_json) for pos_json in json_files]
    # %%
    bodyparts_set = set(bodyparts)
    labels_set = set()

    # %% check the file with desired labels
    for i, json_file in enumerate(json_fullfiles):
        # get the file name of json_file
        json_file_name = os.path.basename(json_file)
        print(i, json_file_name)
        with open(json_file) as f:
            data = json.load(f)
            shapes = data["shapes"]
            labels = [shape["label"] for shape in shapes]
            labels_set.update(labels)
            assert labels_set.issubset(
                bodyparts_set
            ), "The labels are not the same as the ones in the template"
    else:
        print("All files are read")

    # %% create new dataframe with multiple headers as 'scorer', 'bodyparts' and 'coords'
    scorers = [scorer]
    coords = ["x", "y"]

    # create multiindex header from the above lists
    header = pd.MultiIndex.from_product(
        [scorers, bodyparts, coords], names=["scorer", "bodyparts", "coords"]
    )
    df_out = pd.DataFrame(columns=header, dtype="float", index=json_fullfiles)

    # %% read all data from json files to pandas
    for i, json_file in enumerate(json_fullfiles):
        # get the file name of json_file
        json_file_name = os.path.basename(json_file)
        print(i, json_file_name)
        with open(json_file) as f:
            data = json.load(f)
            shapes = data["shapes"]
            df_file = df_out.loc[json_file]
            for shape in shapes:
                label = shape["label"]
                point_x, point_y = shape["points"][0]
                df_out.loc[json_file, (scorer, label, "x")] = point_x
                df_out.loc[json_file, (scorer, label, "y")] = point_y

    # %% get the indexs of the dataframe
    # get the nakename of the json files without the extension
    image_fullfiles = [
        "labeled-data/output/{}.png".format(
            os.path.splitext(os.path.basename(json_file))[0]
        )
        for json_file in json_fullfiles
    ]
    df_out.index = image_fullfiles

    # %% save the dataframe to hdf5 file and csv file
    h5_file_out = os.path.join(json_folder, "CollectedData_{}.h5".format(scorer))
    csv_file_out = os.path.join(json_folder, "CollectedData_{}.csv".format(scorer))
    df_out.to_hdf(h5_file_out, "df_with_missing")
    df_out.to_csv(csv_file_out)

    # %% create the directory 'labeled/output' if not exsit
    # get the parent directory of the yaml file
    dlc_project_dir = os.path.dirname(os.path.abspath(yamlfile))
    target_folder = os.path.join(dlc_project_dir, "labeled/output")
    os.makedirs(target_folder, exist_ok=True)
    # %% find all the image in the json_folder, and copy them to the 'labeled/output/' folder without overwritten
    image_files = [
        pos_image for pos_image in os.listdir(json_folder) if pos_image.endswith(".png")
    ]
    image_fullfiles = [
        os.path.join(json_folder, pos_image) for pos_image in image_files
    ]
    for image_file in image_fullfiles:
        image_file_name = os.path.basename(image_file)
        target_file = os.path.join(target_folder, image_file_name)
        if not os.path.exists(target_file):
            shutil.copy(image_file, target_folder)
    shutil.copy(h5_file_out, target_folder)
    shutil.copy(csv_file_out, target_folder)


# %% arg parse the yaml file and json folder to __main__
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Labelme2deeplabcut")
    parser.add_argument(
        "--yaml", type=str, default="config.yaml", help="config.yaml file"
    )
    parser.add_argument(
        "--json_folder", type=str, default="rat_keypoints", help="labelme json folder"
    )
    args = parser.parse_args()
    main(yamlfile=args.yaml, json_folder=args.json_folder)
