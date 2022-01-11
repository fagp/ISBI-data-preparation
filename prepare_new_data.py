import os
import argparse
import imageio
from shutil import copyfile
import numpy as np
import json

os.environ["MKL_THREADING_LAYER"] = "GNU"

training_parameters = "#!/bin/bash\n\n# --experiment          indicate the name of the experiment. A folder with the same name will be created to save the model\n# --configuration_path  folder with a json files with data and metrics configuration\n# --dataset             training configuration corresponding to a key in dataconfig_train.json\n# --optimizer           optimizer to be used, i.e., SGD, Adam, AdaBelief. Support to any PyTorch optimizer and AdaBelief if installed.\n# --optimizer_param     optimizer parameters. In this example is used to initialize learning rate. Support to any argument in the form of a dict.\n# --model_param         model parameters. In this example is used to specify an initial model. The default model is UNET with 2d convolutions\n# --loss                loss function to be used during training. In this example is used Cross Entropy + J regularization\n# --loss_param          loss parameters. In this example is used to initialize lambda matrix to 0.05 for J regularization\n# --epochs              number of epochs\n# --batch_size          mini-batch size\n# --use_gpu             used to specify which gpu to use begining from 1. If included more than once then it works in a multi-gpu setting\n# --visdom              flag to indicate that log must be created. So far only support visdom.\n\n# for help in the rest of the parameters type `jcell train --help` in your terminal\n\njcell_models=$(cat ~/.jcell/jcellrc)\n\n"


def create_train_script(datasets, modes):
    script_folder = "train_scripts"

    for data in datasets:
        for mode in modes:
            if not os.path.exists(os.path.join("train_val", data, "labels_" + mode)):
                continue
            current_dataset = "{}-{}".format(data, mode)
            script_file = os.path.join(script_folder, "{}_{}.sh".format(data, mode))
            execution_line = (
                "jcell train --experiment "
                + current_dataset
                + " --configuration_path configuration --output_path ../ --dataset "
                + current_dataset
                + " --optimizer AdaBelief --model_param \"{'init':'${jcell_models}/models/GENERALIST_v0.t7'}\" --optimizer_param \"{'lr':0.00001}\" --save_rate 20 --train_worker 64 --dev_worker 16 --visdom --batch_size 2  --epochs 100 --use_gpu 1 --loss_param \"{'lambda_const':0.05}\" --loss wce_j"
            )
            with open(script_file, "w") as f:
                f.write(training_parameters)
                f.write(execution_line)


def create_configuration(datasets, modes):
    config_file = os.path.join(
        "train_scripts", "configuration", "dataconfig_train.json"
    )

    config = json.load(open(config_file))

    for data in datasets:
        for mode in modes:
            if not os.path.exists(os.path.join("train_val", data, "labels_" + mode)):
                continue
            current_dataset = "{}-{}".format(data, mode)
            entry = config.get(current_dataset, dict())

            entry["dataset_folder"] = "../train_val/{}".format(data)
            entry["image_folder"] = "images_{}".format(
                mode if mode != "GT+ST" else "ST"
            )
            entry["label_folder"] = "labels_{}".format(mode)
            entry["number_classes"] = "4"
            entry["is_3D"] = "True" if "3D" in data else "False"
            entry["split"] = ""
            entry[
                "transform_parameter"
            ] = "NormalizePercentile(), Rotation(180), RandomFlip(0.5), RandomCrop((512,512)), ToTensor()"

            config[current_dataset] = entry

    json.dump(config, open(config_file, "w"))
    json.dump(
        config,
        open(
            config_file.replace(
                "dataconfig_train.json", "dataconfig_train_backup.json"
            ),
            "w",
        ),
    )


# function for downloading all test datasets
def check_challenge_datasets(datasets):
    if not os.path.exists("test"):
        raise ValueError("{}/test folder not found".format(os.curdir))

    for data in datasets:
        if not os.path.exists(os.path.join("test", data)):
            raise ValueError("{}/test/{} folder not found".format(os.curdir, data))


# function for downloading all training datasets
def check_training_datasets(datasets):
    if not os.path.exists("train_val"):
        raise ValueError("{}/train_val folder not found".format(os.curdir))

    for data in datasets:
        if not os.path.exists(os.path.join("train_val", data)):
            raise ValueError("{}/train_val/{} folder not found".format(os.curdir, data))


# transform instance label to semantic label following our paper
# [1] J Regularization Improves Imbalanced Multiclass Segmentation
# https://ieeexplore.ieee.org/abstract/document/9098550/
# arxiv: https://arxiv.org/pdf/1910.09783
def create_semantic_label(datasets, sequences, modes):
    # only applies to images with annotation
    os.chdir("train_val")

    # convert instance to semantic in the two given folders (GT and ST)
    subset_modes = [m for m in modes if m in ["GT", "ST"]]

    for mode in subset_modes:
        # for every dataset
        for data in datasets:
            # for every sequence (01 and 02)
            for seq in sequences:

                # check whether we are in 2D or 3D case specify in the name.
                # 3D cases uses only slices of the volume following CTC instructions
                is_3D = "3D" in data

                # path to instances labels
                path_instances = (
                    os.path.join(data, "{}_{}".format(seq, mode), "SEG")
                    if not is_3D
                    else os.path.join(data, "slices_instances_{}_{}".format(seq, mode))
                )

                # path to images
                path_source_images = (
                    os.path.join(data, seq)
                    if not is_3D
                    else os.path.join(data, "slices_images_{}".format(seq))
                )

                # path to output semantic labels
                path_labels = os.path.join(data, "labels_{}".format(mode))

                # path to output images.
                # only images with a corresponding label are copied
                path_target_images = os.path.join(data, "images_{}".format(mode))

                # check if the instance path exist
                # useful to check if 3D slices were sucessfully computed
                if not (
                    os.path.exists(path_instances) and os.path.isdir(path_instances)
                ):
                    continue

                print("Creating semantic labels for {}:{}_{}".format(data, seq, mode))

                # 2D annotations are depth 3 relative the current folder
                # 3D slices (previously extracted) are depth 2 relative the current folder
                # then, the output folder will be in the same level as the image folder for any dataset
                output_path = (
                    "../../../" + path_labels if not is_3D else "../../" + path_labels
                )

                # system call to jcell-dataproc.
                # convert instance annotation to semantic label for every image in the input folder
                # following [1]. The output folder is created if doesn't exist.
                os.system(
                    "jcell-dataproc inst2sem --input {} --output {} --se1 1 --se2 7 --overwrite".format(
                        path_instances, output_path
                    )
                )

                # then copy only images with a corresponding label annotation
                os.makedirs(path_target_images, exist_ok=True)

                # annotation are expected to follow CTC naming convention, i.e., man_seg%Nd.tif
                label_list = [f for f in os.listdir(path_labels) if f[:7] == "man_seg"]

                # for every annotation file
                for label in label_list:
                    # change name to "data-seq-mode_%Nd.tif" regex.
                    new_name = label.replace(
                        "man_seg", "{}-{}-{}_".format(data, seq, mode)
                    )
                    # rename semantic label file
                    os.rename(
                        os.path.join(path_labels, label),
                        os.path.join(path_labels, new_name),
                    )
                    # copy and rename image file
                    # images are expected to follow CTC naming convention, i.e., t%Nd.tif
                    copyfile(
                        os.path.join(path_source_images, label.replace("man_seg", "t")),
                        os.path.join(path_target_images, new_name),
                    )

    os.chdir("../")


# extract slices from 3D volumes for semantic labels and images
def extract_slices3D(datasets, sequences, modes):
    # only applies to images with annotations
    os.chdir("train_val")

    subset_modes = [m for m in modes if m in ["GT", "ST"]]

    for mode in subset_modes:
        for data in datasets:
            # check whether we are in 2D or 3D case specify in the name.
            # 3D cases uses only slices of the volume following CTC instructions
            if "3D" not in data:
                # slicing function only applies to 3D datasets
                continue

            for seq in sequences:
                # path to instances labels
                # labels may be either 3D volumes or 2D slices from the volumes
                path_instances = os.path.join(data, "{}_{}".format(seq, mode), "SEG")
                # path to original 3D volumes
                path_source_images = os.path.join(data, seq)

                # path to output instances labels slices
                path_sliced_instances = os.path.join(
                    data, "slices_instances_{}_{}".format(seq, mode)
                )

                # path to output volumes slices
                path_target_images = os.path.join(data, "slices_images_{}".format(seq))

                # check if the instance path exist
                if not (
                    os.path.exists(path_instances) and os.path.isdir(path_instances)
                ):
                    continue

                print("Slicing {}:{}_{}".format(data, seq, mode))

                # create annotation output folder
                os.makedirs(path_sliced_instances, exist_ok=True)
                # annotation are expected to follow CTC naming convention, i.e., man_seg%Nd.tif
                instances_list = [
                    f for f in os.listdir(path_instances) if f[:7] == "man_seg"
                ]

                # for every annotation file
                for instance in instances_list:
                    # read instance label volume
                    image = imageio.volread(os.path.join(path_instances, instance))
                    # if 2D slices already
                    if image.ndim == 2:
                        # create new name following CTC naming convention for slices,
                        # i.e., man_seg%Nd_%Nd.tif
                        name = "man_seg{}".format(instance[8:])

                        # copy the slice into the output folder with the new name
                        imageio.imwrite(
                            os.path.join(path_sliced_instances, name),
                            image,
                        )

                    # if 3D volume
                    else:
                        # for every slices of the volume
                        for sl in range(image.shape[0]):
                            # create new name following CTC naming convention for slices,
                            # i.e., man_seg%Nd_%Nd.tif
                            name = (
                                "{}_{:03d}.tif".format(
                                    os.path.splitext(instance)[0], sl
                                )
                                if image.shape[0] > 1
                                else "man_seg{}".format(instance[7:])
                            )
                            # copy the slice into the output folder with the new name
                            imageio.imwrite(
                                os.path.join(path_sliced_instances, name),
                                image[sl, :, :],
                            )

                # only slice volume if mode is GT. Avoid slicing multiple times the same volume
                if mode != "GT":
                    continue

                # create image output folder
                os.makedirs(path_target_images, exist_ok=True)
                # get all images inside the folder
                images_list = [f for f in os.listdir(path_source_images)]

                # for every volume
                for org_image in images_list:
                    # read input volume
                    image = imageio.volread(os.path.join(path_source_images, org_image))

                    # compute 1 and 99 percentile to be used in
                    # percentile normalization during training
                    vol_min, vol_max = np.percentile(image, 1), np.percentile(image, 99)
                    # modify the first 6 pixels to encode normalization factors
                    # these values will be read during training and used to
                    # normalize the image. guarantees volume-wise normalization.
                    # the first four pixels are used to encode the word `norm`
                    # indicating that normalization values are given.
                    image[:, 0, 0] = 110  # char n
                    image[:, 0, 1] = 111  # char o
                    image[:, 0, 2] = 114  # char r
                    image[:, 0, 3] = 109  # char m
                    image[:, 0, 4] = vol_min
                    image[:, 0, 5] = vol_max

                    # for every slice in the volume
                    for sl in range(image.shape[0]):
                        # create new name following CTC naming convention for slices,
                        # i.e., man_seg%Nd_%Nd.tif
                        name = (
                            "{}_{:03d}.tif".format(os.path.splitext(org_image)[0], sl)
                            if image.shape[0] > 1
                            else org_image
                        )
                        # copy the slice into the output folder with the new name
                        imageio.imwrite(
                            os.path.join(path_target_images, name),
                            image[sl, :, :],
                        )

    os.chdir("../")


# creates GT+ST modality
def create_GT_ST(datasets, sequences, modes):
    # only applies to images with annotations
    os.chdir("train_val")

    subset_modes = [m for m in modes if m in ["GT+ST"]]

    for mode in subset_modes:
        for data in datasets:
            for seq in sequences:

                # check whether we are in 2D or 3D case specify in the name.
                # 3D cases uses only slices of the volume following CTC instructions
                is_3D = "3D" in data

                # path to semantic labels for GT and ST modes.
                # this folders must be created first by calling create_semantic_label()
                path_instances_GT = os.path.join(data, "labels_GT")
                path_instances_ST = os.path.join(data, "labels_ST")

                # path to output folder labels_GT+ST
                path_labels = os.path.join(data, "labels_{}".format(mode))

                # check if the instance path exist
                if not (
                    os.path.exists(path_instances_GT)
                    and os.path.isdir(path_instances_GT)
                    and os.path.exists(path_instances_ST)
                    and os.path.isdir(path_instances_ST)
                ):
                    continue

                print("Creating GT+ST modality for {}:{}_{}".format(data, seq, mode))

                # create output folder
                os.makedirs(path_labels, exist_ok=True)

                # get images inside semantic labels folders
                images_GT = [f for f in os.listdir(path_instances_GT)]
                images_ST = [f for f in os.listdir(path_instances_ST)]

                # for every image in ST folder
                for label in images_ST:
                    # get path for ST label
                    path = os.path.join(path_instances_ST, label)

                    # if the same image has a GT annotation
                    if os.path.exists(
                        os.path.join(path_instances_GT, label.replace("ST", "GT"))
                    ):
                        # then update path for GT annotation
                        path = os.path.join(
                            path_instances_GT, label.replace("ST", "GT")
                        )

                    # copy the selected image to GT+ST annotation folder
                    copyfile(
                        path,
                        os.path.join(path_labels, label),
                    )
    os.chdir("../")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help="list of datasets",
    )
    parser.add_argument(
        "--sequences",
        default=["01", "02"],
        type=str,
        nargs="+",
        help="list of sequences",
    )
    parser.add_argument(
        "--modes",
        default=["GT", "ST", "GT+ST"],
        type=str,
        nargs="+",
        help="list of modes",
    )

    return parser


def main():

    args = get_parser().parse_args()
    datasets = args.datasets
    sequences = args.sequences
    modes = args.modes

    check_training_datasets(datasets)
    extract_slices3D(datasets, sequences, modes)
    create_semantic_label(datasets, sequences, modes)
    create_GT_ST(datasets, sequences, modes)
    # check_challenge_datasets(datasets)
    create_configuration(datasets, modes)
    create_train_script(datasets, modes)
    os.system("./fix_relative_path.sh")


if __name__ == "__main__":
    """
    Main function to train model.
    """
    main()
