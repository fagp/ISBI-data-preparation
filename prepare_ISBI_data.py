import os
import imageio
from shutil import copyfile
import numpy as np

os.environ["MKL_THREADING_LAYER"] = "GNU"

# define here the list of datasets to be used for training
datasets = [
    "BF-C2DL-HSC",
    "BF-C2DL-MuSC",
    "DIC-C2DH-HeLa",
    "Fluo-C2DL-MSC",
    "Fluo-C3DH-A549",
    "Fluo-C3DH-H157",
    "Fluo-C3DL-MDA231",
    "Fluo-N2DH-GOWT1",
    "Fluo-N2DL-HeLa",
    "Fluo-N3DH-CE",
    "Fluo-N3DH-CHO",
    "PhC-C2DH-U373",
    "PhC-C2DL-PSC",
]

# following CTC guideline there are two sequences
sequences = ["01", "02"]

# and for the primary track there are six training configurations
modes = ["GT", "ST", "GT+ST", "allGT", "allST", "allGT+ST"]


# function for downloading a dataset from CTC server
def download_dataset(dataset, training=True):
    """
    ARGS:
    =====
        -dataset: dataset name to be downloaded from CTC site
        -training: flag to specify either training or testing data
    """

    print(
        "Downloading {} - {}".format(dataset, "train" if training else "test")
    )

    # system call to wget for downloading the data
    os.system(
        "wget https://data.celltrackingchallenge.net/{}-datasets/{}.zip".format(
            "training" if training else "challenge", dataset
        )
    )
    # unzip the retrieved file
    os.system("unzip {}.zip".format(dataset))
    # delete the zip file
    os.system("rm {}.zip".format(dataset))


# function for downloading all test datasets
def download_challenge_dataset():
    # data will be saved in 'test/'
    os.makedirs("test", exist_ok=True)
    os.chdir("test")

    for data in datasets:
        download_dataset(data, training=False)

    os.chdir("../")


# function for downloading all training datasets
def download_training_dataset():
    # data will be saved in 'train_val/'
    os.makedirs("train_val", exist_ok=True)
    os.chdir("train_val")

    for data in datasets:
        download_dataset(data, training=True)

    os.chdir("../")


# transform instance label to semantic label following our paper
# [1] J Regularization Improves Imbalanced Multiclass Segmentation
# https://ieeexplore.ieee.org/abstract/document/9098550/
# arxiv: https://arxiv.org/pdf/1910.09783
def create_semantic_label():
    # only applies to images with annotation
    os.chdir("train_val")

    # convert instance to semantic in the two given folders (GT and ST)
    for mode in ["GT", "ST"]:
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
                    else os.path.join(
                        data, "slices_instances_{}_{}".format(seq, mode)
                    )
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
                path_target_images = os.path.join(
                    data, "images_{}".format(mode)
                )

                # check if the instance path exist
                # useful to check if 3D slices were sucessfully computed
                if not (
                    os.path.exists(path_instances)
                    and os.path.isdir(path_instances)
                ):
                    continue

                print(
                    "Creating semantic labels for {}:{}_{}".format(
                        data, seq, mode
                    )
                )

                # 2D annotations are depth 3 relative the current folder
                # 3D slices (previously extracted) are depth 2 relative the current folder
                # then, the output folder will be in the same level as the image folder for any dataset
                output_path = (
                    "../../../" + path_labels
                    if not is_3D
                    else "../../" + path_labels
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
                label_list = [
                    f for f in os.listdir(path_labels) if f[:7] == "man_seg"
                ]

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
                        os.path.join(
                            path_source_images, label.replace("man_seg", "t")
                        ),
                        os.path.join(path_target_images, new_name),
                    )

    os.chdir("../")


# extract slices from 3D volumes for semantic labels and images
def extract_slices3D():
    # only applies to images with annotations
    os.chdir("train_val")

    for mode in ["GT", "ST"]:
        for data in datasets:
            # check whether we are in 2D or 3D case specify in the name.
            # 3D cases uses only slices of the volume following CTC instructions
            if "3D" not in data:
                # slicing function only applies to 3D datasets
                continue

            for seq in sequences:
                # path to instances labels
                # labels may be either 3D volumes or 2D slices from the volumes
                path_instances = os.path.join(
                    data, "{}_{}".format(seq, mode), "SEG"
                )
                # path to original 3D volumes
                path_source_images = os.path.join(data, seq)

                # path to output instances labels slices
                path_sliced_instances = os.path.join(
                    data, "slices_instances_{}_{}".format(seq, mode)
                )

                # path to output volumes slices
                path_target_images = os.path.join(
                    data, "slices_images_{}".format(seq)
                )

                # check if the instance path exist
                if not (
                    os.path.exists(path_instances)
                    and os.path.isdir(path_instances)
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
                    image = imageio.volread(
                        os.path.join(path_instances, instance)
                    )
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
                    image = imageio.volread(
                        os.path.join(path_source_images, org_image)
                    )

                    # compute 1 and 99 percentile to be used in
                    # percentile normalization during training
                    vol_min, vol_max = np.percentile(image, 1), np.percentile(
                        image, 99
                    )
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
                            "{}_{:03d}.tif".format(
                                os.path.splitext(org_image)[0], sl
                            )
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
def create_GT_ST():
    # only applies to images with annotations
    os.chdir("train_val")

    for mode in ["GT+ST"]:
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

                print(
                    "Creating GT+ST modality for {}:{}_{}".format(
                        data, seq, mode
                    )
                )

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
                        os.path.join(
                            path_instances_GT, label.replace("ST", "GT")
                        )
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


download_training_dataset()
extract_slices3D()
create_semantic_label()
create_GT_ST()
download_challenge_dataset()
