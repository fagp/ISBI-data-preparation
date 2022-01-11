JCell training - CTC VI Challenge
=================================

1. **Installing the software**

Install ``jcell`` and dependencies by typing ``bash prepare_software.sh``. Alternatively, you can manually install them using the provided wheel and requirements files:

```bash
apt install wget unzip
pip install jcell-ISBI==0.0.1a4
pip install -r requirements.txt
jcell-update
```

2.1 **Preparing Challenge datasets**

Our approach operates with semantic labels, which is different from CTC challenge's instance-level annotations. For such purpose, an instance-to-semantic conversion must be applied:

```bash
python prepare_ISBI_data.py
```

The script will automatically download both training and test datasets. Additionally, it will slice 3D volumes converting them to 2D images. Finally, GT, ST, and GT+ST data configuration will be organized as expected by ``jcell``.

2.2 **Preparing new datasets**

Prepare new datasets using the instance-to-semantic transformation script. In the following example it is shown the execution for the Fluo-C2DL-Huh7 dataset.

```bash
python prepare_new_data.py --datasets Fluo-C2DL-Huh7
```
The required folder structure is:



    ./training
    ├── train_val
    │   └── Fluo-C2DL-Huh7
    │       ├── 01
    │       ├── 01_GT
    │       ├── 02
    │       ├── 02_GT
    ├── train_scripts
    │   └── configuration
    ├── fix_relative_path.sh
    ├── jcell-0.0.1a4_ISBI-py3-none-any.whl
    ├── prepare_ISBI_data.py
    ├── prepare_new_data.py
    ├── prepare_software.sh
    ├── README.md
    └── requirements.txt

where the folder ``train_val`` will contain a typical CTC dataset. Such a folder must be on the same level of the ``prepare_new_data.py`` script.

3. **Configuring datasets**

A ``json`` configuration file is available for specifying the path to a given dataset (``./train_scripts/configuration/dataconfig_train.json``). All entries must have the same structure to define a new dataset successfully. However, the current version of our software doesn't allow relatives paths. For running our current solution in your directory, we recommend executing the following script  after installation: 

```bash
bash fix_relative_path.sh
```

For adding new datasets, you can modify ``dataconfig_train.json`` directly. Please, verify that ``"dataset_folder"`` key point to your dataset. 

4. **Training with jcell**

We provide all the scripts used for training. You can find them in ``./train_scripts`` folder. For running a training job, do, for example:

```bash
cd train_scripts
bash DIC-C2DH-HeLa_GT.sh
```

The resulting experiment (models, logs, etc.) will be saved into ``./train_scripts/out`` folder.

5. **Evaluation with jcell**

A general evaluation script is provided under the name ``general_test.sh``. However, this script must be modified to properly point to the test dataset path and the corresponding model.

```bash
bash general_test.sh
```

NOTE: The model trained for your custom dataset will be saved into ``./train_scripts/out/YOUR_DATASET/model/lastmodel.t7``
