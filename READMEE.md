# What does this repository contain?

This repository contains the code for conducted experiment regarding the master thesis "A systematic impact analysis of common post-training pre-processing methods to mitigate racial bias in face verification". The structure is the following:

* ``evaluation_notebooks`` contains all notebooks used to conduct the analysis of the experiment results. Each processing method has its own notebook witht he results for all chosen models.
* ``experimentation`` is the folder which contains one notebook per processing method to execute the experiment
* ``results`` contains all csv with the data used for experiments and their corresponding results. These are later used in notebooks to evaluate the results

# Data Access

# Spin up Docker

To spin up a Docker container to run these experiments please go to the root folder of this repository and run ``docker-compose up``

# Run notebooks

The notebooks can be exectued by running all cells. This should work without any error message.

Before running the notebooks please download the file ``Equalizedface.tar.gz`` from this link:

https://console.cloud.google.com/storage/browser/racial_bias_data;tab=permissions?forceOnBucketsSortingFiltering=false&hl=de&project=racialbias&prefix=&forceOnObjectsSortingFiltering=false

The following emails have currently access to the data:

* calamia.tino@gmail.com
* philipp.koch@limebit.de

After downloading the file please unzip the file and put the contained folder named ``race_per_7000`` into a folder called ``data/`` in the root directory.

To re-create the experimentation data execute all notebooks in ``experimentation``. However this should not be needed as all experimentation data has been created already and lives under ``data/val_data``.

The evaluation notebooks can be executed and and access the stored ``results``.

