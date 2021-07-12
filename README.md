Modification of the QuickNet from larq.sota

Includes all the code for versions of quicknet with some or all of the batchnorms replaced with fixed normalizations.

CustomRenorm has the code for all the renormalizations.

training.py is the training loop. Easy to replace if you have your own training loop.

All the other files are versions of quicknet.

To use, install the packages in the requirements file, then make local directories "weights" and "tabular." 
Make sure you have ImageNet installed correctly, so that tfds (tensorflow-datasets) can see it and import it.

