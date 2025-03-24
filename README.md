# PassiveScalarPINNs
Physics-Informed Neural Network (PINN) for reconstructing 2D RANS flows with passive scalar dispersion.

The folder 'Code\' contains the Python files executed by the HPC to generate the models for all flow cases and investigation types. The programs require OpenFOAM time-averaged velocity fields provided as raw .txt files on an unstructured mesh for training of the PINNs, which are not provided due to large file sizes but can be reproduced using the conditions described in the article.

The folder 'Models\' contains a varied collection of sample models that were generated in the study, which can be queried to produce reconstructed fields. To do this, write a program that creates an empty Pytorch model of the same architecture as those in the 'Code\' folder, then use the DeepXDE model.restore(path_to_model) function to copy the weights from a pre-trained model, followed by model.predict((x, y)) to produce the reconstructed fields at the coordinates of choice. Note: some models require different architectures or domain sizes. See article for more information.
