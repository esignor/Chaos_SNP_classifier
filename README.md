# Chaos_SNP_classifier
Machine Learning techniques for the phenotypic classification of a sea basses population, expressed in the form of SNPs data, and decoded into images using the Chaos Game Representation algorithm


Software configuration:
The code developed in “Chaos_SNP_classifier” is in Python programming language version 3.9.21. 
To compile the software correctly, installation of the following python libraries is required:
pandas, matplotlib, scikit-learn, keras

It is recommended to use in binary package manager such as conda, anconda or miniconda to create a system-level environment independent of the machine's operating system.

The developed software contains:
- The Encoder Unit, which decodes the haplotype sequences of each sea bass in image through Chaos Game Representation algorithm on genomic sequences. 
- A Classifier Unit is developed, consisting of shallow CNN networks (i.e., Simple CNN and Complex CNN) and Deep Convolutional Neural Networks (i.e., AlexNet, ResNet50 and ResNet101)

To run the Encoder Unit, it is necessary to run the 'EncoderGS-AQUACULTURE' file.  It is necessary that multithread is supported by the machine.
In the Classifier Unit a model is implemented for each network tested (Simple CNN, Complex CNN, AlexNet, ResNet50, and ResNet101). To perform learning and related classification on a specific network, it is necessary to run the model of the network it refers to (e.g., to classify with AlexNet run the AlexNetGS_classifier-AQUACULTURE model).
In each model, it is possible to set the number of epochs and batch size on which to set up neural network learning on CGR images e.g., batch_size=15; epoch=150.

Work in progress:
In Deep Convolutional Neural Networks there are extensions for learning on grayscale networks (with GS flag) and the possibility of explaining classification using the SHAP library for image classification. In addition, the algorithm for encoding haplotype sequences on 3-point CGR is implemented.

