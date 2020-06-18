# NeuralSpikeClassifier
*Submission for the "Neural Spike Sorting" assignment as part of EE40098: Computational Intelligence module.*

A multi-layer perceptron (MLP) neural network was implemented to classify various neural spikes (extracellular action potentials) recorded using a simple bipolar electrode inserted within the cortical region of the brain. There are 4 different types of spike (Type 1, 2, 3 & 4), each with a subtly different morphology.

Signal processing techniques are firstly employed to smooth the signal, and detect the various spikes.

The neural network is trained using the training.mat file (1440000 samples @ 25kHz sampling frequency). This dataset was divided into training and validation subsets via a 90%/10% split. Within the validation subset, approximately 88% of spikes were classified correctly.

The trained neural network is then run against the spikes within the *submission.mat* file. The first 25 were identified by hand, as the actual 'type' of each was unknown. The neural network classified 92% of these correctly.

![System Flow Diagram](/doc/sysDiagram.png?raw=true "System Flow Diagram")

