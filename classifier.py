import scipy.io as spio
import scipy.special as scsp
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from collections import Counter


class NeuralNetwork:
    # Init the network, this gets run whenever we make a new instance of this class
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set the number of nodes in each input, hidden and output layer
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        # Weight matrices, wih (input -> hidden) and who (hidden -> output)
        self.wih = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = np.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        # Set the learning rate
        self.lr = learning_rate

        # Set the activation function, the logistic sigmoid
        self.activation_function = lambda x: scsp.expit(x)

    # Train the network using back-propagation of errors
    def train(self, inputs_list, targets_list):
        # Convert inputs into 2D arrays
        inputs_array = np.array(inputs_list, ndmin=2).T
        targets_array = np.array(targets_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs_array)

        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # Current error is (target - actual)
        output_errors = targets_array - final_outputs

        # Hidden layer errors are the output errors, split by the weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))

        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                     np.transpose(inputs_array))

    # Query the network
    def query(self, inputs_list):
        # Convert the inputs list into a 2D array
        inputs_array = np.array(inputs_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs_array)

        # Calculate output from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # Calculate outputs from the final layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


def sort_data(train_location, train_class):
    """Sort spike locations into ascending order, keeping each spikes class with the spike."""

    # Concatenate spike class and index, so they can be sorted by index
    spikes = np.zeros(shape=(len(train_location), 2))
    spikes[:, 0] = train_location
    spikes[:, 1] = train_class

    # Sort indices into ascending order, keeping corresponding classes together
    spikes = spikes[spikes[:, 0].argsort()]

    # Extract training spike locations and classes from sorted data
    train_location = spikes[:, 0]
    train_class = spikes[:, 1]

    return train_location, train_class


def butterworth_filter(data, sample_freq, low=30, high=1500, order=4):
    """Apply Butterworth band-pass filter to input data."""

    # Determine Nyquist frequency
    nyq = sample_freq / 2

    # Set lower and upper cut-off frequencies
    low = low / nyq
    high = high / nyq

    # Calculate digital butterworth band-pass filter coefficients
    b, a = butter(order, [low, high], btype='band', output='ba')

    # Apply filter backwards & forwards
    filtered_data = filtfilt(b, a, data)

    return filtered_data


def apply_filters(data, sampling_freq):
    """Apply filters to input data."""

    # Apply Butterworth band-pass filter
    data_filtered = butterworth_filter(data, sampling_freq)

    # Apply Savitzky-Golay filter
    data_filtered = savgol_filter(data_filtered, 25, 6)

    return data_filtered


def spike_detector(data):
    """Detect spikes within input data."""

    # Determine threshold by finding the standard deviation of the input data
    threshold = 3 * np.std(data)

    # Find spikes using scipy's find_peaks to find spikes within the data
    peak_locations, _ = find_peaks(data, height=threshold)

    spikes_detected = np.zeros(len(peak_locations))  # Initialise output array

    # For each detected peak...
    for i in range(0, len(peak_locations)):

        # Determine the gradient of points 15 values before the peak (15 gives the best result)
        diff = np.diff(data[peak_locations[i] - 15:peak_locations[i]])

        # Find the location of the highest gradient value within 'diff' array
        diff_max = (np.argmax(diff)) - 1

        # Set the spike location
        spikes_detected[i] = peak_locations[i] - 15 + diff_max

    return spikes_detected


def split_data(train_data, train_location, train_class, detected_location, percentage=0.90):
    """Split data into training and validation set."""

    # Find index that splits the data at given percentage (90% by default)
    split_data_index = int(len(train_data) * percentage)

    # Define the data values within validation set
    valid_values = train_location[train_location > split_data_index]

    # Find locations of spikes within remaining percentage of data (10% by default)
    split_valid_index = int(np.where(train_location == valid_values[0])[0])

    # Find locations and classes of spikes within the validation set
    validation_location = train_location[split_valid_index:len(train_location)]
    validation_class = train_class[split_valid_index:len(train_class)]

    # Find location of spikes within given percentage of data (90% by default)
    train_location = train_location[0:split_valid_index]

    # Find locations of spikes detected within the validation set
    detected_location = detected_location[detected_location > split_data_index]

    return train_location, validation_location, validation_class, detected_location


def compare_spikes(detected_spikes, known_spikes, known_classes):
    """Given the locations of the known spikes within the validation data, determine the percentage
    of spikes correctly detected by the spike_detector function."""

    # Initialise allowance value and matrix to store the correctly detected spikes
    allowance = 4
    correct_spikes = np.zeros(shape=(len(detected_spikes), 2))

    i = 0
    for spike in detected_spikes:

        # Define upper and lower limits
        lower_limit = spike - allowance
        upper_limit = spike + allowance

        # Find index of known spike within allowance range of detected spikes
        correct_index = np.where((known_spikes > lower_limit) & (known_spikes < upper_limit))
        correct_index = correct_index[0]  # Access the array within the returned tuple

        # If the spike was identified correctly, assign it to correct_spikes matrix
        if len(correct_index) != 0:
            correct_spikes[i, 0] = spike
            correct_spikes[i, 1] = known_classes[correct_index[0]]

        i += 1

    # Remove rows which contain zeros (i.e. not identified)
    correct_spikes = correct_spikes[correct_spikes[:, 0] != 0]

    print("Number of spikes correctly identified: %d" % len(correct_spikes))

    # Determine percentage of spikes correctly identified
    correct_percentage = (len(correct_spikes) / len(known_spikes)) * 100
    print("Percentage of spikes correctly identified: %f %%" % correct_percentage)

    return correct_spikes, correct_percentage


def spike_data_matrix(data, spike_location, window_size=50):
    """Create sample of data points to define each spike, in preparation for neural network use.
    Organised into a matrix."""

    # Initialise window
    data_matrix = np.zeros((len(spike_location), window_size))

    # Fill window with data
    for i in range(0, len(spike_location)):
        data_matrix[i, :] = data[int(spike_location[i]):int(spike_location[i]) + window_size]

    # Determine max value within the window
    max_value = np.amax(data_matrix)

    return data_matrix, max_value


def train_network(neural_network, train_location, train_class, training_window, training_max):
    """Train the neural network on the input data."""

    # Train the neural network on each training sample
    for i in range(0, len(train_location)):
        # Normalise training input data
        training_inputs = (np.asfarray(training_window[i, :]) / training_max * 0.99) + 0.01

        # Define the correct/target label (all 0.01, except the desired label which is 0.99)
        training_targets = np.zeros(neural_network.o_nodes) + 0.01

        # Assign target label for this data entry
        training_targets[int(train_class[i]) - 1] = 0.99

        # Train the network
        neural_network.train(training_inputs, training_targets)


def test_network(neural_network, input_window, input_max, test_location=None, test_class=None, submission=False):
    """Test the performance of the neural network classifier against known spikes classes."""

    if submission:
        # If submission flag, define the indices of submission data spikes and their classes
        # These are manually identified spikes & classes
        test_location = list(range(0, 25))
        test_class = [1, 2, 4, 2, 1, 1, 1, 3, 2, 2, 2, 1, 3, 1, 2, 3, 4, 1, 4, 4, 3, 4, 2, 1, 4]
        test_matrix = input_window[0:25, :]

    else:
        # Else... performance of classifier on validation data set is being tested
        # test_index and test_class should be passed as arguments if this is the case
        test_matrix = input_window[0:len(test_location), :]

    scorecard = []  # Initialise scorecard list to determine performance of network
    output_classes = []  # Initialise output_classes list to store correctly identified spikes

    # Loop through the entries within the test data set
    for i in range(0, len(test_location)):

        # Extract the correct class label
        correct_label = test_class[i]

        # Normalise data to be input to neural network
        test_inputs = (np.asfarray(test_matrix[i, :]) / input_max * 0.99) + 0.01

        # Query the network
        test_outputs = neural_network.query(test_inputs)

        # The index of the highest value output corresponds to the label
        output_label = np.argmax(test_outputs)

        # Shift the label by 1 (due to Python's zero indexing)
        output_label = output_label + 1

        # Append a 1 to the scorecard list if correct, along with correct class to output_classes
        if output_label == correct_label:
            scorecard.append(1)
            output_classes.append(output_label)

        # Append a 0 to the scorecard list if not classified correctly
        else:
            scorecard.append(0)
            pass
        pass

    # Identify quantity of each class detected using Counter (collections library)
    num_of_classes = Counter(output_classes)

    # Convert scorecard to numpy array, calculate percentage of spike correctly identified
    scorecard_array = np.asarray(scorecard)
    performance_percentage = (scorecard_array.sum() / scorecard_array.size) * 100

    return performance_percentage, num_of_classes


def create_submission_file(neural_network, submission_window, submission_max, detected_spikes):
    """Run neural network on entirety of submission data, create MATLAB file containing identified spike
    and their classes."""

    # Initialise submission class list to store classes of spikes identified
    detected_classes = []

    # Loop through the spikes detected within the submission data set
    for i in range(0, len(detected_spikes)):

        # Normalise data to be input to neural network
        inputs = (np.asfarray(submission_window[i, :]) / submission_max * 0.99) + 0.01

        # Query the network
        outputs = neural_network.query(inputs)

        # The index of the highest value output corresponds to the label
        label = np.argmax(outputs)

        # Shift the label by 1 (due to Python's zero indexing)
        label = label + 1

        # Append the output label onto the submission_class list
        detected_classes.append(label)

    print("Number of spikes detected within submission data: {}".format(len(detected_spikes)))

    # Create a .mat file containing the detected spikes and their classes
    final_matlab_file = {'Index': detected_spikes, 'Class': detected_classes}
    spio.savemat('12847.mat', final_matlab_file)

    # Tally up quantity of each class using Counter (from collections library)
    quantity_classes = Counter(detected_classes)

    print("Quantity of each class detected within submission data:" +
          "\n Type 1: {}\n Type 2: {}\n Type 3: {}\n Type 4: {}\n".format(quantity_classes[1], quantity_classes[2],
                                                                          quantity_classes[3], quantity_classes[4]))


if __name__ == '__main__':

    # Initialise neural network
    neuralNet = NeuralNetwork(input_nodes=50, hidden_nodes=125, output_nodes=4, learning_rate=0.25)

    # Import MATLAB training data
    matTraining = spio.loadmat('training.mat', squeeze_me=True)
    trainData = matTraining['d']
    trainIndex = matTraining['Index']
    trainClass = matTraining['Class']

    # Import MATLAB submission data
    matSubmission = spio.loadmat('submission.mat', squeeze_me=True)
    submissionData = matSubmission['d']

    # Define sampling frequency
    samplingFreq = 25 * (10 ** 3)

    # Sort data
    trainIndex, trainClass = sort_data(trainIndex, trainClass)

    # Apply filters to training & submission data
    trainData = apply_filters(trainData, samplingFreq)
    submissionData = apply_filters(submissionData, samplingFreq)

    # Detect spikes within training & submission data
    trainingSpikes = spike_detector(trainData)
    submissionSpikes = spike_detector(submissionData)

    # Split data into training and validation set
    trainIndex, validationIndex, validationClass, detectedValidationSpikes = split_data(trainData, trainIndex,
                                                                                        trainClass, trainingSpikes)

    print("No. of validation spikes found: %d" % len(detectedValidationSpikes))
    print("Actual no. of validation spikes: %d \n" % len(validationIndex))

    # Determine no. of spikes within validation set that were identified correctly
    correctSpikes, detectionPercentage = compare_spikes(detectedValidationSpikes, validationIndex, validationClass)
    correctValidSpikes = correctSpikes[:, 0]
    correctValidClasses = correctSpikes[:, 1]

    # Create matrices containing data points of each spike, for training, validation and submission data
    trainingMatrix, trainingMax = spike_data_matrix(trainData, trainIndex)
    validationMatrix, validationMax = spike_data_matrix(trainData, correctValidSpikes)
    submissionMatrix, submissionMax = spike_data_matrix(submissionData, submissionSpikes)

    # Train the neural network
    print("\nTraining...")
    for _ in range(5):
        train_network(neuralNet, trainIndex, trainClass, trainingMatrix, trainingMax)
    print("Training complete! \n")

    # Test the performance of the network against validation set
    validationPercentage, NoC = test_network(neuralNet, validationMatrix, validationMax, correctValidSpikes,
                                             correctValidClasses)

    # Take into account percentage of spikes not detected correctly
    print("Percentage of correctly identified spikes that are classified correctly: %f %%" % validationPercentage)
    validationPercentage = validationPercentage * (detectionPercentage / 100)
    print("Network performance on validation data: %f %%" % validationPercentage)
    print("Quantity of each class detected correctly detected within validation set:" +
          "\n Type 1: {}\n Type 2: {}\n Type 3: {}\n Type 4: {}\n".format(NoC[1], NoC[2], NoC[3], NoC[4]))

    # Test the performance of the network against snippet of submission data (first 25 spikes)
    submissionPercentage, NoC = test_network(neuralNet, submissionMatrix, submissionMax, submission=True)
    print("Estimated network performance on submission data: %f %%" % submissionPercentage)
    print("Quantity of each class detected correctly detected within snippet of submission set:" +
          "\n Type 1: {}\n Type 2: {}\n Type 3: {}\n Type 4: {}\n".format(NoC[1], NoC[2], NoC[3], NoC[4]))

    # Generate MATLAB submission file by running classifier for entire submission data set
    create_submission_file(neuralNet, submissionMatrix, submissionMax, submissionSpikes)
