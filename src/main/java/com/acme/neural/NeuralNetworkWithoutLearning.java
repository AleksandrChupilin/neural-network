package com.acme.neural;

import java.util.Arrays;

/**
 * First implementation without learning.
 *
 * @author Aleksandr Chupilin
 * @version 1.0 2020-02-02
 *
 */
public class NeuralNetworkWithoutLearning {

    /**
     * The weights of the edge between the nodes of the input and hidden layers.
     */
    private static final double[][] WEIGHTS_INPUT_TO_HIDDEN =
        {
                { 0.25, 0.25, 0.0 }, // hidden 1
                { 0.5, -0.4,  0.9 }  // hidden 2
        };

    /**
     * The weights of the edge between the nodes of the hidden and output layers.
     */
    private static final double[] WEIGHTS_HIDDEN_TO_OUTPUT = { -1.0, 1.0 };

    /**
     * Array of input signals.
     */
    private double[] inputs;

    public double[] getInputs() {
        return inputs;
    }

    public void setInputs(double[] inputs) {
        this.inputs = inputs;
    }

    /**
     * Constructor.
     *
     * @param input array of input conditions
     */
    public NeuralNetworkWithoutLearning (final double[] input) {
        this.inputs = input;
    }

    /**
     * Activation function.
     *
     * @param input for function
     * @return calculated output
     */
    private double activationFunction(final double input) {
        return input < 0.5 ? 0 : 1;
    }

    private double[] applyActivationFunctionTo(final double[] input) {
        final double[] result = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            result[i] = activationFunction(input[i]);
        }
        return result;
    }

    private double[] multiplyMatrixByVector(final double[][] matrix, final double[] vector) {
        double[] result = new double[matrix.length];
        for (int row = 0; row < result.length; row++) {
            result[row] = multiplyMatrixByVector(matrix[row], vector);
        }
        return result;
    }

    private double multiplyMatrixByVector(final double[] matrix, final double[] vector) {
        double cell = 0;
        for (int col = 0; col < vector.length; col++) {
            cell += matrix[col] * vector[col];
        }
        return cell;
    }

    /**
     *
     * @return prediction bases on some inputs
     */
    public boolean predict() {
        final double[] inputsOfHidden = multiplyMatrixByVector(WEIGHTS_INPUT_TO_HIDDEN, inputs);
        System.out.println("Inputs of hiden layer:");
        System.out.println(Arrays.toString(inputsOfHidden));

        final double[] outputsOfHidden = applyActivationFunctionTo(inputsOfHidden);
        System.out.println("Outputs of hiden layer:");
        System.out.println(Arrays.toString(outputsOfHidden));

        final double output = multiplyMatrixByVector(WEIGHTS_HIDDEN_TO_OUTPUT, outputsOfHidden);
        System.out.println("Output:");
        System.out.println(output);

        return activationFunction(output) == 1;
    }

    /**
     * Unit test.
     *
     * @param strings
     */
    public static void main(final String... strings) {
        double[] inputs = { 0.0, 1.0, 0.0 };
        NeuralNetworkWithoutLearning nnwl = new NeuralNetworkWithoutLearning(inputs);
        System.out.println(nnwl.predict());
    }
}
