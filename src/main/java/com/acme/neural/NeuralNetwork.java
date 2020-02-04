package com.acme.neural;

import java.util.Arrays;

public class NeuralNetwork {

    /**
     *
     */
    private double learningRate;

    /**
     * The weights of the edge between the nodes of the input and hidden layers.
     */
    private double[][] weightsInputToHidden;

    /**
     * The weights of the edge between the nodes of the hidden and output layers.
     */
    private double[] weightsHiddenToOutput;

    public NeuralNetwork() {
        weightsInputToHidden = NeuralUtils.getRandomlyFilledMatrix(2, 3, 0, 1);
        System.out.println(Arrays.toString(weightsInputToHidden[0]));
        System.out.println(Arrays.toString(weightsInputToHidden[1]));
        weightsHiddenToOutput = NeuralUtils.getRandomlyFilledVector(2, 0, 1);
        System.out.println(Arrays.toString(weightsHiddenToOutput));
    }

    public double predict(double[] inputs) {
        final double[] inputsOfHidden = NeuralUtils.multiplyMatrixByVector(weightsInputToHidden, inputs);
        // System.out.println("Inputs of hiden layer:");
        //  System.out.println(Arrays.toString(inputsOfHidden));

        final double[] outputsOfHidden = NeuralUtils.applySigmoidTo(inputsOfHidden);
        //  System.out.println("Outputs of hiden layer:");
        //  System.out.println(Arrays.toString(outputsOfHidden));

        final double output = NeuralUtils.multiplyVectors(weightsHiddenToOutput, outputsOfHidden);
        //  System.out.println("Output:");
        //  System.out.println(output);

        return NeuralUtils.getSigmoid(output);
    }

    public void train(final double[] inputs, final double expectation) {
        final double[] inputsOfHidden = NeuralUtils.multiplyMatrixByVector(weightsInputToHidden, inputs);
        final double[] outputsOfHidden = NeuralUtils.applySigmoidTo(inputsOfHidden);
        final double output = NeuralUtils.multiplyVectors(weightsHiddenToOutput, outputsOfHidden);
        double prediction = NeuralUtils.getSigmoid(output);

        double errorOutout = prediction - expectation;
        double derivativeOutput = prediction * (1 - prediction);
        double correctionFactorWeightsHiddenToOutput = errorOutout * derivativeOutput;
        NeuralUtils.correct(weightsHiddenToOutput, outputsOfHidden, correctionFactorWeightsHiddenToOutput, learningRate);

        double[] errorHidden = NeuralUtils.multiplyVectorByNumber(weightsHiddenToOutput, correctionFactorWeightsHiddenToOutput);
        double[] derivativeHidden = NeuralUtils.applyDerivativeOfSigmoidTo(outputsOfHidden);
        double[] correctionFactorsWeightsInputToHidden = NeuralUtils.multiplyElementsOfMatrix(errorHidden, derivativeHidden);
        NeuralUtils.correct(weightsInputToHidden, inputs, correctionFactorsWeightsInputToHidden, learningRate);
    }

    public double mse(double p, double a) {
        return Math.pow((p - a), 2);
    }

    /**
     * getter
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * Setter
     *
     * @param learningRate
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }



    public double[][] getWeightsInputToHidden() {
        return weightsInputToHidden;
    }

    public void setWeightsInputToHidden(double[][] weightsInputToHidden) {
        this.weightsInputToHidden = weightsInputToHidden;
    }

    public double[] getWeightsHiddenToOutput() {
        return weightsHiddenToOutput;
    }

    public void setWeightsHiddenToOutput(double[] weightsHiddenToOutput) {
        this.weightsHiddenToOutput = weightsHiddenToOutput;
    }

    public static void main(final String[] args) throws InterruptedException {
        double[][] inputs = {
                { 0, 0, 0 },
                { 0, 0, 1 },
                { 0, 1, 0 },
                { 0, 1, 1 },
                { 1, 0, 0 },
                { 1, 0, 1 },
                { 1, 1, 0 },
                { 1, 1, 1 }
        };

        double[] expectation = {0, 1, 0, 0, 1, 1, 0, 0};
        int epocha = 16000;
        double rate = 0.07;

        NeuralNetwork nn = new NeuralNetwork();
        nn.setLearningRate(rate);

        double mse = 0;
        for (int i = 0; i < epocha; i++) {
            for (int j = 0; j < inputs.length; j++) {
                nn.train(inputs[j], expectation[j]);

                if (i == (epocha - 1)) {
                    double currentPrediction = nn.predict(inputs[j]);
                    mse += nn.mse(currentPrediction, expectation[j]);
                }
            }
        }

        System.out.println("MSE: " + mse / 8);
        System.out.println();
        for (int j = 0; j < inputs.length; j++) {
            double prediction = nn.predict(inputs[j]);
            System.out.println("prediction: " + prediction + " | expectation: " + expectation[j]);
        }
    }

}
