package com.acme.neural;

import java.util.Random;

/**
 * The Class contains useful utilities for working with arrays.
 *
 * @author Aleksandr Chupilin
 * @version 2019-10-02
 */
public class NeuralUtils {

    /**
     * prevent initialization
     */
    private NeuralUtils() {
    }

    /**
     * Return Sigmoid function of specified value. The function is used as activation function.
     *
     * @param x
     * @return sigmoid of x
     */
    public static double getSigmoid(double x) {
        return 1 / (1 + Math.pow(Math.E, -x));
    }

    /**
     * Apply sigmoid function to each item of the array.
     *
     * @param arr the array of doubles
     * @return the array with modified values
     */
    public static double[] applySigmoidTo(final double[] arr) {
        final double[] result = new double[arr.length];
        for (int i = 0; i < arr.length; i++) {
            result[i] = getSigmoid(arr[i]);
        }
        return result;
    }

    public static double[] applyDerivativeOfSigmoidTo(final double[] arr) {
        final double[] result = new double[arr.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = arr[i] * (1 - arr[i]);
        }
        return result;
    }

    /**
     * Get vector with random doubles values of the specified range.
     *
     * @param  size   size of vector
     * @param  origin beginning of range (inclusive)
     * @param  bound  end of range (exclusive)
     * @return randomly filled vector of doubles
     */
    public static double[] getRandomlyFilledVector(final int size, final double origin, final double bound) {
        return new Random().doubles(size, origin, bound).toArray();
    }

    /**
     * Get matrix with random doubles values of the specified range.
     *
     * @param  row    numbers of rows
     * @param  col    numbers of columns
     * @param  origin beginning of range (inclusive)
     * @param  bound  end of range (exclusive)
     * @return randomly filled matrix of doubles
     */
    public static double[][] getRandomlyFilledMatrix(final int row, final int col, final double origin, final double bound) {
        double[][] result = new double[row][col];
        for (int i = 0; i < row; i++) {
            result[i] = getRandomlyFilledVector(col, origin, bound);
        }
        return result;
    }

    /**
     *
     * @param matrix
     * @param vector
     * @return
     */
    public static double[] multiplyMatrixByVector(final double[][] matrix, final double[] vector) {
        double[] result = new double[matrix.length];
        for (int row = 0; row < result.length; row++) {
            result[row] = multiplyVectors(matrix[row], vector);
        }
        return result;
    }

    /**
     * Multiply two vectors
     *
     * @param v1 the first vector
     * @param v2 the second vector
     * @return result of multiplication
     */
    public static double multiplyVectors(final double[] v1, final double[] v2) {
        double cell = 0;
        for (int col = 0; col < v2.length; col++) {
            cell += v1[col] * v2[col];
        }
        return cell;
    }

    /**
     * Multiply each element of vector by number
     *
     * @param vector
     * @param number
     * @return
     */
    public static double[] multiplyVectorByNumber(final double[] vector, double number) {
        double[] result = new double[vector.length];
        for (int col = 0; col < result.length; col++) {
            result[col] = vector[col] * number;
        }
        return result;
    }

    public static double[] multiplyElementsOfMatrix(final double[] v1, final double[] v2) {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException("size of arrays are not equals");
        }
        double[] result = new double[v1.length];
        for (int i = 0; i < v1.length; i++) {
            result[i] = v1[i] * v2[i];
        }

        return result;
    }

    public static void correct(double[] weights, double[] outputs, double corr, double rate) {
        if (weights.length != outputs.length) {
            throw new IllegalArgumentException("size of arrays are not equals: weight =" +
                    weights.length + " outputs =  " + outputs.length);
        }
        for (int i = 0; i < weights.length; i++) {
            weights[i] -= outputs[i] * corr * rate;
        }
    }

    public static void correct(double[][] weights, double[] outputs, double[] corr, double rate) {
        if (weights[0].length != outputs.length) {
            throw new IllegalArgumentException("size of arrays are not equals: weight =" +
                    weights.length + " outputs =  " + outputs.length);
        }
        for (int row = 0; row < weights.length; row++) {
            correct(weights[row], outputs, corr[row], rate);
            // weights[i] -= outputs[i] * corr[i] * rate;
        }
    }

}
