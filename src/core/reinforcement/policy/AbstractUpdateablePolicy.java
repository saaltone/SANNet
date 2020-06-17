/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement.policy;

import core.NeuralNetworkException;
import core.reinforcement.RLSample;
import core.reinforcement.function.FunctionEstimator;
import utils.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.LinkedHashMap;
import java.util.TreeMap;

/**
 * Class that defines AbstractUpdateablePolicy. Contains common functions fo updateable policies.
 *
 */
public abstract class AbstractUpdateablePolicy extends ActionableBasicPolicy implements UpdateablePolicy {

    /**
     * Current value error;
     *
     */
    private transient double valueError;

    /**
     * Constructor for AbstractUpdateablePolicy.
     *
     * @param policy reference to policy.
     * @param functionEstimator reference to FunctionEstimator.
     */
    public AbstractUpdateablePolicy(Policy policy, FunctionEstimator functionEstimator) {
        super(policy, functionEstimator);
    }

    /**
     * Updates policy FunctionEstimator.
     *
     * @param samples samples used for policy FunctionEstimator update.
     * @param hasImportanceSamplingWeights if true samples contain importance sampling weights otherwise false.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void updateFunctionEstimator(TreeMap<Integer, RLSample> samples, boolean hasImportanceSamplingWeights) throws NeuralNetworkException, MatrixException, DynamicParamException {
        preProcess();
        LinkedHashMap<Integer, MMatrix> states = new LinkedHashMap<>();
        LinkedHashMap<Integer, MMatrix> policyGradients = new LinkedHashMap<>();
        for (Integer sampleIndex : samples.descendingKeySet()) {
            RLSample sample = samples.get(sampleIndex);
            states.put(sampleIndex, new MMatrix(sample.state.stateMatrix));
            Matrix policyGradient = new DMatrix(getFunctionEstimator().getNumberOfActions(), 1);
            policyGradient.setValue(sample.state.action, 0, -getPolicyGradientValue(sample, hasImportanceSamplingWeights));
            policyGradients.put(sampleIndex, new MMatrix(policyGradient));
        }
        postProcess();
        getFunctionEstimator().train(states, policyGradients);
    }

    /**
     * Preprocesses policy gradient setting.
     *
     */
    protected abstract void preProcess();

    /**
     * Returns policy gradient value for sample.
     *
     * @param sample sample
     * @param hasImportanceSamplingWeight if true sample has importance sample weight set.
     * @return policy gradient value for sample.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract double getPolicyGradientValue(RLSample sample, boolean hasImportanceSamplingWeight) throws NeuralNetworkException, MatrixException;

    /**
     * Postprocesses policy gradient setting.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract void postProcess() throws MatrixException;

    /**
     * Sets current value error.
     *
     * @param valueError current value error.
     */
    public void setValueError(double valueError) {
        this.valueError = valueError;
    }

    /**
     * Returns current value error.
     *
     * @return current value error.
     */
    protected double getValueError() {
        return valueError;
    }

}
