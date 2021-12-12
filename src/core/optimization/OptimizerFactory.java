/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.optimization;

import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;

/**
 * Defines factory class to construct optimizers.<br>
 *
 */
public class OptimizerFactory {

    /**
     * Constructs optimizer.
     *
     * @param optimizationType optimizer type.
     * @param params parameters for specific optimizer.
     * @return constructed optimizer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public static Optimizer create(OptimizationType optimizationType, String params) throws DynamicParamException {
        return switch (optimizationType) {
            case GRADIENT_DESCENT -> (params == null) ? new GradientDescent() : new GradientDescent(params);
            case MOMENTUM_GRADIENT_DESCENT -> (params == null) ? new MomentumGradientDescent() : new MomentumGradientDescent(params);
            case NESTEROV_ACCELERATED_GRADIENT -> (params == null) ? new NesterovAcceleratedGradient() : new NesterovAcceleratedGradient(params);
            case ADAGRAD -> (params == null) ? new Adagrad() : new Adagrad(params);
            case ADADELTA -> (params == null) ? new Adadelta() : new Adadelta(params);
            case RMSPROP -> (params == null) ? new RMSProp() : new RMSProp(params);
            case ADAM -> (params == null) ? new Adam() : new Adam(params);
            case ADAMAX -> (params == null) ? new Adamax() : new Adamax(params);
            case NADAM -> (params == null) ? new NAdam() : new NAdam(params);
            case RADAM -> (params == null) ? new RAdam() : new RAdam(params);
            case AMSGRAD -> (params == null) ? new AMSGrad() : new AMSGrad(params);
            case RESILIENT_PROPAGATION -> new ResilientPropagation();
        };
    }

    /**
     * Constructs optimizer.
     *
     * @param optimizationName optimizer name.
     * @param params parameters for specific optimizer.
     * @return constructed optimizer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public static Optimizer create(String optimizationName, String params) throws DynamicParamException {
        return switch (optimizationName) {
            case "GradientDescent" -> (params == null) ? new GradientDescent() : new GradientDescent(params);
            case "MomentumGradientDescent" -> (params == null) ? new MomentumGradientDescent() : new MomentumGradientDescent(params);
            case "NesterovAcceleratedGradient" -> (params == null) ? new NesterovAcceleratedGradient() : new NesterovAcceleratedGradient(params);
            case "Adagrad" -> (params == null) ? new Adagrad() : new Adagrad(params);
            case "Adadelta" -> (params == null) ? new Adadelta() : new Adadelta(params);
            case "RMSProp" -> (params == null) ? new RMSProp() : new RMSProp(params);
            case "Adam" -> (params == null) ? new Adam() : new Adam(params);
            case "Adamax" -> (params == null) ? new Adamax() : new Adamax(params);
            case "NAdam" -> (params == null) ? new NAdam() : new NAdam(params);
            case "RAdam" -> (params == null) ? new RAdam() : new RAdam(params);
            case "AMSGrad" -> (params == null) ? new AMSGrad() : new AMSGrad(params);
            case "ResilientPropagation" -> new ResilientPropagation();
            default -> throw new DynamicParamException("Unknown optimizer name.");
        };
    }

    /**
     * Constructs optimizer with default parameters.
     *
     * @param optimization optimizer type.
     * @return constructed optimizer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public static Optimizer create(OptimizationType optimization) throws DynamicParamException {
        return create(optimization, null);
    }

    /**
     * Constructs optimizer with default parameters.
     *
     * @param optimizer optimizer of specific type.
     * @return constructed optimizer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if optimizer is of an unknown type.
     */
    public static Optimizer create(Optimizer optimizer) throws DynamicParamException, NeuralNetworkException {
        return create(getOptimizationType(optimizer), null);
    }

    /**
     * Constructs optimizer with default parameters.
     *
     * @param optimizer optimizer of specific type.
     * @param params parameters of optimizer.
     * @return constructed optimizer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if optimizer is of an unknown type.
     */
    public static Optimizer create(Optimizer optimizer, String params) throws DynamicParamException, NeuralNetworkException {
        return create(getOptimizationType(optimizer), params);
    }

    /**
     * Constructs default optimizer (ADAM).
     *
     * @return constructed optimizer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public static Optimizer createDefault() throws DynamicParamException {
        return create(OptimizationType.ADAM);
    }

    /**
     * Returns type of optimizer.
     *
     * @param optimizer given optimizer.
     * @return type of optimizer.
     * @throws NeuralNetworkException throws exception if optimizer is of an unknown type.
     */
    public static OptimizationType getOptimizationType(Optimizer optimizer) throws NeuralNetworkException {
        if (optimizer instanceof GradientDescent) return OptimizationType.GRADIENT_DESCENT;
        if (optimizer instanceof MomentumGradientDescent) return OptimizationType.MOMENTUM_GRADIENT_DESCENT;
        if (optimizer instanceof NesterovAcceleratedGradient) return OptimizationType.NESTEROV_ACCELERATED_GRADIENT;
        if (optimizer instanceof Adagrad) return OptimizationType.ADAGRAD;
        if (optimizer instanceof Adadelta) return OptimizationType.ADADELTA;
        if (optimizer instanceof RMSProp) return OptimizationType.RMSPROP;
        if (optimizer instanceof Adam) return OptimizationType.ADAM;
        if (optimizer instanceof Adamax) return OptimizationType.ADAMAX;
        if (optimizer instanceof NAdam) return OptimizationType.NADAM;
        if (optimizer instanceof RAdam) return OptimizationType.RADAM;
        if (optimizer instanceof AMSGrad) return OptimizationType.AMSGRAD;
        if (optimizer instanceof ResilientPropagation) return OptimizationType.RESILIENT_PROPAGATION;
        throw new NeuralNetworkException("Unknown optimizer type");
    }

}

