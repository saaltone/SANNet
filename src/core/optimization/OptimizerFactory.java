/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.optimization;

import core.NeuralNetworkException;
import utils.DynamicParamException;

/**
 * Defines factory class to construct optimizers.
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
        switch (optimizationType) {
            case GRADIENT_DESCENT:
                return (params == null) ? new GradientDescent(optimizationType) : new GradientDescent(optimizationType, params);
            case MOMENTUM_GRADIENT_DESCENT:
                return (params == null) ? new MomentumGradientDescent(optimizationType) : new MomentumGradientDescent(optimizationType, params);
            case NESTEROV_ACCELERATED_GRADIENT:
                return (params == null) ? new NesterovAcceleratedGradient(optimizationType) : new NesterovAcceleratedGradient(optimizationType, params);
            case ADAGRAD:
                return (params == null) ? new Adagrad(optimizationType) : new Adagrad(optimizationType, params);
            case ADADELTA:
                return (params == null) ? new Adadelta(optimizationType) : new Adadelta(optimizationType, params);
            case RMSPROP:
                return (params == null) ? new RMSProp(optimizationType) : new RMSProp(optimizationType, params);
            case ADAM:
                return (params == null) ? new Adam(optimizationType) : new Adam(optimizationType, params);
            case ADAMAX:
                return (params == null) ? new Adamax(optimizationType) : new Adamax(optimizationType, params);
            case NADAM:
                return (params == null) ? new NAdam(optimizationType) : new NAdam(optimizationType, params);
            case RADAM:
                return (params == null) ? new RAdam(optimizationType) : new RAdam(optimizationType, params);
            case AMSGRAD:
                return (params == null) ? new AMSGrad(optimizationType) : new AMSGrad(optimizationType, params);
            case RESILIENT_PROPAGATION:
                return new ResilientPropagation(optimizationType);
        }
        return null;
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

