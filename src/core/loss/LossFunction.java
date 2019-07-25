/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.loss;

import utils.Matrix;

import java.io.Serializable;

/**
 * Defines class for loss function of neural network.<br>
 * <br>
 * Reference: https://isaacchanghau.github.io/post/loss_functions/<br>
 *
 */
public class LossFunction implements Serializable {

    private static final long serialVersionUID = 6218297482907539129L;

    /**
     * Reference for loss function.
     *
     */
    private Matrix.MatrixBiOperation function;

    /**
     * Reference for loss function derivative.
     *
     */
    private Matrix.MatrixBiOperation derivative;

    /**
     * Type of loss function.
     *
     */
    private LossFunctionType functionType;

    /**
     * Constructor for loss function.
     *
     * @param functionType type of loss function.
     */
    public LossFunction(LossFunctionType functionType) {
        this.functionType = functionType;
        switch(functionType) {
            case MEAN_SQUARED_ERROR:
                function = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> 0.5 * Math.pow(pred - target, 2);
                derivative = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> (pred - target);
                break;
            case MEAN_ABSOLUTE_ERROR:
                function = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> Math.abs(pred - target);
                derivative = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> Math.signum(pred - target);
                break;
            case CROSS_ENTROPY:
                function = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> -(target * Math.log(pred > 0 ? pred : 10E-8));
                derivative = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> -(target / (pred > 0 ? pred : 10E-8));
                break;
            case KULLBACK_LEIBLER:
                function = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> (target * Math.log(target > 0 ? target : 10E-8) - target * Math.log((pred > 0 ? pred : 10E-8)));
                derivative = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> -(target / (pred > 0 ? pred : 10E-8));
                break;
            case NEGATIVE_LOG_LIKELIHOOD:
                function = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> -Math.log((pred > 0 ? pred : 10E-8));
                derivative = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> -1 / (pred > 0 ? pred : 10E-8);
                break;
            case POISSON:
                function = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> pred - target * Math.log((pred > 0 ? pred : 10E-8));
                derivative = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> 1 - target / (pred > 0 ? pred : 10E-8);
                break;
            default:
                break;
        }
    }

    /**
     * Gets loss function.
     *
     * @return loss function.
     */
    public Matrix.MatrixBiOperation getFunction() {
        return function;
    }

    /**
     * Gets loss function derivative.
     *
     * @return loss function derivative.
     */
    public Matrix.MatrixBiOperation getDerivative() {
        return derivative;
    }

    /**
     * Gets loss function type.
     *
     * @return loss function type.
     */
    public LossFunctionType getType() {
        return functionType;
    }

}

