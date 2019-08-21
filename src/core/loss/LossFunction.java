/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.loss;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Matrix;

import java.io.Serializable;
import java.util.HashMap;

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
     * Alpha value for Huber loss.
     *
     */
    private double huber_delta = 1;

    /**
     * Margin value for Hinge loss.
     *
     */
    private double hinge_margin = 1;

    /**
     * Constructor for loss function.
     *
     * @param functionType type of loss function.
     * @throws DynamicParamException throws exception if parameters are not properly given.
     */
    public LossFunction(LossFunctionType functionType) throws DynamicParamException {
        this(functionType, null);
    }

    /**
     * Constructor for loss function.
     * <br>
     * Supported parameters are:<br>
     *     - alpha: default value for Huber loss 0.5.<br>
     *
     * @param functionType type of loss function.
     * @param params parameters as DynamicParam type for activation function.
     * @throws DynamicParamException throws exception if parameters are not properly given.
     */
    public LossFunction(LossFunctionType functionType, String params) throws DynamicParamException {
        this.functionType = functionType;
        switch(functionType) {
            case MEAN_SQUARED_ERROR:
                function = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> 0.5 * Math.pow(target - pred, 2);
                derivative = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> pred - target;
                break;
            case MEAN_SQUARED_LOGARITHMIC_ERROR:
                function = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> Math.pow(Math.log(target + 1 > 0 ? target + 1 : 10E-8) - Math.log(pred + 1 > 0 ? pred + 1 : 10E-8), 2);
                derivative = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> -2 * (Math.log(target + 1 > 0 ? target + 1 : 10E-8) - Math.log(pred + 1 > 0 ? pred + 1 : 10E-8)) / (target + 1 != 0 ? target + 1 : Double.MAX_VALUE);
                break;
            case MEAN_ABSOLUTE_ERROR:
                function = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> Math.abs(target - pred);
                derivative = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> -Math.signum(target - pred);
                break;
            case MEAN_ABSOLUTE_PERCENTAGE_ERROR:
                function = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> target != 0 ? 100 * Math.abs((pred - target) / target) : Double.MAX_VALUE;
                derivative = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> 100 * (pred - target) / ((target != 0 && (pred - target) != 0) ? Math.abs(target) * Math.abs(pred - target) : Double.MAX_VALUE);
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
            case HINGE:
                if (params != null) {
                    HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
                    paramDefs.put("margin", DynamicParam.ParamType.DOUBLE);
                    DynamicParam dParams = new DynamicParam(params, paramDefs);
                    if (dParams.hasParam("margin")) hinge_margin = dParams.getValueAsDouble("margin");
                }
                function = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> hinge_margin - target * pred <= 0 ? 0 : hinge_margin - target * pred;
                derivative = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> hinge_margin - target * pred <= 0 ? 0 : - target;
                break;
            case SQUARED_HINGE:
                function = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> 1 - target * pred <= 0 ? 0 : Math.pow(1 - target * pred, 2);
                derivative = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> 1 - target * pred <= 0 ? 0 : - 2 * target * (1 - target * pred);
                break;
            case HUBER:
                if (params != null) {
                    HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
                    paramDefs.put("delta", DynamicParam.ParamType.DOUBLE);
                    DynamicParam dParams = new DynamicParam(params, paramDefs);
                    if (dParams.hasParam("delta")) huber_delta = dParams.getValueAsDouble("delta");
                }
                function = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> Math.abs(pred - target) <= huber_delta ? 0.5 * Math.pow(pred - target, 2) : huber_delta * Math.abs(pred - target) - 0.5 * Math.pow(huber_delta, 2);
                derivative = (Matrix.MatrixBiOperation & Serializable) (target, pred) -> Math.abs(pred - target) <= huber_delta ? pred - target : huber_delta * Math.signum(pred - target);
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

