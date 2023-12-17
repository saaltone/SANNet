/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.normalization;

import core.layer.AbstractExecutionLayer;
import core.layer.NeuralNetworkLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.procedure.Procedure;
import utils.procedure.ProcedureFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * Implements layer for weight normalization.
 *
 */
public class WeightNormalization extends AbstractExecutionLayer {

    /**
     * Parameter name types for weight normalization.
     *     - g: g multiplier value for normalization. Default value 1.<br>
     *
     */
    private final static String paramNameTypes = "(g:INT)";

    /**
     * Weight normalization scalar.
     *
     */
    private double g;

    /**
     * Matrix for g value.
     *
     */
    private Matrix gMatrix;

    /**
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

    /**
     * Procedures for weight normalization.
     *
     */
    private HashMap<Matrix, Procedure> procedures = null;

    /**
     * Template procedure.
     *
     */
    private Procedure templateProcedure;

    /**
     * Constructor for weight normalization layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for weight normalization layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public WeightNormalization(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
        super (layerIndex, initialization, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        g = 1;
        gMatrix = new DMatrix(g);
        gMatrix.setName("g");
        registerConstantMatrix(gMatrix);
        registerStopGradient(gMatrix);
    }

    /**
     * Returns parameters used for weight normalization layer.
     *
     * @return parameters used for weight normalization layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + WeightNormalization.paramNameTypes;
    }

    /**
     * Sets parameters used for weight normalization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - g: g multiplier value for normalization. Default value 1.<br>
     *
     * @param params parameters used for weight normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("g")) {
            g = params.getValueAsInteger("g");
            gMatrix.setValue(0, 0, 0, g);
        }
    }

    /**
     * Returns weight set.
     *
     * @return weight set.
     */
    protected WeightSet getWeightSet() {
        return null;
    }

    /**
     * Initializes neural network layer weights.
     *
     */
    public void initializeWeights() {
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     */
    public TreeMap<Integer, Matrix> getInputMatrices(boolean resetPreviousInput) {
        return new TreeMap<>() {{ put(0, input); }};
    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    protected void defineProcedure() throws MatrixException, DynamicParamException {
        input = new DMatrix(1, 1, 1);
        registerConstantMatrix(input);
        templateProcedure = new ProcedureFactory().getProcedure(this);
    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws MatrixException       throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void defineProcedures() throws MatrixException, DynamicParamException {
        procedures = new HashMap<>();
        for (NeuralNetworkLayer nextLayer : getNextLayers().values()) {
            for (Matrix normalizedWeight : nextLayer.getNormalizedWeights()) {
                input = normalizedWeight;
                Procedure procedure = new ProcedureFactory().getProcedure(this);
                procedures.put(normalizedWeight, procedure);
            }
        }
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getForwardProcedure() throws MatrixException {
        Matrix output = input.multiply(gMatrix).divide(input.normAsMatrix(2));
        output.setName("Output");
        return output;
    }

    /**
     * Takes single forward processing step to process layer input(s).<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void forwardProcess() throws MatrixException, DynamicParamException {
        if (procedures == null) defineProcedures();

        passLayerOutputs();

        if (isTraining()) {
            for (Map.Entry<Matrix, Procedure> entry : procedures.entrySet()) {
                Matrix weight = entry.getKey();
                Procedure procedure = entry.getValue();
                procedure.reset();
                weight.setEqualTo(procedure.calculateExpression(weight));
            }
        }
    }

    /**
     * Takes single backward processing step to process layer output gradient(s) towards input.<br>
     * Applies automated backward (automatic gradient) procedure when relevant to layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backwardProcess() throws MatrixException {
        passLayerOutputGradients();
    }

    /**
     * Executes weight updates with optimizer.
     *
     */
    public void optimize() {
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return "g value: " + g;
    }

    /**
     * Prints expression chains of normalization.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void printExpressions() throws NeuralNetworkException {
        System.out.println(getLayerName() + ": ");
        templateProcedure.printExpressionChain();
        System.out.println();
    }

    /**
     * Prints gradient chains of normalization.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void printGradients() throws NeuralNetworkException {
        System.out.println(getLayerName() + ": ");
        templateProcedure.printGradientChain();
        System.out.println();
    }


}
