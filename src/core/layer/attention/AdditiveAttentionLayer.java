/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.attention;

import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.Map;

/**
 * Implements additive attention layer.
 *
 */
public class AdditiveAttentionLayer extends DotAttentionLayer {

    /**
     * Constructor for additive attention layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for additive attention layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public AdditiveAttentionLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getForwardProcedure() throws MatrixException {
        Matrix joinedInput = null;
        for (Map.Entry<Integer, Matrix> entry : inputs.entrySet()) {
            joinedInput = joinedInput == null ? entry.getValue() : joinedInput.join(entry.getValue(), false);
        }
        assert joinedInput != null;
        joinedInput.setName("JoinedInput");

        Matrix transposedJoinedInput = joinedInput.apply(transposeFunction);
        transposedJoinedInput.setName("TransposedInput");
        Matrix query = transposedJoinedInput.dot(weightSet.queryWeight);
        query.setName("Query");
        Matrix key = transposedJoinedInput.dot(weightSet.keyWeight);
        key.setName("Key");
        Matrix value = weightSet.valueWeight.dot(joinedInput);
        value.setName("Value");

        Matrix output = query.add(key).apply(transposeFunction).multiply(value);
        output.setName("Output");
        return output;
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return "";
    }

}
