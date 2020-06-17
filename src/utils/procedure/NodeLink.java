/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.procedure;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serializable;

/**
 * Class that defines expression and gradient linkage between output (result) and input (arg) node.
 *
 */
public class NodeLink implements Serializable {

    private static final long serialVersionUID = -472658584018963912L;

    /**
     * From (output / result) node.
     *
     */
    private final Node fromNode;

    /**
     * To (input / arg) node.
     *
     */
    private final Node toNode;

    /**
     * Previous matrix.
     *
     */
    private transient Matrix previousMatrix = null;

    /**
     * Previous gradient.
     *
     */
    private transient Matrix previousGradient = null;

    /**
     * Constructor for node link.
     *
     * @param fromNode from (output / result) node.
     * @param toNode to (input / arg) node.
     */
    NodeLink(Node fromNode, Node toNode) {
        this.fromNode = fromNode;
        this.toNode = toNode;
    }

    /**
     * Resets previous matrix and gradient.
     *
     */
    public void reset() {
        previousMatrix = null;
        previousGradient = null;
    }

    /**
     * Returns from node.
     *
     * @return from node.
     */
    public Node getFromNode() {
        return fromNode;
    }

    /**
     * Returns to node.
     *
     * @return to node.
     */
    public Node getToNode() {
        return toNode;
    }

    /**
     * Updates expression from result (output) to arg (input) node.
     *
     * @param index data index.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void updateExpression(int index) throws MatrixException {
        Matrix fromMatrix = fromNode.getMatrix(index - 1);
        toNode.setMatrix(index, fromMatrix != null ? fromMatrix : previousMatrix != null ? previousMatrix : fromNode.getEmptyMatrix());
        previousMatrix = fromMatrix;
    }

    /**
     * Updates gradient from arg (input) to result (output) node.
     *
     * @param index data index.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void updateGradient(int index) throws MatrixException {
        Matrix fromGradient = toNode.getGradient(index + 1);
        fromNode.updateGradient(index, fromGradient != null ? fromGradient : previousGradient != null ? previousGradient : toNode.getEmptyMatrix(), true);
        previousGradient = fromGradient;
    }

    /**
     * Provides copy of linking node.
     *
     * @param copyGradients if true copy of node contains also gradients.
     * @throws MatrixException throws exception is matrix is not defined.
     * @return copy of node.
     */
    public Node copy(boolean copyGradients) throws MatrixException {
        return fromNode.copy(copyGradients);
    }

    /**
     * Copies data from given node.
     *
     * @param node source node.
     * @param copyGradients if true copies also gradient information.
     * @throws MatrixException throws exception is matrix is not defined.
     */
    public void setData(Node node, boolean copyGradients) throws MatrixException {
        fromNode.setData(node, copyGradients);
        toNode.setData(node, copyGradients);
    }

}
