/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.node;

import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashMap;

/**
 * Implements functionality to create and keep register of node instances.<br>
 *
 */
public class NodeRegister implements Serializable {

    @Serial
    private static final long serialVersionUID = 1317148695277485847L;

    /**
     * Record that defines node entry.
     *
     * @param node reference to node instance.
     * @param expressionID expression ID where node was created in.
     */
    private record NodeEntry(Node node, int expressionID) {
    }

    /**
     * Map to maintain dependencies of matrices and node entries.
     *
     */
    private final HashMap<Matrix, NodeEntry> entriesByMatrix = new HashMap<>();

    /**
     * Map to maintain dependencies of multi-matrices and node entries.
     *
     */
    private final HashMap<MMatrix, NodeEntry> entriesByMMatrix = new HashMap<>();

    /**
     * Map to maintain dependencies of matrices and node.
     *
     */
    private final HashMap<Matrix, Node> nodeMatrixMap = new HashMap<>();

    /**
     * Map to maintain dependencies of multi-matrices and node.
     *
     */
    private final HashMap<MMatrix, Node> nodeMMatrixMap = new HashMap<>();

    /**
     * Default constructor for node register.
     *
     */
    public NodeRegister() {
    }

    /**
     * Defines and returns node by matrix.<br>
     * If node is not existing creates node with unique expression ID.<br>
     *
     * @param matrix reference to matrix
     * @param isSingleNode if true node is marked as single type
     * @param expressionID expression ID where node was created in
     * @param nodeID node ID
     * @return node created or retrieved.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Node defineNode(Matrix matrix, boolean isSingleNode, int expressionID, int nodeID) throws MatrixException {
        Node node = nodeMatrixMap.get(matrix);
        if (node == null) {
            node = isSingleNode ? new SingleNode(nodeID, matrix) : new MultiNode(nodeID, matrix);
            NodeEntry nodeEntry = new NodeEntry(node, expressionID);
            entriesByMatrix.put(matrix, nodeEntry);
            nodeMatrixMap.put(matrix, node);
        }
        return node;
    }

    /**
     * Checks if node corresponding matrix exists.
     *
     * @param matrix reference to matrix
     * @return true if node corresponding matrix exists otherwise false.
     */
    public boolean nodeExists(Matrix matrix) {
        return nodeMatrixMap.containsKey(matrix);
    }

    /**
     * Defines and returns node by matrix.<br>
     * If node is not existing creates node with unique expression ID.<br>
     *
     * @param mMatrix reference to multi-matrix
     * @param isSingleNode if true node is marked as single type
     * @param expressionID expression ID where node was created in
     * @param nodeID node ID
     * @return node created or retrieved.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Node defineNode(MMatrix mMatrix, boolean isSingleNode, int expressionID, int nodeID) throws MatrixException {
        Node node = nodeMMatrixMap.get(mMatrix);
        if (node == null)  {
            node = isSingleNode ? new SingleNode(nodeID, mMatrix.getReferenceMatrix()) : new MultiNode(nodeID, mMatrix.getReferenceMatrix());
            NodeEntry nodeEntry = new NodeEntry(node, expressionID);
            entriesByMMatrix.put(mMatrix, nodeEntry);
            nodeMMatrixMap.put(mMatrix, node);
        }
        return node;
    }

    /**
     * Checks if node corresponding matrix exists.
     *
     * @param mMatrix reference to multi-matrix
     * @return true if node corresponding matrix exists otherwise false.
     */
    public boolean nodeExists(MMatrix mMatrix) {
        return nodeMMatrixMap.containsKey(mMatrix);
    }

    /**
     * Returns node by matrix.
     *
     * @param matrix matrix corresponding node requested.
     * @return returned node.
     */
    public Node getNode(Matrix matrix) {
        return entriesByMatrix.get(matrix).node;
    }

    /**
     * Returns node by matrix.
     *
     * @param matrix matrix corresponding node requested.
     * @return returned node.
     */
    public Node getNode(MMatrix matrix) {
        return entriesByMMatrix.get(matrix).node;
    }

    /**
     * Checks if node register contains matrix.
     *
     * @param matrix matrix in question.
     * @return true is matrix is contained by the node register otherwise false.
     */
    public boolean contains(Matrix matrix) {
        return entriesByMatrix.containsKey(matrix);
    }

    /**
     * Checks if node register contains multi-matrix.
     *
     * @param matrix multi-matrix in question.
     * @return true is matrix is contained by the node register otherwise false.
     */
    public boolean contains(MMatrix matrix) {
        return entriesByMMatrix.containsKey(matrix);
    }

    /**
     * Removes procedure factory from nodes of node register.
     *
     */
    public void removeProcedureFactory() {
        for (Matrix matrix : nodeMatrixMap.keySet()) matrix.removeProcedureFactory();
    }

}
