/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.node;

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
    private static final long serialVersionUID = -232611915649428858L;

    /**
     * Record that defines node entry.
     *
     * @param node reference to node instance.
     */
    private record NodeEntry(Node node) {
    }

    /**
     * Map to maintain dependencies of matrices and node entries.
     *
     */
    private final HashMap<Matrix, NodeEntry> entriesByMatrix = new HashMap<>();

    /**
     * Map to maintain dependencies of matrices and node.
     *
     */
    private final HashMap<Matrix, Node> nodeMatrixMap = new HashMap<>();

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
     * @param nodeID node ID
     * @return node created or retrieved.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Node defineNode(Matrix matrix, boolean isSingleNode, int nodeID) throws MatrixException {
        Node node = nodeMatrixMap.get(matrix);
        if (node == null) {
            node = isSingleNode ? new SingleNode(nodeID, matrix) : new MultiNode(nodeID, matrix);
            entriesByMatrix.put(matrix, new NodeEntry(node));
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
     * Returns node by matrix.
     *
     * @param matrix matrix corresponding node requested.
     * @return returned node.
     */
    public Node getNode(Matrix matrix) {
        return entriesByMatrix.get(matrix).node;
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
     * Removes procedure factory from nodes of node register.
     *
     */
    public void removeProcedureFactory() {
        for (Matrix matrix : nodeMatrixMap.keySet()) matrix.removeProcedureFactory();
    }

}
