/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.procedure.node;

import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Set;

/**
 * Class that provides node instances and keeps register of them.<br>
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
     * Map to maintain dependencies of nodes and node entries.
     *
     */
    private final HashMap<Node, NodeEntry> entriesByNode = new HashMap<>();

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
     * @return node created or retrieved.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Node defineNode(Matrix matrix, boolean isSingleNode, int expressionID) throws MatrixException {
        if (entriesByMatrix.containsKey(matrix)) return nodeMatrixMap.get(matrix);
        else {
            Node node = isSingleNode ? new SingleNode(getTotalSize() + 1, matrix) : new MultiNode(getTotalSize() + 1, matrix);
            NodeEntry nodeEntry = new NodeEntry(node, expressionID);
            entriesByMatrix.put(matrix, nodeEntry);
            entriesByNode.put(node, nodeEntry);
            nodeMatrixMap.put(matrix, node);
            return node;
        }
    }

    /**
     * Defines and returns node by matrix.<br>
     * If node is not existing creates node with unique expression ID.<br>
     *
     * @param matrix reference to matrix
     * @param isSingleNode if true node is marked as single type
     * @param expressionID expression ID where node was created in
     * @return node created or retrieved.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Node defineNode(MMatrix matrix, boolean isSingleNode, int expressionID) throws MatrixException {
        if (entriesByMMatrix.containsKey(matrix)) return nodeMMatrixMap.get(matrix);
        else {
            Node node = isSingleNode ? new SingleNode(getTotalSize() + 1, matrix) : new MultiNode(getTotalSize() + 1, matrix);
            NodeEntry nodeEntry = new NodeEntry(node, expressionID);
            entriesByMMatrix.put(matrix, nodeEntry);
            entriesByNode.put(node, nodeEntry);
            nodeMMatrixMap.put(matrix, node);
            return node;
        }
    }

    /**
     * Returns total size of node register.
     *
     * @return total size of node register.
     */
    public int getTotalSize() {
        return entriesByMatrix.size() + entriesByMMatrix.size();
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
     * Returns node set contained by register.
     *
     * @return node set contained by register.
     */
    public Set<Node> getNodes() {
        return entriesByNode.keySet();
    }

    /**
     * Returns node map contained by register.
     *
     * @return node map contained by register.
     */
    public HashMap<Matrix, Node> getNodeMatrixMap() {
        return nodeMatrixMap;
    }

    /**
     * Returns node map contained by register.
     *
     * @return node map contained by register.
     */
    public HashMap<MMatrix, Node> getNodeMMatrixMap() {
        return nodeMMatrixMap;
    }

    /**
     * Returns expression ID corresponding the node.
     *
     * @param node node in question.
     * @return expression ID corresponding the node.
     */
    public int getExpressionID(Node node) {
        if (entriesByNode.containsKey(node)) return entriesByNode.get(node).expressionID;
        else return -1;
    }

    /**
     * Checks if node register contains node.
     *
     * @param node node in question.
     * @return true is node is contained by the node register otherwise false.
     */
    public boolean contains(Node node) {
        return entriesByNode.containsKey(node);
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
        for (Matrix matrix : nodeMatrixMap.keySet())  matrix.removeProcedureFactory();
    }

}
