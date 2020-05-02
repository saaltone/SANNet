/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.procedure;

import core.normalization.Normalization;
import utils.matrix.Init;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

/**
 * Class that provides node instances and keeps register of them.
 *
 */
class NodeRegister implements Serializable {

    private static final long serialVersionUID = 1317148695277485847L;

    /**
     * Class structure for node entry.
     *
     */
    private static class NodeEntry implements Serializable {

        private static final long serialVersionUID = 5723809008297279706L;

        /**
         * Reference to node instance.
         *
         */
        final Node node;

        /**
         * Procedure ID where node was created in.
         *
         */
        final int procedureID;

        /**
         * Expression ID where node was created in.
         *
         */
        final int expressionID;

        /**
         * Constructor for node entry.
         *
         * @param node reference to node instance.
         * @param procedureID procedure ID where node was created in.
         * @param expressionID expression ID where node was created in.
         */
        NodeEntry(Node node, int procedureID, int expressionID) {
            this.node = node;
            this.procedureID = procedureID;
            this.expressionID = expressionID;
        }
    }

    /**
     * Map to maintain dependencies of matrices and node entries.
     *
     */
    private final HashMap<Matrix, NodeEntry> entriesByMatrix = new HashMap<>();

    /**
     * Map to maintain dependencies of nodes and node entries.
     *
     */
    private final HashMap<Node, NodeEntry> entriesByNode = new HashMap<>();

    /**
     * Map to maintain dependencies of matrices and node.
     *
     */
    private final HashMap<Matrix, Node> nodeMap = new HashMap<>();

    /**
     * Default constructor for node register.
     *
     */
    NodeRegister() {
    }

    /**
     * Clears node register.
     *
     */
    void clear() {
        entriesByMatrix.clear();
        entriesByNode.clear();
        nodeMap.clear();
    }

    /**
     * Defines and returns node by matrix.<br>
     * If node is not existing creates node and records in which procedure and expression ID it was created in.<br>
     *
     * @param matrix reference to matrix
     * @param constantNode if true node is marked as constant type
     * @param procedureID procedure ID where node was created in
     * @param expressionID expression ID where node was created in
     * @return node created or retrieved.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Node defineNode(Matrix matrix, boolean constantNode, boolean createMatrixIfNone, HashSet<Normalization> normalizers, int procedureID, int expressionID) throws MatrixException {
        if (entriesByMatrix.containsKey(matrix)) return nodeMap.get(matrix);
        else {
            Node node = new Node(matrix, constantNode, createMatrixIfNone, matrix.getInitType() == Init.CONSTANT, normalizers);
            if (constantNode) node.setMatrix(0, matrix);
            NodeEntry nodeEntry = new NodeEntry(node, procedureID, expressionID);
            entriesByMatrix.put(matrix, nodeEntry);
            entriesByNode.put(node, nodeEntry);
            nodeMap.put(matrix, node);
            return node;
        }
    }

    /**
     * Returns node by matrix.
     *
     * @param matrix matrix corresponding node requested.
     * @return returned node.
     */
    Node getNode(Matrix matrix) {
        return entriesByMatrix.get(matrix).node;
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
    HashMap<Matrix, Node> getNodeMap() {
        return nodeMap;
    }

    /**
     * Returns procedure ID corresponding the node.
     *
     * @param node node in question.
     * @return procedure ID corresponding the node.
     */
    public int getProcedureID(Node node) {
        if (entriesByNode.containsKey(node)) return entriesByNode.get(node).procedureID;
        else return -1;
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
     * Removes procedure factory from nodes of node register.
     *
     */
    public void removeProcedureFactory() {
        for (Matrix matrix : nodeMap.keySet())  matrix.removeProcedureFactory();
    }

}
