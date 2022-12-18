/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.memory;

import core.reinforcement.agent.StateTransition;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

/**
 * Implements sum tree using search tree interface.<br>
 * Links leaf nodes together as forward cycled list.<br>
 * Stores mapping of state transition and each leaf node containing respective state transition.<br>
 * <br>
 * Reference: https://www.endtoend.ai/deep-rl-seminar/2#prioritized-experience-replay and https://github.com/jaromiru/AI-blog/blob/master/SumTree.py <br>
 *
 */
public class SumTree implements SearchTree, Serializable {

    @Serial
    private static final long serialVersionUID = 1588699500252286228L;

    /**
     * Implements node of sum tree.
     *
     */
    private static class Node {

        /**
         * Priority sum of node
         *
         */
        private double prioritySum;

        /**
         * Parent node. If null node is root node.
         *
         */
        private final Node parentNode;

        /**
         * Left child node. If null there is no left node and if right node is null as well node is leaf node.
         *
         */
        private Node leftNode;

        /**
         * Right child node. If null there is no right node and if left node is null as well node is leaf node.
         *
         */
        private Node rightNode;

        /**
         * State transition stored in leaf node.
         *
         */
        private StateTransition stateTransition;

        /**
         * Next leaf node in forward traversing order. Last node is linked back to first node creating cyclical traversal.
         *
         */
        private Node nextLeafNode;

        /**
         * Constructor for root node.
         *
         */
        Node() {
            this.parentNode = null;
        }

        /**
         * Constructor for node.
         *
         * @param parentNode parent node.
         */
        Node(Node parentNode) {
            this.parentNode = parentNode;
        }

        /**
         * Checks if node is non-root node;
         *
         * @return true if node is non-root node otherwise false.
         */
        boolean isNonRootNode() {
            return parentNode != null;
        }

        /**
         * Returns parent node.
         *
         * @return parent node.
         */
        Node getParentNode() {
            return parentNode;
        }

        /**
         * Sets left node.
         *
         * @param leftNode left node.
         * @return left node.
         */
        Node setLeftNode(Node leftNode) {
            this.leftNode = leftNode;
            return leftNode;
        }

        /**
         * Returns left node.
         *
         * @return left node.
         */
        Node getLeftNode() {
            return leftNode;
        }

        /**
         * Checks if node has left node.
         *
         * @return true if node has left node otherwise false.
         */
        boolean hasLeftNode() {
            return leftNode !=null;
        }

        /**
         * Sets right node.
         *
         * @param rightNode right node.
         * @return right node.
         */
        Node setRightNode(Node rightNode) {
            this.rightNode = rightNode;
            return rightNode;
        }

        /**
         * Returns right node.
         *
         * @return right node.
         */
        Node getRightNode() {
            return rightNode;
        }

        /**
         * Checks if node has right node.
         *
         * @return true if node has right node otherwise false.
         */
        boolean hasRightNode() {
            return rightNode !=null;
        }

        /**
         * Sets next leaf node.
         *
         * @param nextLeafNode next left node.
         */
        void setNextLeafNode(Node nextLeafNode) {
            this.nextLeafNode = nextLeafNode;
        }

        /**
         * Gets next leaf node.
         *
         * @return next leaf node.
         */
        Node getNextLeafNode() {
            return nextLeafNode;
        }

        /**
         * Sets priority sum for node.
         *
         * @param prioritySum priority sum for node.
         * @return priority sum delta.
         */
        double setPrioritySum(double prioritySum) {
            double prioritySumDelta = prioritySum - getPrioritySum();
            this.prioritySum = prioritySum;
            return prioritySumDelta;
        }

        /**
         * Returns priority sum.
         *
         * @return priority sum.
         */
        double getPrioritySum() {
            return prioritySum;
        }

        /**
         * Increments priority sum delta to node.
         *
         * @param prioritySumDelta priority sum delta.
         */
        void incrementPrioritySum(double prioritySumDelta) {
            this.prioritySum += prioritySumDelta;
        }

        /**
         * Set state transition to node.
         *
         * @param stateTransition state transition to node.
         */
        void setStateTransition(StateTransition stateTransition) {
            this.stateTransition = stateTransition;
        }

        /**
         * Returns state transition of node.
         *
         * @return state transition of node.
         */
        StateTransition getStateTransition() {
            return stateTransition;
        }

        /**
         * Updates priority sum of node and traverses priority change up to root node as priority sum delta.
         *
         * @param prioritySum priority sum for a node.
         */
        void updatePrioritySum(double prioritySum) {
            if (isNonRootNode()) getParentNode().propagatePrioritySum(setPrioritySum(prioritySum));
        }

        /**
         * Propagates priority sum to root node.
         *
         * @param prioritySumDelta priority sum.
         */
        private void propagatePrioritySum(double prioritySumDelta) {
            incrementPrioritySum(prioritySumDelta);
            if (isNonRootNode()) getParentNode().propagatePrioritySum(prioritySumDelta);
        }

        /**
         * Retrieves state transition by priority sum. Traverses sum tree down until leaf node matching priority and containing state transition is found.
         *
         * @param prioritySum priority sum.
         * @return state state transition corresponding priority sum.
         */
        StateTransition getStateTransition(double prioritySum) {
            return !hasLeftNode() ? getStateTransition() : prioritySum <= getLeftNode().getPrioritySum() ? getLeftNode().getStateTransition(prioritySum) : !hasRightNode() ? getStateTransition() : getRightNode().getStateTransition(prioritySum - getLeftNode().getPrioritySum());
        }

    }

    /**
     * Capacity of sum tree.
     *
     */
    private final int capacity;

    /**
     * Map that links state transition store in sum tree to respective leaf node.
     *
     */
    private final HashMap<StateTransition, Node> stateTransitionNodeHashMap = new HashMap<>();

    /**
     * Root node of sum tree.
     *
     */
    private final Node rootNode;

    /**
     * Current leaf node of sum tree where new state transition is to be added.
     *
     */
    private Node currentLeafNode;

    /**
     * List of leaf nodes.
     *
     */
    private final ArrayList<Node> leafNodes = new ArrayList<>();

    /**
     * Current maximum priority of leaf nodes. Used for newly added sample as default priority.
     *
     */
    private double maxPriority = 0.001;

    /**
     * Random number generator.
     *
     */
    private final Random random = new Random();

    /**
     * Default constructor for sum tree.
     *
     * @param capacity capacity (number of leaf nodes) of sum tree.
     */
    SumTree(int capacity) {
        this.capacity = capacity;
        currentLeafNode = getStartNode(rootNode = construct(capacity));
    }

    /**
     * Constructs sum tree.
     *
     * @param capacity capacity of sum tree.
     * @return root node of sum tree.
     */
    private Node construct(int capacity) {
        ArrayDeque<Node> nodes = new ArrayDeque<>();

        Node rootNode = new Node();
        nodes.add(rootNode);

        while (nodes.size() < capacity && !nodes.isEmpty()) {
            Node parentNode = nodes.poll();
            nodes.add(parentNode.setLeftNode(new Node(parentNode)));
            nodes.add(parentNode.setRightNode(new Node(parentNode)));
        }

        Iterator<Node> nodeIterator = nodes.iterator();
        Node previousNode = null;
        Node firstNode = null;
        while (nodeIterator.hasNext()) {
            Node node = nodeIterator.next();
            if (previousNode == null) firstNode = node;
            else previousNode.setNextLeafNode(node);
            previousNode = node;
        }

        if (previousNode != null) previousNode.setNextLeafNode(firstNode);
        return rootNode;
    }

    /**
     * Returns left most leaf node of sum tree.
     *
     * @param rootNode root node of sum tree.
     * @return left most leaf node of sum tree.
     */
    private Node getStartNode(Node rootNode) {
        Node leftNode = rootNode;
        while (leftNode.hasLeftNode()) leftNode = leftNode.getLeftNode();
        return leftNode;
    }

    /**
     * Current size of sum tree i.e. number of state transitions stored.
     *
     * @return size of sum tree.
     */
    public int size() {
        return stateTransitionNodeHashMap.size();
    }

    /**
     * Returns total capacity of sum tree.
     *
     * @return total capacity of sum tree.
     */
    private int capacity() {
        return capacity;
    }

    /**
     * Returns total priority of sum tree i.e. total priority sum of sum tree stored in root node.
     *
     * @return total priority of sum tree.
     */
    public double getTotalPriority() {
        return rootNode.prioritySum;
    }

    /**
     * Adds state transition in sum tree at the location of current node and shifts current node one forward.
     * Updates total priority of sum tree according to priority of added state transition.
     *
     * @param stateTransition state transition to be added.
     */
    public void add(StateTransition stateTransition) {
        if (currentLeafNode.getStateTransition() != null) currentLeafNode.getStateTransition().removePreviousStateTransition();
        stateTransitionNodeHashMap.remove(currentLeafNode.getStateTransition());
        if (leafNodes.size() < capacity()) leafNodes.add(currentLeafNode);
        currentLeafNode.setStateTransition(stateTransition);
        stateTransitionNodeHashMap.put(stateTransition, currentLeafNode);
        stateTransition.priority = maxPriority = Math.max(maxPriority, stateTransition.priority);
        currentLeafNode.updatePrioritySum(maxPriority);
        currentLeafNode = currentLeafNode.getNextLeafNode();
    }

    /**
     * Updates priority of state transition and entire sum tree.
     *
     * @param stateTransition state transition to be updated.
     */
    public void update(StateTransition stateTransition) {
        Node node = stateTransitionNodeHashMap.get(stateTransition);
        maxPriority = Math.max(maxPriority, stateTransition.priority);
        node.updatePrioritySum(stateTransition.priority);
    }

    /**
     * Returns state transition by priority sum.
     *
     * @param prioritySum priority sum.
     * @return state transition according to priority sum.
     */
    public StateTransition getStateTransition(double prioritySum) {
        return rootNode.getStateTransition(prioritySum);
    }

    /**
     * Returns random state transition.
     *
     * @return random state transition.
     */
    public StateTransition getRandomStateTransition() {
        return leafNodes.get(random.nextInt(leafNodes.size())).getStateTransition();
    }

}
