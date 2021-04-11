/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.memory;

import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;

/**
 * Implements SumTree using tree structure.<br>
 * Links leaf nodes together as forward cycled list.<br>
 * Stores mapping of state transition and each leaf node containing respective state transition.<br>
 * <br>
 * Reference: https://www.endtoend.ai/deep-rl-seminar/2#prioritized-experience-replay and https://github.com/jaromiru/AI-blog/blob/master/SumTree.py <br>
 *
 */
public class SumTree implements Serializable {

    private static final long serialVersionUID = 1588699500252286228L;

    /**
     * Class that implements Node of SumTree.
     *
     */
    private static class Node {

        /**
         * Priority sum of node
         *
         */
        double prioritySum;

        /**
         * Parent node. If null node is root node.
         *
         */
        Node parentNode;

        /**
         * Left child node. If null there is no left node and if right node is null as well node is leaf node.
         *
         */
        Node leftNode;

        /**
         * Right child node. If null there is no right node and if left node is null as well node is leaf node.
         *
         */
        Node rightNode;

        /**
         * State transition stored in leaf node.
         *
         */
        StateTransition stateTransition;

        /**
         * Next leaf node in forward traversing order. Last node is linked back to first node creating cyclical traversal.
         *
         */
        Node nextLeafNode;

        /**
         * Default constructor for Node.
         *
         */
        Node() {
        }

        /**
         * Updates priority of node and traverses priority change up to root node as priority delta.
         *
         * @param updatedPriority updated priority for a node.
         */
        void updatePriority(double updatedPriority) {
            if (parentNode != null) parentNode.updatePriority(leftNode == null ? updatedPriority - prioritySum : updatedPriority);
            prioritySum = leftNode == null ? updatedPriority : prioritySum + updatedPriority;
        }

        /**
         * Retrieves state transition by priority sum. Traverses sum tree down until leaf node matching priority and containing state transition is found.
         *
         * @param prioritySum priority sum.
         * @return state state transition corresponding priority sum.
         */
        StateTransition getStateTransition(double prioritySum) {
            if (leftNode == null) return stateTransition;
            else {
                if (prioritySum <= leftNode.prioritySum || rightNode == null) return leftNode.getStateTransition(prioritySum);
                else return rightNode.getStateTransition(prioritySum - leftNode.prioritySum);
            }
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
     * Current maximum priority of leaf nodes. Used for newly added sample as default priority.
     *
     */
    private double maxPriority = 0.001;

    /**
     * Default constructor for SumTree.
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
        Node rootNode = new Node();
        ArrayDeque<Node> nodes = new ArrayDeque<>();
        nodes.add(rootNode);
        while (nodes.size() < capacity && !nodes.isEmpty()) {
            Node node = nodes.poll();
            nodes.add(node.leftNode = new Node());
            node.leftNode.parentNode = node;
            if (nodes.size() == capacity) break;
            nodes.add(node.rightNode = new Node());
            node.rightNode.parentNode = node;
            if (nodes.size() == capacity) break;
        }
        Iterator<Node> nodeIterator = nodes.iterator();
        Node previousNode = null;
        Node firstNode = null;
        while (nodeIterator.hasNext()) {
            Node node = nodeIterator.next();
            if (previousNode != null) previousNode.nextLeafNode = node;
            else firstNode = node;
            previousNode = node;
        }
        if (previousNode != null) previousNode.nextLeafNode = firstNode;
        return rootNode;
    }

    /**
     * Returns left most leaf node of sum tree.
     *
     * @param rootNode root node of sum tree.
     * @return left most leaf node of sum tree.
     */
    private Node getStartNode(Node rootNode) {
        Node node = rootNode;
        while (node.leftNode != null) node = node.leftNode;
        return node;
    }

    /**
     * Current size of sum tree i.e. number of state transitions stored.
     *
     * @return size of sum tree.
     */
    int size() {
        return stateTransitionNodeHashMap.size();
    }

    /**
     * Returns total capacity of sum tree.
     *
     * @return total capacity of sum tree.
     */
    int capacity() {
        return capacity;
    }

    /**
     * Returns total priority of sum tree i.e. total priority sum of sum tree stored in root node.
     *
     * @return total priority of sum tree.
     */
    double getTotalPriority() {
        return rootNode.prioritySum;
    }

    /**
     * Adds state transition in sum tree at the location of current node and shifts current node one forward.
     * Updates total priority of sum tree according to priority of added state transition.
     *
     * @param stateTransition state transition to be added.
     */
    void add(StateTransition stateTransition) {
        stateTransitionNodeHashMap.remove(currentLeafNode.stateTransition);
        currentLeafNode.stateTransition = stateTransition;
        stateTransitionNodeHashMap.put(stateTransition, currentLeafNode);
        stateTransition.priority = maxPriority = Math.max(maxPriority, stateTransition.priority);
        currentLeafNode.updatePriority(maxPriority);
        currentLeafNode = currentLeafNode.nextLeafNode;
    }

    /**
     * Updates priority of state transition and entire sum tree.
     *
     * @param stateTransition state transition to be updated.
     */
    void update(StateTransition stateTransition) {
        Node node = stateTransitionNodeHashMap.get(stateTransition);
        if (node != null) {
            maxPriority = Math.max(maxPriority, stateTransition.priority);
            node.updatePriority(stateTransition.priority);
        }
    }

    /**
     * Returns state transition by priority sum.
     *
     * @param prioritySum priority sum.
     * @return state transition according to priority sum.
     */
    StateTransition getStateTransition(double prioritySum) {
        return rootNode.getStateTransition(prioritySum);
    }

    /**
     * Returns random state transition.
     *
     * @return random state transition.
     */
    StateTransition getRandomStateTransition() {
        StateTransition[] stateTransitionArray = stateTransitionNodeHashMap.keySet().toArray(new StateTransition[0]);
        return stateTransitionArray[new Random().nextInt(stateTransitionArray.length)];
    }

}
