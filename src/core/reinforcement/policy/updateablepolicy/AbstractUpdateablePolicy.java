/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.policy.updateablepolicy;

import core.network.NeuralNetworkException;
import core.optimization.Adam;
import core.optimization.Optimizer;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.AbstractPolicy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.JMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.TreeSet;

/**
 * Class that defines AbstractUpdateablePolicy.<br>
 * Contains common functions fo updateable policies.<br>
 *
 */
public abstract class AbstractUpdateablePolicy extends AbstractPolicy {

    /**
     * Parameter name types for UpdateableSoftQPolicy.
     *     - softQAlpha; entropy regularization coefficient. Default value 1.<br>
     *     - autoSoftAlpha; if true alpha is adjusted automatically otherwise not. Default value true.<br>
     *
     */
    private final static String paramNameTypes = "(softQAlpha:DOUBLE), " +
            "(autoSoftAlpha:BOOLEAN)";

    /**
     * Alpha parameter for entropy control.
     *
     */
    private double softQAlpha;

    /**
     * If true alpha is adjusted automatically otherwise not.
     *
     */
    private boolean autoSoftAlpha;

    /**
     * Alpha parameter for entropy in matrix form.
     *
     */
    private final Matrix softQAlphaMatrix;

    /**
     * Gradient matrix for alpha soft Q value.
     *
     */
    private final Matrix softQAlphaMatrixGradient = new DMatrix(0);

    /**
     * Cumulative alpha loss.
     *
     */
    private transient double cumulativeAlphaLoss = 0;

    /**
     * Update count for alpha loss.
     *
     */
    private transient int alphaLossCount = 0;

    /**
     * Optimizer for alpha loss.
     *
     */
    private final Optimizer optimizer = new Adam();

    /**
     * Constructor for AbstractUpdateablePolicy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to FunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public AbstractUpdateablePolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator);
        this.softQAlphaMatrix = new DMatrix(0);
    }

    /**
     * Constructor for AbstractUpdateablePolicy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to FunctionEstimator.
     * @param softQAlphaMatrix reference to softQAlphaMatrix.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public AbstractUpdateablePolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, Matrix softQAlphaMatrix) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator);
        this.softQAlphaMatrix = softQAlphaMatrix;
    }

    /**
     * Constructor for AbstractUpdateablePolicy.
     *
     * @param executablePolicy executable policy.
     * @param functionEstimator reference to FunctionEstimator.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractUpdateablePolicy(ExecutablePolicy executablePolicy, FunctionEstimator functionEstimator) throws AgentException, DynamicParamException {
        super(executablePolicy, functionEstimator);
        this.softQAlphaMatrix = new DMatrix(0);
    }

    /**
     * Constructor for AbstractUpdateablePolicy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for AbstractUpdateablePolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public AbstractUpdateablePolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, String params) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator, params);
        this.softQAlphaMatrix = new DMatrix(0);
    }

    /**
     * Constructor for AbstractUpdateablePolicy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for AbstractUpdateablePolicy.
     * @param softQAlphaMatrix reference to softQAlphaMatrix.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public AbstractUpdateablePolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, String params, Matrix softQAlphaMatrix) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator, params);
        this.softQAlphaMatrix = softQAlphaMatrix;
    }

    /**
     * Constructor for AbstractUpdateablePolicy.
     *
     * @param executablePolicy executable policy.
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for AbstractExecutablePolicy.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractUpdateablePolicy(ExecutablePolicy executablePolicy, FunctionEstimator functionEstimator, String params) throws AgentException, DynamicParamException {
        super(executablePolicy, functionEstimator, params);
        this.softQAlphaMatrix = new DMatrix(0);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        setSoftAlpha(1);
        autoSoftAlpha = true;
    }

    /**
     * Returns parameters used for AbstractUpdateablePolicy.
     *
     * @return parameters used for AbstractUpdateablePolicy.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + AbstractUpdateablePolicy.paramNameTypes;
    }

    /**
     * Sets parameters used for AbstractUpdateablePolicy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - softQAlpha; entropy regularization coefficient. Default value 1.<br>
     *     - autoSoftAlpha; if true alpha is adjusted automatically otherwise not. Default value true.<br>
     *
     * @param params parameters used for AbstractUpdateablePolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("softQAlpha")) setSoftAlpha(params.getValueAsDouble("softQAlpha"));
        if (params.hasParam("autoSoftAlpha")) autoSoftAlpha = params.getValueAsBoolean("autoSoftAlpha");
    }

    /**
     * Return true if policy is updateable otherwise false.
     *
     * @return true if policy is updateable otherwise false.
     */
    public boolean isUpdateablePolicy() {
        return true;
    }

    /**
     * Takes action by applying defined executable policy.
     *
     * @param stateTransition state transition.
     * @param alwaysGreedy if true greedy action is always taken.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void act(StateTransition stateTransition, boolean alwaysGreedy) throws NeuralNetworkException, MatrixException {
        super.act(stateTransition, alwaysGreedy);
        if (isLearning()) executablePolicy.record(stateTransition);
    }

    /**
     * Updates policy.
     *
     */
    public void update() {
        executablePolicy.finish(isLearning());
    }

    /**
     * Resets FunctionEstimator.
     *
     */
    public void resetFunctionEstimator() {
        functionEstimator.reset();
    }

    /**
     * Notifies that agent is ready to update.
     *
     * @param agent current agent.
     * @throws AgentException throws exception if agent is not registered for function estimator.
     * @return true if all registered agents are ready to update.
     */
    public boolean readyToUpdate(Agent agent) throws AgentException {
        return functionEstimator.readyToUpdate(agent);
    }

    /**
     * Updates policy.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if function estimator update fails.
     */
    public void updateFunctionEstimator() throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException {
        TreeSet<StateTransition> sampledStateTransitions = functionEstimator.getSampledStateTransitions();
        if (sampledStateTransitions == null || sampledStateTransitions.isEmpty()) return;

        preProcess();
        for (StateTransition stateTransition : sampledStateTransitions) functionEstimator.store(stateTransition, getPolicyValues(stateTransition));
        postProcess();

        functionEstimator.update();

        if (isAutoSoftAlpha()) updateAlpha();
    }

    /**
     * Returns policy values.
     *
     * @param stateTransition state transition.
     * @return policy values.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    private Matrix getPolicyValues(StateTransition stateTransition) throws MatrixException, NeuralNetworkException {
        if (isStateActionValueFunction()) {
            Matrix stateValue = new DMatrix(1, 1);
            stateValue.setValue(0, 0, stateTransition.tdTarget);
            Matrix policyValues = new DMatrix(functionEstimator.getNumberOfActions(), 1);
            policyValues.setValue(stateTransition.action, 0, getPolicyValue(stateTransition));
            return new JMatrix(stateValue.getTotalRows() + policyValues.getTotalRows(), 1, new Matrix[] {stateValue, policyValues}, true);
        }
        else {
            Matrix policyValues = new DMatrix(functionEstimator.getNumberOfActions(), 1);
            policyValues.setValue(stateTransition.action, 0, getPolicyValue(stateTransition));
            return policyValues;
        }
    }

    /**
     * Preprocesses policy gradient update.
     *
     */
    protected void preProcess() {
    }

    /**
     * Returns policy value for StateTransition.
     *
     * @param stateTransition state transition.
     * @return policy gradient value for sample.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract double getPolicyValue(StateTransition stateTransition) throws NeuralNetworkException, MatrixException;

    /**
     * Postprocesses policy gradient update.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    protected void postProcess() throws MatrixException, AgentException {
    }

    /**
     * Increments alpha.
     *
     * @param policyValues policy values.
     * @param policyValue policy value
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void incrementAlpha(Matrix policyValues, double policyValue) throws MatrixException {
        double entropy = policyValues.entropy(true);
        cumulativeAlphaLoss += policyValue + entropy;
        alphaLossCount++;
    }

    /**
     * Updates alpha.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void updateAlpha() throws MatrixException, DynamicParamException {
        // https://raw.githubusercontent.com/BY571/Deep-Reinforcement-Learning-Algorithm-Collection/master/ContinousControl/SAC.ipynb
        // self.target_entropy = -action_size  # -dim(A)
        // alpha_loss = - (self.log_alpha * (log_pis + self.target_entropy).detach()).mean()
        softQAlphaMatrix.setValue(0,0, softQAlpha);
        softQAlphaMatrixGradient.setValue(0, 0, softQAlpha * cumulativeAlphaLoss / (double) alphaLossCount);
        optimizer.optimize(softQAlphaMatrix, softQAlphaMatrixGradient);
        softQAlpha = softQAlphaMatrix.getValue(0, 0);
        cumulativeAlphaLoss = 0;
        alphaLossCount = 0;
    }

    /**
     * Returns soft alpha.
     *
     * @return soft alpha.
     */
    protected double getSoftAlpha() {
        return softQAlpha;
    }

    /**
     * Sets soft alpha.
     *
     * @param softAlpha soft alpha.
     */
    protected void setSoftAlpha(double softAlpha) {
        this.softQAlpha = softAlpha;
    }

    /**
     * Returns true if alpha is adjusted automatically otherwise not.
     *
     * @return true if alpha is adjusted automatically otherwise not.
     */
    protected boolean isAutoSoftAlpha() {
        return autoSoftAlpha;
    }

}
