/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.policy.updateablepolicy;

import core.network.NeuralNetworkException;
import core.optimization.Optimizer;
import core.optimization.OptimizerFactory;
import core.reinforcement.agent.AgentException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.agent.State;
import core.reinforcement.memory.Memory;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.value.SoftQValueFunctionEstimator;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements updateable soft Q policy.<br>
 *
 */
public class UpdateableSoftQPolicy extends AbstractUpdateablePolicy {

    /**
     * Parameter name types for updateable soft Q policy.
     *     - softQAlpha; entropy regularization coefficient. Default value 1.<br>
     *     - autoSoftAlpha; if true alpha is adjusted automatically otherwise not. Default value true.<br>
     *     - softQAlphaVerboseInterval; verbose internal for alpha. Default value 25.<br>
     *
     */
    private final static String paramNameTypes = "(softQAlpha:DOUBLE), " +
            "(autoSoftAlpha:BOOLEAN), " +
            "(softQAlphaVerboseInterval:INT)";

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
     * Cumulative alpha loss.
     *
     */
    private transient double cumulativeAlphaLoss = 0;

    /**
     * Update count for cumulative alpha loss.
     *
     */
    private transient int alphaLossCount = 0;

    /**
     * Optimizer for alpha loss.
     *
     */
    private final Optimizer optimizer = OptimizerFactory.createDefault();

    /**
     * Soft Q alpha verbose interval.
     *
     */
    private int softQAlphaVerboseInterval;

    /**
     * Soft Q alpha verbose count.
     *
     */
    private int softQAlphaVerboseCount = 0;

    /**
     * Soft Q value function estimator.
     *
     */
    private final SoftQValueFunctionEstimator softQValueFunctionEstimator;

    /**
     * Constructor for updateable soft Q policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to function estimator.
     * @param softQValueFunctionEstimator reference to soft Q value function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public UpdateableSoftQPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, SoftQValueFunctionEstimator softQValueFunctionEstimator) throws DynamicParamException, AgentException {
        this(executablePolicyType, functionEstimator, null, softQValueFunctionEstimator);
    }

    /**
     * Constructor for updateable soft Q policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator    reference to function estimator.
     * @param params               parameters for updateable soft Q policy.
     * @param softQValueFunctionEstimator reference to soft Q value function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException        throws exception if state action value function is applied to non-updateable policy.
     */
    public UpdateableSoftQPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, String params, SoftQValueFunctionEstimator softQValueFunctionEstimator) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator, params);
        this.softQValueFunctionEstimator = softQValueFunctionEstimator;
        this.softQAlphaMatrix = softQValueFunctionEstimator.getSoftQAlphaMatrix();
        this.softQAlphaMatrix.setValue(0, 0, 0, softQAlpha);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        autoSoftAlpha = true;
        softQAlpha = 0.25;
        softQAlphaVerboseInterval = 25;
    }

    /**
     * Returns parameters used for updateable soft Q policy.
     *
     * @return parameters used for updateable soft Q policy.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + UpdateableSoftQPolicy.paramNameTypes;
    }

    /**
     * Sets parameters used for updateable soft Q policy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - softQAlpha; entropy regularization coefficient. Default value 1.<br>
     *     - autoSoftAlpha; if true alpha is adjusted automatically otherwise not. Default value true.<br>
     *     - softQAlphaVerboseInterval; verbose internal for alpha. Default value 25.<br>
     *
     * @param params parameters used for updateable soft Q policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("softQAlpha")) softQAlpha = params.getValueAsDouble("softQAlpha");
        if (params.hasParam("autoSoftAlpha")) autoSoftAlpha = params.getValueAsBoolean("autoSoftAlpha");
        if (params.hasParam("softQAlphaVerboseInterval")) softQAlphaVerboseInterval = params.getValueAsInteger("softQAlphaVerboseInterval");
    }

    /**
     * Returns reference to policy.
     *
     * @param valueFunctionEstimator value function estimator.
     * @return reference to policy.
     * @throws IOException throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference(FunctionEstimator valueFunctionEstimator) throws DynamicParamException, AgentException, MatrixException, IOException, ClassNotFoundException {
        return new UpdateableSoftQPolicy(executablePolicy.getExecutablePolicyType(), getFunctionEstimator().reference(), params, softQValueFunctionEstimator);
    }

    /**
     * Returns reference to policy.
     *
     * @param valueFunctionEstimator        value function estimator.
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedMemory                  if true shared memory is used between estimators.
     * @return reference to policy.
     * @throws IOException            throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     * @throws AgentException         throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference(FunctionEstimator valueFunctionEstimator, boolean sharedPolicyFunctionEstimator, boolean sharedMemory) throws DynamicParamException, AgentException, MatrixException, IOException, ClassNotFoundException {
        return new UpdateableSoftQPolicy(executablePolicy.getExecutablePolicyType(), sharedPolicyFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(sharedMemory), params, softQValueFunctionEstimator);
    }

    /**
     * Returns reference to policy.
     *
     * @param valueFunctionEstimator        reference to value function estimator.
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param memory                  if true policy will use shared memory otherwise dedicated memory.
     * @return reference to policy.
     * @throws IOException            throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     * @throws AgentException         throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference(FunctionEstimator valueFunctionEstimator, boolean sharedPolicyFunctionEstimator, Memory memory) throws DynamicParamException, IOException, ClassNotFoundException, AgentException, MatrixException {
        return new UpdateableSoftQPolicy(executablePolicy.getExecutablePolicyType(), sharedPolicyFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(memory), params, softQValueFunctionEstimator);
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
     * Returns true if alpha is adjusted automatically otherwise not.
     *
     * @return true if alpha is adjusted automatically otherwise not.
     */
    protected boolean isAutoSoftAlpha() {
        return autoSoftAlpha;
    }

    /**
     * Updates function estimator.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void updateFunctionEstimator() throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException, IOException, ClassNotFoundException {
        super.updateFunctionEstimator();
        if (isAutoSoftAlpha()) updateAlpha();
    }

    /**
     * Returns policy gradient value for update.
     *
     * @param state state.
     * @return policy gradient value.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    protected double getPolicyValue(State state) throws MatrixException, NeuralNetworkException {
        if (isAutoSoftAlpha()) incrementPolicyValues(state);
        return softQValueFunctionEstimator.getTargetValue(state) - getSoftAlpha() * Math.log(state.policyValue);
    }

    /**
     * Increments policy values.
     *
     * @param state state.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    private void incrementPolicyValues(State state) throws MatrixException, NeuralNetworkException {
        double stateEntropy = getValues(state).entropy();
        if (!Double.isNaN(stateEntropy) && !Double.isInfinite(stateEntropy)) {
            cumulativeAlphaLoss += stateEntropy;
            alphaLossCount++;
        }
    }

    /**
     * Updates alpha.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void updateAlpha() throws MatrixException, DynamicParamException {
        if (alphaLossCount == 0) return;
        // https://raw.githubusercontent.com/BY571/Deep-Reinforcement-Learning-Algorithm-Collection/master/ContinousControl/SAC.ipynb
        // self.target_entropy = -action_size  # -dim(A)
        // alpha_loss = - (self.log_alpha * (log_pis + self.target_entropy).detach()).mean()
        optimizer.optimize(softQAlphaMatrix, new DMatrix(softQAlpha * cumulativeAlphaLoss / (double) alphaLossCount));
        softQAlpha = softQAlphaMatrix.getValue(0,0, 0);
        cumulativeAlphaLoss = 0;
        alphaLossCount = 0;

        if (++softQAlphaVerboseCount == softQAlphaVerboseInterval && softQAlphaVerboseInterval > 0) {
            System.out.println("Soft Q alpha: " + softQAlpha);
            softQAlphaVerboseCount = 0;
        }
    }

}
