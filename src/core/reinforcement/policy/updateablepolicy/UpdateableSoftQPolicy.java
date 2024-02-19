/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy.updateablepolicy;

import core.network.NeuralNetworkException;
import core.optimization.Optimizer;
import core.optimization.OptimizerFactory;
import core.reinforcement.agent.State;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.value.SoftQValueFunction;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.io.IOException;

/**
 * Implements updateable soft Q policy.<br>
 *
 */
public class UpdateableSoftQPolicy extends AbstractUpdateablePolicy {

    /**
     * Parameter name types for updateable soft Q policy.
     *     - softQAlpha; entropy regularization coefficient. Default value 0.2.<br>
     *     - autoSoftAlpha; if true alpha is adjusted automatically otherwise not. Default value true.<br>
     *     - softQAlphaVerboseInterval; verbose internal for alpha. Default value 25.<br>
     *
     */
    private final static String paramNameTypes = "(softQAlpha:DOUBLE), " +
            "(autoSoftAlpha:BOOLEAN), " +
            "(softQAlphaVerboseInterval:INT)";

    /**
     * If true alpha is adjusted automatically otherwise not.
     *
     */
    private boolean autoSoftAlpha;

    /**
     * Alpha parameter for entropy control.
     *
     */
    private double softQAlpha;

    /**
     * Alpha parameter for entropy in matrix form.
     *
     */
    private Matrix softQAlphaMatrix;

    /**
     * Log of alpha parameter for entropy control.
     *
     */
    private double softQAlphaLog;

    /**
     * Log of alpha parameter for entropy in matrix form.
     *
     */
    private Matrix softQAlphaLogMatrix;

    /**
     * Cumulative entropy value.
     *
     */
    private transient double cumulativeAlphaLoss;

    /**
     * Cumulative entropy value count.
     *
     */
    private transient double cumulativeAlphaLossCount;

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
    private SoftQValueFunction softQValueFunction;

    /**
     * Target entropy.
     *
     */
    private final double targetEntropy;

    /**
     * Constructor for updateable soft Q policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator    reference to function estimator.
     * @param memory               reference to memory.
     * @param params               parameters for updateable soft Q policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public UpdateableSoftQPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, Memory memory, String params) throws DynamicParamException, MatrixException {
        super(executablePolicyType, functionEstimator, memory, params);
        targetEntropy = -0.98 * (Math.log (1 / (double)functionEstimator.getNumberOfActions()));
    }

    /**
     * Sets soft Q value function.
     *
     * @param softQValueFunction soft Q value function.
     */
    public void setSoftQValueFunction(SoftQValueFunction softQValueFunction) {
        this.softQValueFunction = softQValueFunction;
    }

    /**
     * Returns soft Q value function.
     *
     * @return soft Q value function.
     */
    private SoftQValueFunction getSoftQValueFunction() {
        return softQValueFunction;
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        autoSoftAlpha = true;
        softQAlpha = 0.9;
        softQAlphaMatrix = new DMatrix(0);
        softQAlphaMatrix.setValue(0, 0, 0, softQAlpha);
        softQAlphaLog = Math.log(softQAlpha);
        softQAlphaLogMatrix = new DMatrix(0);
        softQAlphaLogMatrix.setValue(0, 0, 0, softQAlphaLog);
        softQAlphaVerboseInterval = 25;
        cumulativeAlphaLoss = 0;
        cumulativeAlphaLossCount = 0;
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
     *     - softQAlpha; entropy regularization coefficient. Default value 0.2.<br>
     *     - autoSoftAlpha; if true alpha is adjusted automatically otherwise not. Default value true.<br>
     *     - softQAlphaVerboseInterval; verbose internal for alpha. Default value 25.<br>
     *
     * @param params parameters used for updateable soft Q policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("softQAlpha")) {
            softQAlpha = Math.log(params.getValueAsDouble("softQAlpha"));
            softQAlphaMatrix.setValue(0, 0, 0, softQAlpha);
            softQAlphaLog = Math.log(softQAlpha);
            softQAlphaLogMatrix.setValue(0, 0, 0, softQAlphaLog);
        }
        if (params.hasParam("autoSoftAlpha")) autoSoftAlpha = params.getValueAsBoolean("autoSoftAlpha");
        if (params.hasParam("softQAlphaVerboseInterval")) softQAlphaVerboseInterval = params.getValueAsInteger("softQAlphaVerboseInterval");
    }

    /**
     * Returns reference to policy.
     *
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param memory                        if true policy will use shared memory otherwise dedicated memory.
     * @return reference to policy.
     * @throws IOException            throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    public Policy reference(boolean sharedPolicyFunctionEstimator, Memory memory) throws DynamicParamException, IOException, ClassNotFoundException, MatrixException {
        return new UpdateableSoftQPolicy(executablePolicy.getExecutablePolicyType(), sharedPolicyFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(), memory, params);
    }

    /**
     * Returns reference to policy.
     *
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param memory                  if true policy will use shared memory otherwise dedicated memory.
     * @return reference to policy.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    public Policy reference(FunctionEstimator policyFunctionEstimator, Memory memory) throws DynamicParamException, MatrixException {
        return new UpdateableSoftQPolicy(executablePolicy.getExecutablePolicyType(), policyFunctionEstimator, memory, params);
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
     * Returns Soft Q Alpha Matrix.
     *
     * @return Soft Q Alpha Matrix.
     */
    public Matrix getSoftQAlphaMatrix() {
        return softQAlphaMatrix;
    }

    /**
     * Returns policy gradient value for update.
     *
     * @param state state.
     * @return policy gradient value.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    protected Matrix getPolicyGradient(State state) throws MatrixException, NeuralNetworkException, DynamicParamException {
        double policyValue = getFunctionEstimator().predictPolicyValues(state).getValue(state.action, 0, 0);

        if (isAutoSoftAlpha()) cumulateAlphaLoss(policyValue);

        double qValue = getSoftQValueFunction().getTargetValues(state, false).getValue(state.action, 0, 0);

        Matrix policyGradient = new DMatrix(getFunctionEstimator().getNumberOfActions(), 1, 1);
        double policyGradientValue = policyValue * (qValue - softQAlpha * Math.log(policyValue));
        policyGradient.setValue(state.action, 0, 0, policyGradientValue);
        return policyGradient;
    }

    /**
     * Cumulates entropy.
     *
     * @param policyValue policy value.
     */
    private void cumulateAlphaLoss(double policyValue) {
        // https://raw.githubusercontent.com/BY571/Deep-Reinforcement-Learning-Algorithm-Collection/master/ContinousControl/SAC.ipynb
        // self.target_entropy = -action_size  # -dim(A)
        // alpha_loss = - (self.log_alpha * (log_pis + self.target_entropy).detach()).mean()

        cumulativeAlphaLoss += -policyValue * softQAlphaLog * (Math.log(policyValue) + targetEntropy);
        cumulativeAlphaLossCount++;
    }

    /**
     * Updates alpha.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void updateAlpha() throws MatrixException, DynamicParamException {
        optimizer.optimize(softQAlphaLogMatrix, new DMatrix(cumulativeAlphaLoss / cumulativeAlphaLossCount));

        softQAlphaLog = softQAlphaLogMatrix.getValue(0,0, 0);

        softQAlpha = Math.exp(softQAlphaLog);
        softQAlphaMatrix.setValue(0, 0, 0, softQAlpha);

        cumulativeAlphaLoss = 0;
        cumulativeAlphaLossCount = 0;

        if (++softQAlphaVerboseCount == softQAlphaVerboseInterval && softQAlphaVerboseInterval > 0) {
            System.out.println("Soft Q alpha: " + softQAlpha);
            softQAlphaVerboseCount = 0;
        }
    }

    /**
     * Finishes function estimator update.
     *
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    public void finishFunctionEstimator() throws NeuralNetworkException, MatrixException, DynamicParamException {
        super.finishFunctionEstimator();
        if (isAutoSoftAlpha()) updateAlpha();
    }

}
