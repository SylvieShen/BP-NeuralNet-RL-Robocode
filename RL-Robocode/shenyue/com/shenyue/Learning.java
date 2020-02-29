package com.shenyue;

import java.util.Random;

public class Learning {
	public static final double learningRate = 0.1;
	public static final double discountRate = 0.9;
	public static double explorationRate = 0.0;
	private int previousState;
	private int previousAction;
	public LUT LUTable;

	public Learning(LUT LUTable) {
		this.LUTable = LUTable;
	}

	public void LUTlearning(int currentState, int currentAction, double reward, boolean offPolicy) {
		double previousQ = LUTable.getQValue(previousState, previousAction);
		if (offPolicy) {
			double currentQ = (1 - learningRate) * previousQ
					+ learningRate * (reward + discountRate * LUTable.getMaxQ(currentState));
			LUTable.setQValue(previousState, previousAction, currentQ);
		} else { // onPolicy
			double currentQ = (1 - learningRate) * previousQ
					+ learningRate * (reward + discountRate * LUTable.getQValue(currentState, currentAction));
			LUTable.setQValue(previousState, previousAction, currentQ);
		}
		previousState = currentState;
		previousAction = currentAction;
	}

	public int selectAction(int state) {
		double random = Math.random();
		if (explorationRate > random) {
			Random ran = new Random();
			return ran.nextInt(((LUT.actionsNum - 1 - 0) + 1));
		} else { // Pure greedy
			return LUTable.getBestAction(state);
		}
	}
}
