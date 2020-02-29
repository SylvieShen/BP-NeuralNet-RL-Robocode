package com.shenyue;

import robocode.*;
import java.io.*;

public class LUT implements LUTInterface {
	// States Arguments
	public static final int headingNum = 4;
	public static final int OpponentDistanceNum = 4;
	public static final int OpponentBearingNum = 4;
	public static final int XPositionNum = 8;
	public static final int YPositionNum = 6;
	public static final int statesNum = headingNum * OpponentDistanceNum * OpponentBearingNum * XPositionNum * YPositionNum;
	public static final int actionsNum = 5;
	public static int States[][][][][] = new int[headingNum][OpponentDistanceNum][OpponentBearingNum][XPositionNum][YPositionNum];

	public static int getHeading(double argValue) {
		// Four kinds of heading
		int heading = 0;
		if (argValue >= 0 && argValue < Math.PI / 2)
			heading = 0;
		if (argValue >= Math.PI / 2 && argValue < Math.PI)
			heading = 1;
		if (argValue >= Math.PI && argValue < Math.PI * 3 / 2)
			heading = 2;
		if (argValue >= Math.PI * 3 / 2)
			heading = 3;
		return heading;
	}

	public static int getOpponentDistance(double argValue) {
		// Four kinds of distance: close, near, far, very far
		int distance = (int) (argValue / 100);
		if (distance > OpponentDistanceNum - 1)
			distance = OpponentDistanceNum - 1;
		return distance;
	}

	public static int getOpponentBearing(double argValue) {
		// Four kinds of bearing
		int bearing = 0;
		argValue = argValue + Math.PI;
		if (argValue >= 0 && argValue < Math.PI / 2)
			bearing = 0;
		if (argValue >= Math.PI / 2 && argValue < Math.PI)
			bearing = 1;
		if (argValue >= Math.PI && argValue < Math.PI * 3 / 2)
			bearing = 2;
		if (argValue >= Math.PI * 3 / 2)
			bearing = 3;
		return bearing;
	}

	public static int getXPosition(double argValue) {
		int x = (int) (argValue / 100);
		if (x >= 0 && x < 1)
			x = 0;
		if (x >= 1 && x < 2)
			x = 1;
		if (x >= 2 && x < 3)
			x = 2;
		if (x >= 3 && x < 4)
			x = 3;
		if (x >= 4 && x < 5)
			x = 4;
		if (x >= 5 && x < 6)
			x = 5;
		if (x >= 6 && x < 7)
			x = 6;
		if (x >= 7 && x <= 8)
			x = 7;
		return x;
	}

	public static int getYPosition(double argValue) {
		int y = (int) (argValue / 100);
		if (y >= 0 && y < 1)
			y = 0;
		if (y >= 1 && y < 2)
			y = 1;
		if (y >= 2 && y < 3)
			y = 2;
		if (y >= 3 && y < 4)
			y = 3;
		if (y >= 4 && y < 5)
			y = 4;
		if (y >= 5 && y < 6)
			y = 5;
		return y;
	}

	// Look Up Table
	public double[][] LUTable;

	public LUT() {
		LUTable = new double[statesNum][actionsNum];
		initialiseLUT(); // set all function values to be zero
	}

	@Override
	public void initialiseLUT() {
		// Initialize table to zero
		for (int i = 0; i < statesNum; i++)
			for (int j = 0; j < actionsNum; j++)
				LUTable[i][j] = 0.0;

		// Initialize States
		int count = 0;
		for (int a = 0; a < headingNum; a++)
			for (int b = 0; b < OpponentDistanceNum; b++)
				for (int c = 0; c < OpponentBearingNum; c++)
					for (int d = 0; d < XPositionNum; d++)
						for (int e = 0; e < YPositionNum; e++)
							States[a][b][c][d][e] = count++;
	}

	public double getMaxQ(int state) {
		double maxQ = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < LUTable[state].length; i++) {
			if (LUTable[state][i] > maxQ)
				maxQ = LUTable[state][i];
		}
		return maxQ;
	}

	public int getBestAction(int state) {
		double maxQ = Double.NEGATIVE_INFINITY;
		int bestAction = 0;
		for (int i = 0; i < LUTable[state].length; i++) {
			if (LUTable[state][i] > maxQ) {
				maxQ = LUTable[state][i];
				bestAction = i;
			}
		}
		return bestAction;
	}

	public double getQValue(int state, int action) {
		return LUTable[state][action];
	}

	public void setQValue(int state, int action, double value) {
		LUTable[state][action] = value;
	}

	@Override
	public int indexFor(double[] X) {
		return 2;
	}

	@Override
	public double outputFor(double[] X) {
		return 2;
	}

	@Override
	public double train(double[] X, double argValue) {
		return 0;
	}

	@Override
	public void load(File file) {
		BufferedReader read = null;
		try {
			read = new BufferedReader(new FileReader(file));
			for (int i = 0; i < statesNum; i++)
				for (int j = 0; j < actionsNum; j++)
					LUTable[i][j] = Double.parseDouble(read.readLine());
		} catch (IOException e) {
			initialiseLUT();
		} catch (NumberFormatException e) {
			initialiseLUT();
		} finally {
			try {
				if (read != null)
					read.close();
			} catch (IOException e) {

			}
		}
	}

	@Override
	public void save(File file) {
		PrintStream write = null;
		try {
			write = new PrintStream(new RobocodeFileOutputStream(file));
			for (int i = 0; i < statesNum; i++)
				for (int j = 0; j < actionsNum; j++)
					write.println(LUTable[i][j]);

			if (write.checkError())
				System.out.println("Could not save the data!");
			write.close();
		} catch (IOException e) {

		} finally {
			try {
				if (write != null)
					write.close();
			} catch (Exception e) {

			}
		}
	}

}