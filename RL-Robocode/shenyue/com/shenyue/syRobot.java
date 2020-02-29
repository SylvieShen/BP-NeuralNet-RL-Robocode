package com.shenyue;

import java.awt.*;
import java.awt.geom.*;
import java.io.IOException;
import java.io.PrintStream;

import robocode.*;

public class syRobot extends AdvancedRobot {
  // Action
  public static final int moveAhead = 0;
  public static final int moveBack = 1;
  public static final int turnLeft = 2;
  public static final int turnRight = 3;
  public static final int robotfire = 4;
  public static final double aheadDistance = 150.0;
  public static final double backDistance = 100.0;
  public static final double turnDegree = 20.0;
  public static final int actionsNum = 5;

  // Opponent
  public String opponentName;
  public double opponentSpeed;
  public double opponentBearing;
  public long opponentTime;
  public double opponentX;
  public double opponentY;
  public double opponentDistance;
  public double opponentHead;
  public double opponentChangehead;
  public double opponentEnergy;

  public Learning learning;
  private double firePower;
  public static int count;
  public static int winCount;
  public static double reward;
  public static double winningRates;

  public static boolean intermediate = true; // false for only terminal reward
  public static boolean offPolicy = true; // false for on-Policy

  public double goodReward = 5;
  public double badReward = -5;
  


  public void run() {
    learning = new Learning(new LUT());
    loadTable();
    opponentDistance = 1000; // Set Opponent distance to 'far'
    // Set my syRobot
    setColors(Color.white, Color.pink, Color.pink);
    setAdjustGunForRobotTurn(true);
    setAdjustRadarForGunTurn(true);
    turnRadarRightRadians(2 * Math.PI);

    while (true) {
      int state = getState();
      int action = learning.selectAction(state);
      learning.LUTlearning(state, action, reward, offPolicy);
      reward = 0.0;
      switch (action) {
      case moveAhead:
        setAhead(aheadDistance);
        break;
      case moveBack:
        setBack(backDistance);
        break;
      case turnLeft:
        setTurnLeft(turnDegree);
        break;
      case turnRight:
        setTurnRight(turnDegree);
        break;
      case robotfire:
        fire(1);
        break;
      }
      radarAction();
      gunAction(2);
      execute();
    }
  }

  private int getState() {
    int heading = LUT.getHeading(getHeading());
    int oppoDistance = LUT.getOpponentDistance(opponentDistance);
    int oppoBearing = LUT.getOpponentBearing(opponentBearing);
    int x = LUT.getXPosition(getX());
    int y = LUT.getYPosition(getY());
    int state = LUT.States[heading][oppoDistance][oppoBearing][x][y];
    return state;
  }

  private void radarAction() {
    double radarRotate;
    if (getTime() - opponentTime > 4) {
      radarRotate = 4 * Math.PI; // Rotate radar to find an opponent
    } else {
      radarRotate = getRadarHeadingRadians() - (Math.PI / 2 - Math.atan2(opponentY - getY(), opponentX - getX()));
      radarRotate = nomslizeBearing(radarRotate);
      if (radarRotate < 0)
        radarRotate -= Math.PI / 10;
      else
        radarRotate += Math.PI / 10;
    }
    setTurnRadarLeftRadians(radarRotate);
  }

  private void gunAction(double power) {
    long currentTime;
    long nextTime;
    Point2D.Double opponentPosition = new Point2D.Double(opponentX, opponentY);
    // Distance between my robot and opponent
    double distance = Math.sqrt((opponentPosition.x - getX()) * (opponentPosition.x - getX())
        + (opponentPosition.y - getY()) * (opponentPosition.y - getY()));
    for (int i = 0; i < 20; i++) {
      // Calculate time to reach the opponent
      nextTime = (int) Math.round(distance / (20 - 3 * firePower));
      currentTime = getTime() + nextTime;
      opponentPosition = guessPosition(currentTime);
    }
    // Set off the gun
    double gunOffSet = getGunHeadingRadians()
        - (Math.PI / 2 - Math.atan2(opponentPosition.y - getY(), opponentPosition.x - getX()));
    setTurnGunLeftRadians(nomslizeBearing(gunOffSet));
    if (getGunHeat() == 0) {
      setFire(power);
    }
  }

  public void onScannedRobot(ScannedRobotEvent e) {
    if ((e.getDistance() < opponentDistance) || (opponentName == e.getName())) {
      double absBearing = (getHeadingRadians() + e.getBearingRadians()) % (2 * Math.PI);
      opponentName = e.getName();
      double head = nomslizeBearing(e.getHeadingRadians() - opponentHead);
      head = head / (getTime() - opponentTime);
      opponentChangehead = head;
      opponentX = getX() + Math.sin(absBearing) * e.getDistance();
      opponentY = getY() + Math.cos(absBearing) * e.getDistance();
      opponentBearing = e.getBearingRadians();
      opponentHead = e.getHeadingRadians();
      opponentTime = getTime();
      opponentSpeed = e.getVelocity();
      opponentDistance = e.getDistance();
      opponentEnergy = e.getEnergy();
    }
  }

  public Point2D.Double guessPosition(long time) {
    double newX, newY;
    if (Math.abs(opponentChangehead) > 0.00001) {
      double radius = opponentSpeed / opponentChangehead;
      double totalHead = (time - opponentTime) * opponentChangehead;
      newX = opponentX + (Math.cos(opponentHead) * radius) - (Math.cos(opponentHead + totalHead) * radius);
      newY = opponentY + (Math.sin(opponentHead + totalHead) * radius) - (Math.sin(opponentHead) * radius);
    } else {
      newX = opponentX + Math.sin(opponentHead) * opponentSpeed * (time - opponentTime);
      newY = opponentY + Math.cos(opponentHead) * opponentSpeed * (time - opponentTime);
    }
    return new Point2D.Double(newX, newY);
  }

  double nomslizeBearing(double argValue) {
    if (argValue > Math.PI)
      argValue -= 2 * Math.PI;
    if (argValue < -Math.PI)
      argValue += 2 * Math.PI;
    return argValue;
  }

  double nomalizeHeading(double argValue) {
    if (argValue > 2 * Math.PI)
      argValue -= 2 * Math.PI;
    if (argValue < 0)
      argValue += 2 * Math.PI;
    return argValue;
  }

  public void onBulletHit(BulletHitEvent e) {
    if (intermediate) {
      if (opponentName == e.getName()) {
        reward += 0.2;
      }
    }
  }

  public void onBulletMissed(BulletMissedEvent e) {
    if (intermediate) {
      reward -= 0.2;
    }

  }

  public void onHitByBullet(HitByBulletEvent e) {
    if (intermediate) {
      if (opponentName == e.getName()) {
        reward -= 0.2;
      }
    }
  }

  public void onHitWall(HitWallEvent e) {
    if (intermediate) {
      reward -= 0.1;
    }
  }

  public void onRobotDeath(RobotDeathEvent e) {
    if (e.getName() == opponentName)
      opponentDistance = 1000;
  }

  // Terminal rewards
  public void onWin(WinEvent event) {
    saveTable();
    reward += goodReward;
    count += 1;
    winCount += 1;
    PrintStream file = null;
    try {
      file = new PrintStream(new RobocodeFileOutputStream(getDataFile("winning-rates.dat").getAbsolutePath(), true));
      if (count == 19) {
        winningRates = (double) (winCount) / 20;
        file.println(winningRates);
        reward = 0;
        winCount = 0;
        count = 0;
        if (file.checkError())
          System.out.println("Save Error!");
        file.close();
      }
    } catch (IOException e) {
      System.out.println(e);
    } finally {
      try {
        if (file != null)
          file.close();
      } catch (Exception e) {
        System.out.println(e);
      }
    }
  }

  public void onDeath(DeathEvent event) {
    saveTable();
    reward += badReward;
    count += 1;
    PrintStream file = null;
    try {
      file = new PrintStream(new RobocodeFileOutputStream(getDataFile("winning-rates.dat").getAbsolutePath(), true));
      if (count == 19) {
        winningRates = (double) (winCount) / 20;
        file.println(winningRates);
        reward = 0;
        winCount = 0;
        count = 0;
        if (file.checkError())
          System.out.println("Save Error!");
        file.close();
      }
    } catch (IOException e) {
      System.out.println(e);
    } finally {
      try {
        if (file != null)
          file.close();
      } catch (Exception e) {
        System.out.println(e);
      }
    }
  }

  public void loadTable() {
    try {
      learning.LUTable.load(getDataFile("LUT.dat"));
    } catch (Exception e) {
      out.println("Load Error!" + e);
    }
  }

  public void saveTable() {
    try {
      learning.LUTable.save(getDataFile("LUT.dat"));
    } catch (Exception e) {
      out.println("Save Error!" + e);
    }
  }
}
