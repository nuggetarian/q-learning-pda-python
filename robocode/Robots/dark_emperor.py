import numpy as np
import math
import random

from Objects.robot import Robot
from Objects.graph import Graph

FIRE_DISTANCE = 500
BULLET_POWER = 2

class DarkEmperor(Robot): #Create a Robot

    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1
    ctr = 1
    num_states = 3
    num_actions = 4
    last_state = 0

    # Initialize the Q-table with zeros
    q_table = np.zeros((num_states, num_actions))

    def init(self):# NECESARY FOR THE GAME   To initialyse your robot
        #Set the bot color in RGB
        self.setColor(0, 200, 100)
        self.setGunColor(200, 200, 0)
        self.setRadarColor(255, 60, 0)
        self.setBulletsColor(0, 200, 100)
        
        #get the map size
        size = self.getMapSize() #get the map size
        self.radarVisible(True) # show the radarField

        self.lockRadar("gun")
        self.setRadarField("thin")

    def run(self):
        self.loadQTable()
        self.gunTurn(5)

        state = self.getState()
        action = self.selectAction(state)
        self.performAction(action)

        # TODO
        reward = self.getReward(state, action)
        self.updateQValue(state, action, reward)

        self.ctr = self.ctr + 1
        if self.ctr == 25:
            self.saveQTable()
        if self.ctr == 50:
            self.saveQTable()
        if self.ctr == 75:
            self.saveQTable()
        if self.ctr == 100:
            self.saveQTable()    

    def sensors(self):  #NECESARY FOR THE GAME
        pass

    def onHitWall(self):
        pass

    def sensors(self): 
        pass
        
    def onRobotHit(self, robotId, robotName):
        pass
        
    def onHitByRobot(self, robotId, robotName):
        pass

    def onHitByBullet(self, bulletBotId, bulletBotName, bulletPower):
        pass
        
    def onBulletHit(self, botId, bulletId):
        pass
        
    def onBulletMiss(self, bulletId):
        pass
        
    def onRobotDeath(self):
        pass
        
    def onTargetSpotted(self, botId, botName, botPos):
        pos = self.getPosition()
        dx = botPos.x() - pos.x()
        dy = botPos.y() - pos.y()

        my_gun_angle = self.getGunHeading() % 360
        enemy_angle = math.degrees(math.atan2(dy, dx)) - 90
        a = enemy_angle - my_gun_angle
        if a < -180:
            a += 360
        elif 180 < a:
            a -= 360
        self.gunTurn(a)

        dist = math.sqrt(dx**2 + dy**2)
        if dist < FIRE_DISTANCE:
            self.fire(BULLET_POWER)  

    # TODO
    def getState(self):
        state = 0

    # TODO, mozno spravit stylom, ze sa reward bude ukladat do vsetkych onbullethit, onbulletmiss... ale state neviem
    def getReward(self, state, action):
        pass

    def updateQValue(self, state, action, reward):
        old_value = self.q_table[self.last_state][action]
        next_max = np.max(self.q_table[state][action])
        
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value

        pass

    def performAction(self, action):
        if action == 0:
            self.move(100)
        elif action == 1:
            self.turn(-50)
        elif action == 2:
            self.turn(50)
        elif action == 3:
            self.move(-100)

    def selectAction(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, 3)
            self.last_state = state
            return action
        else:
            max_value = float("-inf")
            best_action = 0

            for action, value in enumerate(self.q_table[state]):
                if value > max_value:
                    max_value = value
                    best_action = action  # Update the best action index
                    self.last_state = state
            return best_action
        

    def saveQTable(self):
        with open('qtable.txt', 'w') as file:
            for row in self.q_table:
                row_data = ' '.join(str(element) for element in row)
                file.write(row_data + '\n')
    
    def loadQTable(self):
        loaded_array = np.zeros((self.num_states, self.num_actions))
        with open('qtable.txt', 'r') as file:
            for line in file:
                row = np.array([float(element) if '.' in element else int(element) for element in line.strip().split()])
                self.q_table = np.vstack((self.q_table, row))
        
        self.q_table = loaded_array