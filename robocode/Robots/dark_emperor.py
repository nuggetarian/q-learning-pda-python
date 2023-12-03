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
    num_states = 7
    num_actions = 4
    last_state = 0
    reward = 100
    state = 6

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
        self.loadQTable()

    def run(self):
        
        
        self.gunTurn(90)

        state = self.state
        state = self.state
        action = self.selectAction(state)
        self.performAction(action)

        print(state)
        print(action)

        # # TODO
        # reward = self.getReward(state, action)  
        print("Reward:" + str(self.reward))     
        self.updateQValue(state, action, self.reward)

        self.ctr = self.ctr + 1
        if self.ctr % 2 == 0:
            print("Saving QTable")
            self.saveQTable()

    def sensors(self):  #NECESARY FOR THE GAME
        pass

    def onHitWall(self):
        if self.reward > 0 and self.reward - 5 >= 0:
            self.reward = self.reward - 5
        else:
            self.reward = self.reward - 0
        
        self.state = 0
        #self.move(-20)
        self.move(-random.randrange(10,20))

    def sensors(self): 
        pass
        
    def onRobotHit(self, robotId, robotName):
        if self.reward > 0 and self.reward - 5 >= 0:
            self.reward = self.reward - 5
        else:
            self.reward = self.reward - 0

        self.state = 1
        
    def onHitByRobot(self, robotId, robotName):
        if self.reward > 0 and self.reward - 5 >= 0:
            self.reward = self.reward - 5
        else:
            self.reward = self.reward - 0
        
        self.state = 2


    def onHitByBullet(self, bulletBotId, bulletBotName, bulletPower):
        if self.reward > 0 and self.reward - 5 >= 0:
            self.reward = self.reward - 5
        else:
            self.reward = self.reward - 0
        
        self.state = 3
        
    
    def onBulletHit(self, botId, bulletId):
        if self.reward < 100 and self.reward + 10 <= 100:
            self.reward = self.reward + 10
        else:
            self.reward = self.reward + 0
        
        self.state = 4
        
    def onBulletMiss(self, bulletId):
        if self.reward > 0 and self.reward - 5 >= 0:
            self.reward = self.reward - 5
        else:
            self.reward = self.reward - 0
        
        self.state = 5
        
    def onRobotDeath(self):
        if self.reward > 0 and self.reward - 10 >= 0:
            self.reward = self.reward - 10
        else:
            self.reward = self.reward - 0
        
        
    def onTargetSpotted(self, botId, botName, botPos):
        if self.reward < 100 and self.reward + 5 <= 100:
            self.reward = self.reward + 5
        else:
            self.reward = self.reward + 0
        
        self.state = 6

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
        return 1

    # TODO sposobuje error.
    def updateQValue(self, state, action, reward):
        # print("USING OLD VALUE")
        # print("This is the last state:" + str(self.last_state) + "\nThis is the action:" + str(action))
        old_value = self.q_table[self.last_state][action]
        next_max = np.max(self.q_table[state][action])
        
        # new_value = ((1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max))/100
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)

        if new_value < 10:
            new_value = new_value/10
        elif new_value > 10:
            new_value = new_value/100

        self.q_table[state][action] = new_value

        # old_value = self.q_table[0][1]
        # next_max = np.max(self.q_table[0][1])
        
        # new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        # self.q_table[0][1] = new_value

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
            return action
        else:
            best_action = np.argmax(self.q_table[state])
            return best_action
        

    def saveQTable(self):
        with open('qtable.txt', 'w') as file:
            for row in self.q_table:
                row_data = ' '.join(str(element) for element in row)
                file.write(row_data + '\n')
    
    def loadQTable(self):
        # loaded_array = np.zeros((self.num_states, self.num_actions))
        # with open('qtable.txt', 'r') as file:
        #     for line in file:
        #         row = np.array([float(element) if '.' in element else int(element) for element in line.strip().split()])
        #         self.q_table = np.vstack((self.q_table, row))

        self.q_table = np.loadtxt('qtable.txt', dtype=float)
        
        # self.q_table = loaded_array
        print(self.q_table)
