import numpy as np
import math
import random

from Objects.robot import Robot

FIRE_DISTANCE = 1000
BULLET_POWER = 2

class DarkEmperor(Robot):

    alpha = 0.1
    gamma = 0.6
    epsilon = 0.3
    ctr = 1
    num_states = 7
    num_actions = 4
    last_state = 0
    reward = 100
    state = 6
    q_table = np.zeros((num_states, num_actions))

    def init(self):
        self.setColor(72,209,204)
        self.setGunColor(200, 200, 0)
        self.setRadarColor(255, 60, 0)
        self.setBulletsColor(0, 200, 100)
        

        size = self.getMapSize()
        self.radarVisible(True)

        self.lockRadar("gun")
        self.setRadarField("thin")
        self.loadQTable()

    def run(self):
        
        
        self.gunTurn(90)

        state = self.state
        action = self.selectAction(state)
        self.performAction(action)
        self.updateQValue(state, action, self.reward)

        self.ctr = self.ctr + 1
        if self.ctr % 2 == 0:
            self.saveQTable()

    def sensors(self):
        pass

    def onHitWall(self):
        if self.reward > 0 and self.reward - 5 >= 0:
            self.reward = self.reward - 5
        else:
            self.reward = self.reward - 0
        
        self.state = 0
        self.move(-random.randrange(10,20))

    def sensors(self): 
        pass
        
    def onRobotHit(self, robotId, robotName):
        if self.reward > 0 and self.reward - 5 >= 0:
            self.reward = self.reward - 5
        else:
            self.reward = self.reward - 0
        
        self.turn(random.randrange(1,50))
        self.move(random.randrange(1,20))

        self.state = 1
        
    def onHitByRobot(self, robotId, robotName):
        if self.reward > 0 and self.reward - 5 >= 0:
            self.reward = self.reward - 5
        else:
            self.reward = self.reward - 0

        self.turn(random.randrange(1,50))
        self.move(random.randrange(1,20))    
        
        self.state = 2


    def onHitByBullet(self, bulletBotId, bulletBotName, bulletPower):
        if self.reward > 0 and self.reward - 5 >= 0:
            self.reward = self.reward - 5
        else:
            self.reward = self.reward - 0

        self.turn(random.randrange(1,50))
        self.move(random.randrange(1,20))
        
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


    def updateQValue(self, state, action, reward):
        old_value = self.q_table[self.last_state][action]
        next_max = np.max(self.q_table[state][action])
        
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        
        if new_value < 10:
            new_value = new_value/10
        elif new_value > 10:
            new_value = new_value/100
        elif new_value > 100:
            new_value = new_value/1000
        
        self.q_table[state][action] = new_value


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
        self.q_table = np.loadtxt('qtable.txt', dtype=float)
