from gpiozero import Robot
from gpiozero import RGBLED
import pygame
import random

#R2-D2 class
class R2_D2():
	def __init__(self):
		#movement variables
		self.legs = Robot(left=(5, 6), right=(17, 27))
		self.turning_right = False
		self.turning_left = False
		self.moving_forward = False
		self.moving_backward = False

        #light variables
        #used to blink red-blue light
        self.light = RGBLED(red=16, green=20, blue=21)
        self.light.color = (1, 0, 0)
        self.red = True
        self.time_since_last_blink = 0

	#function calls Robot.right() if not already moving right
	def turn_right(self):
		if(not self.turning_right):
			self.legs.right()
			self.turning_right = True

	#funtion calls Robot.left() if not already moving left
	def turn_left(self):
		if(not self.turning_left):
			self.legs.left()
			self.turning_left = True

	#function calls Robot.forward() if not already moving forward
	def move_forward(self):
		if(not self.moving_forward):
			self.legs.forward()
			self.moving_forward = True

	#function calls Robot.backward() if not already moving backward
	def move_backward(self):
		if(not self.moving_backward):
			self.legs.backward()
			self.moving_backward = True

	#functions stops all robot movements, sets all movement variables to false
	def stop_movement(self):
		self.turning_right = False
		self.turning_left = False
		self.moving_forward = False
		self.moving_backward = False
		moving = False

		keys = pygame.key.get_pressed()
		if(keys[pygame.K_UP]):
			self.move_forward()
			moving = True
		elif(keys[pygame.K_RIGHT]):
			self.turn_right()
			moving = True
		elif(keys[pygame.K_DOWN]):
			self.move_backward()
			moving = True
		elif(keys[pygame.K_LEFT]):
			self.turn_left()
			moving = True

		if(not moving):
			self.legs.stop()

	#plays a sound based on given input, does nothing if input invalid
	#assumes pygame is initialized 
	def play_sound(self, sound_name):
		sound_location = './sounds/' + sound_name + '.wav'
		sound = pygame.mixer.Sound(sound_location)
		pygame.mixer.Sound.play(sound)

    #updates all time dependent component of R2-D2
    #expected delta_time in seconds
    def update(self, delta_time):
        #update light
        self.update_light(delta_time)

    #keeps track of time since last light blink and blinks light
    #if that time is greater than 1 second. 
    #delta time parameter is in seconds
    def update_light(self, delta_time):
        self.time_since_last_blink = self.time_since_last_blink + delta_time

        #blink about every 1 second
        if(self.time_since_last_blink >= 1.0):
            self.time_since_last_blink = 0
            if(self.red):
                self.red = False
                self.light.color = (0, 0, 1)
            else:
                self.red = True
                self.light.color = (1, 0, 0)

#create an artoo
artoo = R2_D2()

#intialize dummy screen to make keybaord input possible
screen = pygame.display.set_mode((50,50))

#initiate pygame loop
pygame.init()
running = True
while(running):
	#control artoo with the arrow keys
	events = pygame.event.get()
	for event in events:
		if event.type == pygame.KEYDOWN:
			#movement
			if event.key == pygame.K_LEFT:
				artoo.turn_left()
			if event.key == pygame.K_RIGHT:
				artoo.turn_right()
			if event.key == pygame.K_UP:
				artoo.move_forward()
			if event.key == pygame.K_DOWN:
				artoo.move_backward()
			#sounds
			if event.key == pygame.K_h:
				artoo.play_sound('helpme_short')
			if event.key == pygame.K_s:
				artoo.play_sound('scream')
			if event.key == pygame.K_e:
				artoo.play_sound('yes')
            if event.key == pygame.K_c:
                randomChatter = random.randint(1,8)
                chatterName = 'chatter' + str(randomChatter)
                artoo.play_sound(chatterName)
            if event.key == pygame.K_l:
                artoo.play_sound('helpme_long')
            if event.key == pygame.K_u:
                artoo.play_sound('unhappy')

 			#quit by pressing q key
			if event.key == pygame.K_q:
				running = False

		if event.type == pygame.KEYUP: 
			if event.key == pygame.K_LEFT:
				artoo.stop_movement()
			if event.key == pygame.K_RIGHT:
				artoo.stop_movement()
			if event.key == pygame.K_UP:
			  	artoo.stop_movement()
			if event.key == pygame.K_DOWN:
			  	artoo.stop_movement()

pygame.quit()







