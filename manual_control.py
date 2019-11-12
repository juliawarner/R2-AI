from gpiozero import Robot
from gpiozero import RGBLED
import pygame

#R2-D2 class
class R2_D2():
	def __init__(self):
		#movement variables
		legs = Robot(left=(5, 6), right=(17, 27))
		turning_right = False
		turning_left = False
		moving_forward = False
		moving_backward = False

	#function calls Robot.right() if not already moving right
	def turn_right(self):
		if(not turning_right):
			self.legs.right()
			self.turning_right = True

	#funtion calls Robot.left() if not already moving left
	def turn_left(self):
		if(not turning_left):
			self.legs.left()
			self.turning_left = True

	#function calls Robot.forward() if not already moving forward
	def move_forward(self):
		if(not moving_forward):
			self.legs.forward()
			self.moving_forward = True

	#function calls Robot.backward() if not already moving backward
	def move_backward(self):
		if(not moving_backward):
			self.legs.backward()
			self.moving_backward = True

	#functions stops all robot movements, sets all movement variables to false
	def stop_movement(self):
		self.legs.stop()
		self.turning_right = False
		self.turning_left = False
		self.moving_forward = False
		self.moving_backward = False

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
			if event.key == pygame.K_LEFT:
				artoo.turn_left()
			if event.key == pygame.K_RIGHT:
				artoo.turn_right()
			if event.key == pygame.K_UP:
				artoo.move_forward()
			if event.key == pygame.K_DOWN:
				artoo.move_backward()
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







