from gpiozero import Robot
from gpiozero import RGBLED
import pygame

#R2-D2 class
class R2_D2():
	def __init__(self):
		#movement variables
		self.legs = Robot(left=(5, 6), right=(17, 27))
		self.turning_right = False
		self.turning_left = False
		self.moving_forward = False
		self.moving_backward = False

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


	#initializes R2's sounds
	#call only after pygame.init has been called
	def init_sounds(self):
		self.scream_sound = pygame.mixer.Sound('./sounds/scream.wav')
		self.helpme_long_sound = pygame.mixer.Sound('./sounds/helpme_long.wav')
		self.helpme_short_sound = pygame.mixer.Sound('./sounds/helpme_short.wav')
		self.chatter1_sound = pygame.mixer.Sound('./sounds/chatter1.wav')
		self.chatter2_sound = pygame.mixer.Sound('./sounds/chatter2.wav')
		self.chatter3_sound = pygame.mixer.Sound('./sounds/chatter3.wav')
		self.chatter4_sound = pygame.mixer.Sound('./sounds/chatter4.wav')
		self.chatter5_sound = pygame.mixer.Sound('./sounds/chatter5.wav')
		self.chatter6_sound = pygame.mixer.Sound('./sounds/chatter6.wav')
		self.chatter6_sound = pygame.mixer.Sound('./sounds/chatter7.wav')
		self.cute_sound = pygame.mixer.Sound('./sounds/cute.wav')
		self.excited_sound = pygame.mixer.Sound('./sounds/excited.wav')
		self.insistent_sound = pygame.mixer.Sound('./sounds/insistent.wav')
		self.laughter_sound = pygame.mixer.Sound('./sounds/laughter.wav')
		self.no_sound = pygame.mixer.Sound('./sounds/no.wav')
		self.processing_sound = pygame.mixer.Sound('./sounds/processing.wav')
		self.reallysad_sound = pygame.mixer.Sound('./sounds/reallysad.wav')
		self.sad_sound = pygame.mixer.Sound('./sounds/sad.wav')
		self.searching_whistle_sound = pygame.mixer.Sound('./sounds/searching_whistle.wav')
		self.unhappy_sound = pygame.mixer.Sound('./sounds/unhappy.wav')
		self.warning_sound = pygame.mixer.Sound('./sounds/warning.wav')
		self.yes_sound = pygame.mixer.Sound('./sounds/yes.wav')

	#plays a sound based on given input, does nothing if input invalid
	#assumes pygame is initialized 
	def play_sound(self, sound_name):
		sound_location = './sounds/' + sound_name + '.wav'
		sound = pygame.mixer.Sound(sound_location)
		pygame.mixer.Sound.play(sound)

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
				artoo.play_sound('scream1')
			if event.key == pygame.K_e:
				artoo.play_sound('excited')
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







