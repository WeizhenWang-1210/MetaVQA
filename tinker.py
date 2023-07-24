import pygame
import sys
 
pygame.init()
screen = pygame.display.set_mode((400, 400))
clock = pygame.time.Clock()
 
dialogue_font = pygame.font.SysFont('arial', 15)
name_font = pygame.font.SysFont('Helvetica', 20)
game_over_font = pygame.font.SysFont('Verdana', 60)
#<class 'pygame.font.Font'>
dialogue = dialogue_font.render("Hey there, Beautfiul weather today!",
                                True, (0,0,0))
print(type(dialogue_font))
name = name_font.render("John Hubbard", True, (0,0,255))
game_over = game_over_font.render("Game Over", True, (255,0,0))
 
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
     
    screen.fill((255, 255, 255))
     
    screen.blit(dialogue, (40,40))
    screen.blit(name, (40,140))
    screen.blit(game_over, (40,240))
     
    pygame.display.flip()
    clock.tick(60)