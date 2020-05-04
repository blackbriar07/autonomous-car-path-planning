# makin grids for the car to pass

import pygame
import math
import numpy as np

# Define some colors
BLACK    = (   0,   0,   0)
WHITE    = ( 255, 255, 255)
GREEN    = (   0, 200,   0)
RED      = ( 200,   0,   0)
bright_green = (0,255,0)
bright_red = (255,0,0)


# This sets the width and height of each grid location


# This sets the margin between each cell
margin = 0

# Create a 2 dimensional array. A two dimensional
# array is simply a list of lists.
grid = []
grids_startend = []
car_move_automatic = False



array2 = [0,0,"b"]

for row in range(10):
    # Add an empty array that will hold each cell
    # in this row   
    grid.append([])
    for column in range(10):
        grid[row].append(0) # Append a cell

for row in range(10):
    # Add an empty array that will hold each cell
    # in this row   
    grids_startend.append([])
    for column in range(10):
        grids_startend[row].append(0)
# Set row 1, cell 5 to one. (Remember rows and
# column numbers start at zero.)






# Initialize pygame
pygame.init()

# Set the height and width of the screen
num_rect_hor = 10 
num_rect_ver = 10
screen_width = 800
screen_height = 550 #480
width  = 48  # 60% of the whole screen width will be covered with the grids  
height = 48  
size = [screen_width, screen_height]
screen = pygame.display.set_mode(size)
thickness_rect = 1

# Set title of screen

position_vector = []
row_inc = 0
col_inc = 0

for pos_row in range(10):
    array_make =[]
    for pos_column in range(10):
        array_make.append([row_inc,col_inc])
        col_inc += width
    position_vector.append(array_make)
    row_inc += width
    col_inc = 0
#print("grid_vector" + str(len(position_vector)))


#Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

block_grid =[]

# Load and set up graphics.
#background_image = pygame.image.load("saturn_family1.jpg").convert()
player_image = pygame.image.load("playerShip1_orange.png").convert()
player_image.set_colorkey(BLACK)
image = pygame.transform.scale(player_image, (int(width-10),int(height-10)))

#image1 = pygame.transform.rotate(image, 0)
pos_rect = []
grid_rect = []
i_num = 0
for i in range(num_rect_hor):
        j_num = 0
        pos_rect = []
        for j in range(num_rect_ver):
            pos_rect.append([j_num,i_num])
            j_num += 48
        grid_rect.append(pos_rect)
        i_num += 48

def car(image1,x,y):
    screen.blit(image1,(x,y))

def text_objects(text, font):
    textSurface = font.render(text, True, BLACK)
    return textSurface, textSurface.get_rect()

def button(msg,x,y,w,h,ic,ac, action = None):

    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    #print(click)

    if x+w > mouse[0] > x and y+h >mouse[1] > y:
        pygame.draw.rect(screen,ac,(x,y,w,h))
        if click[0] == 1 and action != None:
            if action == "play":
                start_game()
            if action == "start_end":
                start_end_screen()
    else:
        pygame.draw.rect(screen,ic,(x,y,w,h))

    smallText = pygame.font.Font("freesansbold.ttf",20)
    textSurf, textRect = text_objects(msg,smallText)
    textRect.center = ((x + (w/2)), (y + (h/2)))
    screen.blit(textSurf, textRect)


def car(image1,x,y):
    screen.blit(image1,(x,y))


def car_movement_automatic(event,image1,x_car,y_car,x_carchange,y_carchange,car_move,direction_side,position):
    
    prev_image = image1
    prev_diection = direction_side

    
    if position[t-1][0] > position[t][0] and position[t-1][1] == position[t][1] : 
        image1 = pygame.transform.rotate(image,0)
        y_carchange = -car_move
        direction_side = "N"
    if position[t-1][0] < position[t][0] and position[t-1][1] == position[t][1] : 
        image1 = pygame.transform.rotate(image,180)
        y_carchange = -car_move
        direction_side = "S"
    if position[t-1][0] == position[t][0] and position[t-1][1] > position[t][1] : 
        image1 = pygame.transform.rotate(image,90)
        y_carchange = -car_move
        direction_side = "W"
    if position[t-1][0] == position[t][0] and position[t-1][1] < position[t][1] : 
        image1 = pygame.transform.rotate(image,-90)
        y_carchange = -car_move
        direction_side = "E"

    if block_cancel == True:
        for i in block_grid:
            if grid_rect[i[0]][i[1]][0] <=  x_car + x_carchange < grid_rect[i[0]][i[1]][0] + width and grid_rect[i[0]][i[1]][1] <=  y_car + y_carchange < grid_rect[i[0]][i[1]][1] + height:  
                x_carchange = 0
                y_carchange = 0
                image1 = prev_image
                direction_side = prev_direction
                

    x_car += x_carchange
    y_car += y_carchange

    if x_car < 0 or x_car >= width*10 or y_car < 0 or y_car >= height*10 :    
        x_car -= x_carchange
        y_car -= y_carchange

    return x_car , y_car , image1, direction_side
    
        
    



def car_movement_manual(event,image1,x_car,y_car,x_carchange,y_carchange,car_move,direction_side):

    prev_image = image1
    prev_direction = direction_side
    
    
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LEFT:
            image1 = pygame.transform.rotate(image,90)
            x_carchange = -car_move
            direction_side = "W"
        elif event.key == pygame.K_RIGHT:
            image1 = pygame.transform.rotate(image, -90)
            x_carchange = car_move
            direction_side = "E"
        elif event.key == pygame.K_UP:
            image1 = pygame.transform.rotate(image, 0)
            y_carchange = -car_move
            direction_side = "N"
        elif event.key == pygame.K_DOWN:
            image1 = pygame.transform.rotate(image, 180)
            y_carchange = car_move
            direction_side = "S"
        
    if event.type == pygame.KEYUP:
        #if event.key == pygame.K_LEFT or pygame.K_RIGHT or pygame.K_UP or pygame.K_DOWN:
            x_carchange = 0
            y_carchange = 0
            
            
    if block_cancel == True:
        for i in block_grid:
            if grid_rect[i[0]][i[1]][0] <=  x_car + x_carchange < grid_rect[i[0]][i[1]][0] + width and grid_rect[i[0]][i[1]][1] <=  y_car + y_carchange < grid_rect[i[0]][i[1]][1] + height:  
                x_carchange = 0
                y_carchange = 0
                image1 = prev_image
                direction_side = prev_direction
                



    x_car += x_carchange
    y_car += y_carchange

    if x_car < 0 or x_car >= width*10 or y_car < 0 or y_car >= height*10 :    
        x_car -= x_carchange
        y_car -= y_carchange

    
    return x_car , y_car , image1, direction_side




def grids():
  
    for row in range(num_rect_ver):
            for column in range(num_rect_hor):
                color = WHITE
                pygame.draw.rect(screen,
                                color,
                                [(margin+width)*column+margin,
                                (margin+height)*row+margin,
                                width,
                                height],thickness_rect)
                
                if grid[row][column] == 1:
                    color = RED
                    
                    pygame.draw.rect(screen,
                                    color,
                                    [(margin+width)*column+margin,
                                    (margin+height)*row+margin,
                                    width,
                                    height])


def grids_start_end():
  
    for row in range(num_rect_ver):
            for column in range(num_rect_hor):
                color = WHITE
                pygame.draw.rect(screen,
                                color,
                                [(margin+width)*column+margin,
                                (margin+height)*row+margin,
                                width,
                                height],thickness_rect)
                
                if grids_startend[row][column] == 1:
                    color = GREEN
                    
                    pygame.draw.rect(screen,
                                    color,
                                    [(margin+width)*column+margin,
                                    (margin+height)*row+margin,
                                    width,
                                    height])


def start_end_screen():

    start_end_exit = True
    global start_end_array 
    start_end_array = []

    while start_end_exit:
        screen.fill(BLACK)
        pygame.display.set_caption("Select start end points")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                start_end_exit = False
                pygame.quit()
                quit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                
                if 0 <= pos[0] <= num_rect_hor*width  and 0 <= pos[1] <= num_rect_ver*height: 
                    start_end_array.append(pos)
                    #print("Hello")
                    # Change the x/y screen coordinates to grid coordinates
                    column = pos[0] // (width + margin)
                    row = pos[1] // (height + margin)
                    #print(column,row)
                    # Set that location to zero

                    grids_startend[row][column] = 1
                    #print(grids_startend)
                    #block_cancel = True 
                    #block_grid.append([row,column])
                    #print("Click ", pos, "Grid coordinates: ", row, column)
        grids_start_end()

        grids()
        
        button("start game",550,50,150,50,GREEN,bright_green,"play")

        pygame.display.update()
        clock.tick(100)


def start_game():

    
    x_car = 0
    y_car = 0
    x_carchange = 0
    y_carchange = 0
    car_move = width
    direction_side = "stationary"
    image1 = pygame.transform.rotate(image,0)
    array2 = [0,0,"b"]
    movement_vector = np.array([[0,0],[0,1],[0,2]])
    
    gameExit = False
    print(start_end_array)

    while not gameExit:
        screen.fill(BLACK)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
            
            # declaring car movement to be manual or automatic. The reiforcement learning would be taking it in automatic mode
            if car_move_automatic == True :
                
                x_car,y_car,image1,direction_side = car_movement_automatic(event,image1,x_car,y_car,x_carchange,y_carchange,car_move,direction_side,movement_vector)
                print(x_car,y_car)
                   
            else:
                x_car,y_car,image1,direction_side = car_movement_manual(event,image1,x_car,y_car,x_carchange,y_carchange,car_move,direction_side)
            
        
        grids()
        grids_start_end()

        

        car(image1,x_car,y_car)
        array1 = [x_car // (width + margin) , y_car // (height + margin) , direction_side ]
        
        
        if array1 != array2 : 
        	print("state : (" + str(y_car // (height + margin)) + "," + str(x_car // (width + margin)) + "," + direction_side  +  ")" )
            
        array2 = array1
        
        button("game started" ,550,50,150,50,GREEN,bright_green,"haha")
        #aloo = aloo+1
        pygame.display.update()
        clock.tick(50)

    


def game_intro():
    intro =True
    global block_cancel 
    block_cancel = False

    while intro:
        screen.fill(BLACK)
        pygame.display.set_caption("Array Backed Grid")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:   #if event.type == pygame.QUIT:
                intro = False
                pygame.quit()
                quit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                # User clicks the mouse. Get the position
                pos = pygame.mouse.get_pos()
                if 0 <= pos[0] <= num_rect_hor*width  and 0 <= pos[1] <= num_rect_ver*height: 
                    # Change the x/y screen coordinates to grid coordinates
                    column = pos[0] // (width + margin)
                    row = pos[1] // (height + margin)
                                
                    # Set that location to zero
                    grid[row][column] = 1
                    block_cancel = True 
                    block_grid.append([row,column])
                    print("Click ", pos, "Grid coordinates: ", row, column)
                    
        grids()

        
        
        

        #button("start game",550,50,150,50,GREEN,bright_green,"play")
        button("Start_End",550,110,200,50,RED,bright_red,"start_end")
        
        pygame.display.update()
        clock.tick(60)

game_intro()
pygame.quit()


# acknowledgement : sentdex harrison kinsley