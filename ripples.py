import numpy as np
import pygame
import sys

# Initialize Pygame
pygame.init()

# Simulation parameters
Lx, Ly = 9, 9  # Dimensions of the pool
Nx, Ny = 90, 90  # Grid resolution
dx, dy = Lx / Nx, Ly / Ny  # Grid spacing
dt = 0.025  # Time step
Nt = 4000  # Number of iterations
c = 1  # Wave speed
damping_factor = 0.930  # Damping factor for absorbing boundaries
damping_zone_width = 12  # Width of the damping zone
d=15
res = 5

# Create meshgrid for x and y dimensions
x_vec = np.linspace(0, Lx, Nx)
y_vec = np.linspace(0, Ly, Ny)

# Initialize u (wave height at each point) and set the disturbance at the center
u = np.zeros([len(x_vec), len(y_vec)])
u_prev = np.zeros([len(x_vec), len(y_vec)])
u_next = np.zeros([len(x_vec), len(y_vec)])

# Pygame window setup
width, height = res * Nx, res * Ny  # Size of the window
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Wave Simulation')

# Scaling factors to map the simulation grid to the screen
scale_x = width / Nx
scale_y = height / Ny

# Function to convert height to brightness
def height_to_color(h):
    brightness = max(min(int(255 * (h)), 255), -255)  # Map u to a value between 0 and 255
    if h < 0: 
        return (abs(brightness), 0, 0)  # Red for negative values
    else: 
        return (0, 0, brightness)  # Blue for positive values

# Function to calculate the damping factor for each grid point
def get_damping_factor(x, y, Nx, Ny, damping_zone_width):
    # Calculate the distance to the nearest boundary
    dist_x = min(x, Nx - x - 1)
    dist_y = min(y, Ny - y - 1)
    
    # If within the damping zone, calculate the damping factor
    if dist_x < damping_zone_width or dist_y < damping_zone_width:
        # Calculate damping based on proximity to the boundary
        damping_x = max(0, (damping_zone_width - dist_x) / damping_zone_width)
        damping_y = max(0, (damping_zone_width - dist_y) / damping_zone_width)
        
        # Apply more damping as you get closer to the boundary
        return damping_factor ** (damping_x + damping_y)
    return 1  # No damping if outside the damping zone

# Simulation loop
running = True
t = 0

clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            sys.exit()
            
    # Detect mouse wheel scrolling
        if event.type == pygame.MOUSEWHEEL:
            if event.y > 0:  # Scrolled up
                print("Scrolled up")
                d += 1
            elif event.y < 0:  # Scrolled down
                print("Scrolled down")
                d -= 1
        
        if pygame.mouse.get_pressed()[0]:
            u[pygame.mouse.get_pos()[0]//res, pygame.mouse.get_pos()[1]//res] += 0.4
    
    # Continuous disturbance at a specific point
    u[20, -d + Ny // 2] += 0.4 * np.sin(t / 5)
    
    u[20, d + Ny // 2] += 0.4 * np.sin(t / 5)
    
    # Calculate the next time step for u
    for x in range(1, Nx - 1):
        for y in range(1, Ny - 1):
            u_next[x, y] = (
                c**2 * dt**2 * (
                    (u[x + 1, y] - 2 * u[x, y] + u[x - 1, y]) / dx**2 +
                    (u[x, y + 1] - 2 * u[x, y] + u[x, y - 1]) / dy**2
                )
                + 2 * u[x, y] - u_prev[x, y]
            )

            # Apply damping near the edges
            damping = get_damping_factor(x, y, Nx, Ny, damping_zone_width)
            u_next[x, y] *= damping
            u[x, y] *= damping
            u_prev[x, y] *= damping

    # Draw the current state of the wave
    for x in range(Nx):
        for y in range(Ny):
            color = height_to_color(u[x, y])
            pygame.draw.rect(screen, color, (x * scale_x, y * scale_y, scale_x, scale_y))

    pygame.display.flip()

    # Update previous states for the next iteration
    u_prev = np.copy(u)
    u = np.copy(u_next)

    # Limit frame rate
    clock.tick(600)

    # Increment time step
    t += 1
    if t > Nt:
        break
        
    print(pygame.mouse.get_pos())
# Quit Pygame
pygame.quit()

