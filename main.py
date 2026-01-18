import pygame
import sys
import time
from openpyxl import load_workbook

# Initialize Pygame
pygame.init()

# Load arctic map image first to get its dimensions for window size
try:
    temp_img = pygame.image.load("visuals/islands.png")
    img_width, img_height = temp_img.get_size()
    width, height = img_width, img_height
except pygame.error as e:
    print(f"Couldn't load islands image for sizing: {e}")
    width, height = 800, 600  # Fallback to default size

# Set up the display
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Ship Selection")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (100, 150, 255)
DARK_BLUE = (50, 100, 200)
GRAY = (200, 200, 200)
LIGHT_GRAY = (240, 240, 240)
GREEN = (50, 200, 50)
DARK_GREEN = (30, 150, 30)
RED = (255, 0, 0)

# Load background image
try:
    background_img = pygame.image.load("visuals/arctic-ice-ridges.jpg")
    background_img = pygame.transform.scale(background_img, (width, height))
except pygame.error as e:
    print(f"Couldn't load background image: {e}")
    background_img = None

# Load ship placeholder image
try:
    ship_placeholder_img = pygame.image.load("visuals/salacola.png")
except pygame.error as e:
    print(f"Couldn't load ship placeholder image: {e}")
    ship_placeholder_img = None

# Load arctic map image for page 2 (already loaded for sizing, reload for use)
arctic_map_img = None
try:
    arctic_map_img = pygame.image.load("visuals/islands.png")
    # No scaling needed - window is already the same size as the image
except pygame.error as e:
    print(f"Couldn't load arctic map image: {e}")
    arctic_map_img = None

# Function to check if a point is on blue (water) surface
def is_blue_surface(x, y):
    """Check if the pixel at (x, y) is blue (water)"""
    if arctic_map_img is None:
        return False
    
    # Make sure coordinates are within image bounds
    if x < 0 or y < 0 or x >= width or y >= height:
        return False
    
    try:
        # Get pixel color at the click position
        pixel_color = arctic_map_img.get_at((x, y))
        r, g, b, a = pixel_color
        
        # Check if it's blue (blue channel is dominant)
        # Blue water typically has: b > r and b > g, and b is relatively high
        is_blue = b > r and b > g and b > 100
        
        return is_blue
    except:
        return False

# Load ships data from Excel
def load_ships_data():
    ships = []
    try:
        wb = load_workbook('Ships.xlsx')
        ws = wb.active
        
        # Read header row - columns are at indices 0, 2, 4, 6, 7, 9
        header_row = list(ws[1])
        headers = {
            0: header_row[0].value.strip() if header_row[0].value else None,  # Ship type
            2: header_row[2].value.strip() if header_row[2].value else None,  # Ship name
            4: header_row[4].value.strip() if header_row[4].value else None,  # Fuel Consumption
            6: header_row[6].value.strip() if header_row[6].value else None,  # Speed
            7: header_row[7].value.strip() if header_row[7].value else None,  # Durability
            9: header_row[9].value.strip() if header_row[9].value else None   # Durability Rating
        }
        
        # Read ship data (rows 2-6, which gives us 5 ships)
        for row_idx in range(2, 7):  # Rows 2-6 (5 ships)
            row = list(ws[row_idx])
            ship_data = {}
            
            # Map data to headers based on column indices
            if row[0].value:  # Ship type
                ship_data[headers[0]] = str(row[0].value).strip()
            if row[2].value:  # Ship name
                ship_data[headers[2]] = str(row[2].value).strip()
            if row[4].value:  # Fuel Consumption
                ship_data[headers[4]] = str(row[4].value).strip()
            if row[6].value:  # Speed
                ship_data[headers[6]] = str(row[6].value).strip()
            if row[7].value:  # Durability
                ship_data[headers[7]] = str(row[7].value).strip()
            if row[9].value is not None:  # Durability Rating (can be a number)
                ship_data[headers[9]] = str(row[9].value).strip()
            
            if ship_data:  # Only add if we have data
                ships.append(ship_data)
    except Exception as e:
        print(f"Error loading ships data: {e}")
        import traceback
        traceback.print_exc()
    
    return ships

# Load ships
ships = load_ships_data()
selected_ship = None
current_page = 1  # 1 = selection page, 2 = next page
confirmed_ship = None
selected_ship_type = None  # Store the selected ship type
page2_start_time = None  # Track when page 2 was entered
point_a = None  # Store point A position
point_b = None  # Store point B position
points_confirmed = False  # Track if points A and B have been confirmed

# Font setup
font_large = pygame.font.Font(None, 36)
font_medium = pygame.font.Font(None, 24)
font_small = pygame.font.Font(None, 20)

# Helper function to draw text with word wrapping
def draw_text_wrapped(surface, text, font, color, rect, aa=True):
    words = text.split(' ')
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        test_width, _ = font.size(test_line)
        if test_width <= rect.width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    
    y_offset = 0
    for line in lines:
        if y_offset + font.get_height() > rect.height:
            break
        text_surface = font.render(line, aa, color)
        surface.blit(text_surface, (rect.x, rect.y + y_offset))
        y_offset += font.get_height() + 2

# Helper function to format ship description
def format_ship_description(ship):
    description = []
    
    # Define the order and labels for each field
    fields = [
        ('Ship type', 'Ship type'),
        ('Ship name', 'Ship name'),
        ('Fuel Consumption         ', 'Fuel Consumption'),
        ('Speed', 'Speed'),
        ('Durability', 'Durability'),
        ('Durability Rating', 'Durability Rating')
    ]
    
    for excel_key, display_label in fields:
        value = ship.get(excel_key, '')
        if value and value != 'None' and str(value).strip():
            description.append(f"{display_label}:")
            description.append(f"  {value}")
            description.append("")  # Empty line for spacing
    
    return "\n".join(description)

# Helper function to draw ship description with proper formatting
def draw_ship_description(surface, ship, font_label, font_value, color, rect):
    y_offset = 0
    line_height = font_label.get_height() + 4
    
    # Define the order and labels for each field
    fields = [
        ('Ship type', 'Ship type'),
        ('Ship name', 'Ship name'),
        ('Fuel Consumption         ', 'Fuel Consumption'),
        ('Speed', 'Speed'),
        ('Durability', 'Durability'),
        ('Durability Rating', 'Durability Rating')
    ]
    
    for excel_key, display_label in fields:
        value = ship.get(excel_key, '')
        if value and value != 'None' and str(value).strip():
            # Draw label
            if y_offset + line_height > rect.height:
                break
            label_text = font_label.render(f"{display_label}:", True, color)
            surface.blit(label_text, (rect.x, rect.y + y_offset))
            y_offset += line_height
            
            # Draw value (with word wrapping if needed)
            if y_offset + line_height > rect.height:
                break
            words = str(value).split(' ')
            current_line = []
            for word in words:
                test_line = ' '.join(current_line + [word])
                test_width, _ = font_value.size(test_line)
                if test_width <= rect.width - 20:  # Leave some margin
                    current_line.append(word)
                else:
                    if current_line:
                        value_text = font_value.render('  ' + ' '.join(current_line), True, color)
                        surface.blit(value_text, (rect.x, rect.y + y_offset))
                        y_offset += line_height
                        if y_offset + line_height > rect.height:
                            break
                    current_line = [word]
            
            if current_line:
                value_text = font_value.render('  ' + ' '.join(current_line), True, color)
                surface.blit(value_text, (rect.x, rect.y + y_offset))
                y_offset += line_height
            
            # Add spacing between fields
            y_offset += 8

# Main game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            
            if current_page == 1:  # Selection page
                if event.button == 1:  # Left click only on page 1
                    # Check if clicking on confirmation button (in description panel)
                    if selected_ship:
                        panel_x = 300
                        panel_y = 60
                        panel_width = width - panel_x - 50
                        panel_height = height - panel_y - 50
                        button_width = 180
                        button_height = 40
                        confirm_button_x = panel_x + panel_width - button_width - 15
                        confirm_button_y = panel_y + panel_height - button_height - 15
                        confirm_button_rect = pygame.Rect(confirm_button_x, confirm_button_y, button_width, button_height)
                        
                        if confirm_button_rect.collidepoint(mouse_x, mouse_y):
                            confirmed_ship = selected_ship
                            selected_ship_type = selected_ship.get('Ship type', 'Unknown') if selected_ship else None
                            current_page = 2
                            page2_start_time = time.time()  # Record when page 2 starts
                            continue
                    
                    # Check if clicking on a ship button
                    button_width = 250
                    button_height = 50
                    button_start_y = 60
                    button_spacing = 60
                    
                    for i, ship in enumerate(ships):
                        button_x = 50
                        button_y = button_start_y + i * button_spacing
                        button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
                        
                        if button_rect.collidepoint(mouse_x, mouse_y):
                            selected_ship = ship
                            break
            
            elif current_page == 2:  # Page 2: Map clicking
                if not points_confirmed:  # Only allow point changes if not confirmed
                    if event.button == 1:  # Left click
                        # Check if clicking on confirm button
                        if point_a and point_b:
                            button_width = 150
                            button_height = 40
                            button_x = 10
                            button_y = height - button_height - 10
                            confirm_button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
                            
                            if confirm_button_rect.collidepoint(mouse_x, mouse_y):
                                # Confirm the points
                                points_confirmed = True
                            else:
                                # Set point A on map click only if on blue surface
                                if is_blue_surface(mouse_x, mouse_y):
                                    point_a = (mouse_x, mouse_y)
                        else:
                            # Set point A on map click only if on blue surface
                            if is_blue_surface(mouse_x, mouse_y):
                                point_a = (mouse_x, mouse_y)
                    elif event.button == 3:  # Right click - point B
                        # Set point B on map click only if on blue surface
                        if is_blue_surface(mouse_x, mouse_y):
                            point_b = (mouse_x, mouse_y)

    # Draw background
    if current_page == 1:
        # Page 1: Show arctic background
        if background_img:
            screen.blit(background_img, (0, 0))
        else:
            screen.fill(WHITE)
    elif current_page == 2:
        # Page 2: Arctic map background
        if arctic_map_img:
            screen.blit(arctic_map_img, (0, 0))
        else:
            screen.fill(WHITE)

    # Page 1: Ship Selection
    if current_page == 1:
        # Draw title
        title_text = font_large.render("Select a Ship", True, WHITE)
        screen.blit(title_text, (50, 10))

        # Draw ship selection buttons
        button_width = 250
        button_height = 50
        button_start_y = 60
        button_spacing = 60
        
        for i, ship in enumerate(ships):
            button_x = 50
            button_y = button_start_y + i * button_spacing
            button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
            
            # Highlight selected ship
            if selected_ship == ship:
                pygame.draw.rect(screen, DARK_BLUE, button_rect)
            else:
                pygame.draw.rect(screen, BLUE, button_rect)
            
            # Draw button border
            pygame.draw.rect(screen, BLACK, button_rect, 2)
            
            # Draw ship name (full name, with word wrapping if needed)
            ship_name = ship.get('Ship name', f'Ship {i+1}')
            # Try to fit the full name, use smaller font if needed
            button_text = font_medium.render(ship_name, True, WHITE)
            text_width, text_height = button_text.get_size()
            
            # If text is too wide, use smaller font
            if text_width > button_width - 20:
                button_text = font_small.render(ship_name, True, WHITE)
                text_width, text_height = button_text.get_size()
                # If still too wide, wrap to multiple lines
                if text_width > button_width - 20:
                    words = ship_name.split(' ')
                    lines = []
                    current_line = []
                    for word in words:
                        test_line = ' '.join(current_line + [word])
                        test_text = font_small.render(test_line, True, WHITE)
                        if test_text.get_width() <= button_width - 20:
                            current_line.append(word)
                        else:
                            if current_line:
                                lines.append(' '.join(current_line))
                            current_line = [word]
                    if current_line:
                        lines.append(' '.join(current_line))
                    
                    # Draw multiple lines
                    line_height = font_small.get_height() + 2
                    total_height = len(lines) * line_height
                    start_y = button_rect.centery - total_height // 2
                    for line in lines:
                        line_text = font_small.render(line, True, WHITE)
                        line_rect = line_text.get_rect(centerx=button_rect.centerx, y=start_y)
                        screen.blit(line_text, line_rect)
                        start_y += line_height
                else:
                    text_rect = button_text.get_rect(center=button_rect.center)
                    screen.blit(button_text, text_rect)
            else:
                text_rect = button_text.get_rect(center=button_rect.center)
                screen.blit(button_text, text_rect)

        # Draw description panel
        if selected_ship:
            panel_x = 300
            panel_y = 60
            panel_width = width - panel_x - 50
            panel_height = height - panel_y - 50
            
            # Draw semi-transparent panel background
            panel_surface = pygame.Surface((panel_width, panel_height))
            panel_surface.set_alpha(230)
            panel_surface.fill(LIGHT_GRAY)
            screen.blit(panel_surface, (panel_x, panel_y))
            
            # Draw panel border
            panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
            pygame.draw.rect(screen, BLACK, panel_rect, 2)
            
            # Draw description title
            title_text = font_large.render("Ship Details", True, BLACK)
            screen.blit(title_text, (panel_x + 10, panel_y + 10))
            
            # Draw ship image (placeholder)
            image_size = 150
            image_x = panel_x + 10
            image_y = panel_y + 50
            if ship_placeholder_img:
                # Scale image to fit
                img_width, img_height = ship_placeholder_img.get_size()
                scale = min(image_size / img_width, image_size / img_height)
                scaled_width = int(img_width * scale)
                scaled_height = int(img_height * scale)
                scaled_img = pygame.transform.scale(ship_placeholder_img, (scaled_width, scaled_height))
                screen.blit(scaled_img, (image_x, image_y))
            
            # Draw ship description (leave space for image and button at bottom)
            # Position text to the right of the image, or below if image is wide
            desc_x = panel_x + 10 + image_size + 10
            desc_y = panel_y + 50
            desc_width = panel_width - 20 - image_size - 10
            desc_height = panel_height - 70
            desc_rect = pygame.Rect(desc_x, desc_y, desc_width, desc_height)
            draw_ship_description(screen, selected_ship, font_medium, font_small, BLACK, desc_rect)

            # Draw green confirmation button in bottom right of description panel
            button_width = 180
            button_height = 40
            confirm_button_x = panel_x + panel_width - button_width - 15
            confirm_button_y = panel_y + panel_height - button_height - 15
            confirm_button_rect = pygame.Rect(confirm_button_x, confirm_button_y, button_width, button_height)
            
            # Draw button
            pygame.draw.rect(screen, GREEN, confirm_button_rect)
            pygame.draw.rect(screen, BLACK, confirm_button_rect, 2)
            
            # Draw button text
            confirm_text = font_medium.render("Confirm Selection", True, WHITE)
            text_rect = confirm_text.get_rect(center=confirm_button_rect.center)
            screen.blit(confirm_text, text_rect)

    # Page 2: Next Page
    elif current_page == 2:
        # Display ship name in top left
        if confirmed_ship:
            ship_name = confirmed_ship.get('Ship name', 'Unknown Ship')
            ship_name_text = font_large.render(ship_name, True, BLACK)
            screen.blit(ship_name_text, (10, 10))
        
        # Display prompt for 3 seconds
        if page2_start_time is not None:
            elapsed_time = time.time() - page2_start_time
            if elapsed_time < 3:  # Show for 3 seconds
                # Create a semi-transparent background for the prompt
                prompt_bg = pygame.Surface((400, 60))
                prompt_bg.set_alpha(200)
                prompt_bg.fill(LIGHT_GRAY)
                prompt_x = width // 2 - 200
                prompt_y = height // 2 - 30
                screen.blit(prompt_bg, (prompt_x, prompt_y))
                
                # Draw prompt text
                prompt_text = font_medium.render("Select your points A and B", True, BLACK)
                text_rect = prompt_text.get_rect(center=(width // 2, height // 2))
                screen.blit(prompt_text, text_rect)
        
        # Draw point A if set
        if point_a:
            point_x, point_y = point_a
            # Draw red dot
            pygame.draw.circle(screen, RED, (point_x, point_y), 8)
            # Draw label "A" above the dot
            label_text = font_medium.render("A", True, BLACK)
            label_rect = label_text.get_rect(centerx=point_x, bottom=point_y - 12)
            screen.blit(label_text, label_rect)
        
        # Draw point B if set
        if point_b:
            point_x, point_y = point_b
            # Draw red dot
            pygame.draw.circle(screen, RED, (point_x, point_y), 8)
            # Draw label "B" above the dot
            label_text = font_medium.render("B", True, BLACK)
            label_rect = label_text.get_rect(centerx=point_x, bottom=point_y - 12)
            screen.blit(label_text, label_rect)
        
        # Draw confirm button in bottom left if both points are set and not yet confirmed
        if point_a and point_b and not points_confirmed:
            button_width = 150
            button_height = 40
            button_x = 10
            button_y = height - button_height - 10
            confirm_button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
            
            # Draw button
            pygame.draw.rect(screen, GREEN, confirm_button_rect)
            pygame.draw.rect(screen, BLACK, confirm_button_rect, 2)
            
            # Draw button text
            confirm_text = font_medium.render("Confirm", True, WHITE)
            text_rect = confirm_text.get_rect(center=confirm_button_rect.center)
            screen.blit(confirm_text, text_rect)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
