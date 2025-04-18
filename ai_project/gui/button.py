# gui/button.py
import pygame

class Button:
    def __init__(self, x, y, width, height, text, action=None):
        """
        Initialize button
        
        Args:
            x (int): X position
            y (int): Y position
            width (int): Button width
            height (int): Button height
            text (str): Button text
            action (function): Function to call when clicked
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.is_hovered = False
        self.font = pygame.font.SysFont(None, 24)
        
        # Button colors
        self.color = (52, 152, 219)  # Blue
        self.hover_color = (41, 128, 185)  # Darker blue
        self.text_color = (255, 255, 255)  # White
        
    def draw(self, surface):
        """
        Draw button on surface
        
        Args:
            surface: Pygame surface to draw on
        """
        # Draw button rectangle
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        
        # Draw button text
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)
        
        # Draw border
        pygame.draw.rect(surface, (0, 0, 0), self.rect, 2, border_radius=5)
        
    def check_click(self, pos):
        """
        Check if button was clicked
        
        Args:
            pos: Mouse position (x, y)
            
        Returns:
            bool: True if button was clicked, False otherwise
        """
        # Check if mouse is over button
        self.is_hovered = self.rect.collidepoint(pos)
        
        # If button was clicked and has action, call it
        if self.is_hovered and self.action:
            self.action()
            return True
            
        return False