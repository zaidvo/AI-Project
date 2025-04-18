# gui/themes.py

class Theme:
    """Base theme class with default colors and styles"""
    def __init__(self):
        # Background colors
        self.bg_color = (240, 240, 240)
        self.grid_color = (80, 80, 80)
        self.highlight_color = (255, 255, 0, 128)
        
        # Player colors
        self.x_color = (41, 128, 185)  # Blue
        self.o_color = (231, 76, 60)   # Red
        
        # UI colors
        self.button_color = (52, 152, 219)
        self.button_hover_color = (41, 128, 185)
        self.button_text_color = (255, 255, 255)
        self.text_color = (0, 0, 0)
        
        # Font
        self.font_name = None  # Default system font
        self.font_size = 24

class DarkTheme(Theme):
    """Dark theme with dark background and bright elements"""
    def __init__(self):
        super().__init__()
        # Background colors
        self.bg_color = (40, 44, 52)
        self.grid_color = (120, 120, 120)
        self.highlight_color = (255, 215, 0, 128)
        
        # Player colors
        self.x_color = (46, 204, 113)  # Green
        self.o_color = (231, 76, 60)   # Red
        
        # UI colors
        self.button_color = (52, 73, 94)
        self.button_hover_color = (44, 62, 80)
        self.button_text_color = (255, 255, 255)
        self.text_color = (236, 240, 241)

class BlueTheme(Theme):
    """Blue-focused theme"""
    def __init__(self):
        super().__init__()
        # Background colors
        self.bg_color = (235, 245, 251)
        self.grid_color = (41, 128, 185)
        self.highlight_color = (52, 152, 219, 128)
        
        # Player colors
        self.x_color = (41, 128, 185)  # Blue
        self.o_color = (155, 89, 182)  # Purple
        
        # UI colors
        self.button_color = (52, 152, 219)
        self.button_hover_color = (41, 128, 185)
        self.button_text_color = (255, 255, 255)
        self.text_color = (44, 62, 80)

# Dictionary of available themes
THEMES = {
    'default': Theme(),
    'dark': DarkTheme(),
    'blue': BlueTheme()
}

# Current theme (can be changed at runtime)
current_theme = THEMES['default']

def set_theme(theme_name):
    """
    Set current theme
    
    Args:
        theme_name (str): Name of theme to use
        
    Returns:
        bool: True if theme was set, False if theme doesn't exist
    """
    global current_theme
    if theme_name in THEMES:
        current_theme = THEMES[theme_name]
        return True
    return False

def get_current_theme():
    """Get current theme"""
    return current_theme