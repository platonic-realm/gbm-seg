"""
Author: Arash Fatehi
Date:   01.05.2023
"""

# Python Imprts

# Library Imports
import TermTk as ttk

# Local Imports


class BaseWindow():
    def __init__(self, _configs):
        self.configs = _configs

        # Create a root object (it is a widget that represent the terminal)
        self.root = ttk.TTk(title="GBM Segemntation",
                            mouseTrack=True)

        self.main_frame = ttk.TTkFrame(parent=self.root,
                                       border=False,
                                       size=(self.root.width(),
                                             self.root.height()))
        self.main_frame.setLayout(ttk.TTkGridLayout())

    def run(self):
        # Start the Main loop
        self.root.mainloop()
