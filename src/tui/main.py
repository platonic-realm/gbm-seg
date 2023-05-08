"""
Author: Arash Fatehi
Date:   30.04.2022
"""

# Python Imports
import os

# Library Imports
import TermTk as ttk

# Local Imports
from src.tui.base import BaseWindow


class MainWindow(BaseWindow):
    def __init__(self, _configs):
        super().__init__(_configs)

        self.draw(self.main_frame)

    def draw(self, _root):
        # Define the main Layout
        splitter = ttk.TTkSplitter(parent=_root,
                                   orientation=ttk.TTkK.HORIZONTAL)
        _root.layout().addWidget(splitter, 0, 0)
        self.menu = ttk.TTkFrame(parent=splitter,
                                 border=False,
                                 maxWidth=20,
                                 layout=ttk.TTkGridLayout())
        self.experiments = ttk.TTkFrame(parent=splitter,
                                        border=False,
                                        layout=ttk.TTkVBoxLayout())
        self.draw_menu(self.menu)
        self.draw_experiments(self.experiments)

        return splitter

    def draw_menu(self, _frame):
        list_button = ttk.TTkListWidget(maxWidth=20, minWidth=10)

        self.create_button = ttk.TTkButton(text="Create",
                                           enabled=True,
                                           border=True)
        self.create_button.clicked.connect(self.create_experiment)

        self.train_button = ttk.TTkButton(text="Train",
                                          enabled=False,
                                          border=True)

        self.infer_button = ttk.TTkButton(text="Infer",
                                          enabled=False,
                                          border=True)

        list_button.addItem(self.create_button)
        list_button.addItem(ttk.TTkSpacer())
        list_button.addItem(self.train_button)
        list_button.addItem(ttk.TTkSpacer())
        list_button.addItem(self.infer_button)
        list_button.addItem(ttk.TTkSpacer())

        quit_button = ttk.TTkButton(text="Quit",
                                    border=True,
                                    maxHeight=3)
        quit_button.clicked.connect(ttk.TTkHelper.quit)

        settings_button = ttk.TTkButton(text="Settings",
                                        border=True,
                                        maxHeight=3)

        title_label = ttk.TTkLabel(text="[ GBM-Segmentation ]",
                                   maxHeight=2)
        _frame.layout().addWidget(title_label,      0, 0)
        _frame.layout().addWidget(list_button,      1, 0)
        _frame.layout().addWidget(settings_button,  2, 0)
        _frame.layout().addWidget(quit_button,      3, 0)

    def draw_experiments(self, _frame):
        # Multi Selection List
        ttk.TTkLabel(parent=_frame,
                     text="[ Experiments ]",
                     maxHeight=2)
        self.projects_list = \
            ttk.TTkList(parent=_frame,
                        minWidth=50,
                        selectionMode=ttk.TTkK.MultiSelection)

    @ttk.pyTTkSlot()
    def create_experiment(self):
        self.create_button.setEnabled(False)
        self.create_window = \
            ttk.TTkWindow(pos=(2, 2),
                          layout=ttk.TTkVBoxLayout(),
                          size=(self.root.width()-4,
                                self.root.height()-4),
                          title="New experiment")

        self.create_window.setLayout(ttk.TTkVBoxLayout())

        list_dummy = ttk.TTkListWidget()
        bottons_frame = ttk.TTkFrame(layout=ttk.TTkGridLayout(),
                                     border=False)

        create_confirm = ttk.TTkButton(parent=bottons_frame,
                                       text="Confirm",
                                       enabled=True,
                                       border=True,
                                       maxHeight=1,
                                       maxWidth=10)
        create_confirm.clicked.connect(self.cancel_create_experiment)

        create_cancel = ttk.TTkButton(parent=bottons_frame,
                                      text="Cancel",
                                      enabled=True,
                                      border=True,
                                      maxHeight=1,
                                      maxWidth=10)
        create_cancel.clicked.connect(self.cancel_create_experiment)

        self.create_window.layout().addWidget(list_dummy)
        self.create_window.layout().addWidget(bottons_frame)

        self.root.layout().addWidget(self.create_window)

    @ttk.pyTTkSlot()
    def cancel_create_experiment(self):
        self.create_window.close()
        self.create_button.setEnabled(True)

    def load_experiments(self):
        root_dir = self.configs['experiments']['root']
        project_direcories = [name for name in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, name))]

        for project in project_direcories:
            self.projects_list.addItem(project)
            self.projects_list.addItem(ttk.TTkSpacer())
