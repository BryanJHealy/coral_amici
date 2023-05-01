import PySimpleGUI as sg

class Timeline:
    def __init__(self) -> None:
        self.timeline = [sg.Text('Timeline')]
        pass

    def get_component(self):
        return self.timeline