import PySimpleGUI as sg

class Composer:
    def __init__(self) -> None:
        self.composer = [sg.Text('Composer')]
        pass

    def get_component(self):
        return self.composer

    def add_track(self):
        pass

    def delete_track(self):
        pass