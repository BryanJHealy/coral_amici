import PySimpleGUI as sg
from enum import Enum


class LogLevel(Enum):
    WARNING = 0
    INFO    = 1
    DEBUG   = 2


class MessageLog:
    def __init__(self, log_level=LogLevel.INFO) -> None:
        self.log_level = log_level
        self.messages = 'Message Log:\nWelcome to AMICI Composer!\n'
        self.key = '-LOG-'
        self.message_log = [sg.Text(f'{self.messages}', key=self.key)]

    def get_component(self):
        return self.message_log
    
    # def update_log(self):
    #     self.message_log = [sg.Text(f'{self.messages}')]

    def log(self, level, text):
        if level.value <= self.log_level.value:
            if level.value <= LogLevel.WARNING.value:
                sg.popup_annoying(text)
            else:
                self.messages += f'>> {text}\n'
            
