import PySimpleGUI as sg

class ControlPanel:
    def __init__(self, player) -> None:
        # import, save, generate, previous, play/pause, next
        self.control_panel = [
            sg.Button('Import', key='-IMPORT-'),
            sg.Button('Save', key='-SAVE-'),
            sg.Button('Previous', key='-PREVIOUS-'),
            sg.Button('Play/Pause', key='-PLAY-'),
            sg.Button('Next', key='-NEXT-'),
            sg.Button('Generate', key='-GENERATE-'),
        ]

        self.player = player

    def get_component(self):
        return self.control_panel
    
    def get_event_callbacks(self):
        return [
            ('-IMPORT-', self.import_MIDI),
            ('-SAVE-',self.save),
            ('-PREVIOUS-', self.previous),
            ('-PLAY-', self.play),
            ('-NEXT-', self.next),
            ('-GENERATE-', self.generate),
        ]

    def import_MIDI(self):
        pass

    def save(self):
        pass

    def previous(self):
        pass

    def play(self):
        pass

    def next(self):
        pass

    def generate(self):
        pass