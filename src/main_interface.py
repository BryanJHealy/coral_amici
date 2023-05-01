import pretty_midi
import PySimpleGUI as sg
import os
from matplotlib import pyplot as plt

from interface.components.message_log import MessageLog
from interface.components.message_log import LogLevel
from interface.util.player import Player
import interface.util.midi_util as mu

from generate_accompaniment import AccompanimentGenerator


class MainWindow:
    def __init__(self, model_seq_secs=15, autosave=True, log_level=LogLevel.DEBUG):
        self.autosave = autosave
        self.player = Player()
        self.midi_filepath = None
        self.model_seq_secs = model_seq_secs
        self.selection_start = None
        self.selection_end = None
        self.pm = None
        self.song_duration = 100
        self.log_level=log_level
        self.playing = False
        self.recording = False
        self.window_vals = None
        self.main_dir = os.path.abspath(os.path.dirname(__file__))
        self.interface_dir = os.path.join(self.main_dir, 'interface')

        # # Accompaniment model
        # self.sequence_seconds = 15
        # self.start_secs = start_secs  # sequence start offset from beginning of file
        # self.skip_empty_intro = skip_empty_intro  # sequence starts at first note after offset if True, else at offset
        # self.vocab_size = vocab_size  # 0 to 127, representing the notes from C-1 to G9
        # self.samples_per_sec = samples_per_sec  # data resolution
        # self.only_keep_melody_track = only_keep_melody_track

        self.model_list = ['LSTM VAE', 'LSTM']
        self.model_paths = {self.model_list[0]: os.path.join(self.main_dir, 'models', 'lstm_vae'),
                            self.model_list[1]: os.path.join(self.main_dir, 'models', 'lstm_accompaniment')}
        self.model_name = self.model_list[0]
        self.model_path = self.model_paths[self.model_name]
        # self.model = self.load_model()

        self.control_panel = [
            sg.Button('Import', key='-IMPORT-'),
            sg.Button('Save', key='-SAVE-'),
            sg.Button('Record', key='-RECORD-'),
            sg.Button('Previous', key='-PREVIOUS-'),
            sg.Button('Play/Pause', key='-PLAY-'),
            sg.Button('Next', key='-NEXT-'),
            sg.Button('Generate', key='-GENERATE-'),
            sg.Combo(self.model_list, default_value=self.model_list[0], s=(15,22), enable_events=True, readonly=True, k='-MODEL-')
        ]

        self.timeline = [
            # [sg.Slider((0,100), orientation='h', size=(138,15))],
            [ sg.T('Selection Start:End -> 0s : 15s',size=(32,1), key='-TIMELINE_TEXT-')],
            [ sg.Slider((0,100), key='-SLIDER-', orientation='h', enable_events=True, disable_number_display=True, size=(138,15))]
            # [ sg.T('15', size=(4,1), key='-SLIDER_RIGHT-')]
            # [sg.Slider((0,100), orientation='h', size=(200,15), default_value=self.model_seq_secs)]
        ]

        self.composer = [
            # [sg.Text('Track 1', size=(20,10)), sg.Graph(canvas_size=(1550, 250), graph_bottom_left=(0,0), graph_top_right=(400, 400), key='-GRAPH0-')],
            # [sg.Graph(canvas_size=(1550, 250), graph_bottom_left=(0,0), graph_top_right=(400, 400), key='-GRAPH0-')],
            # [sg.Image('/home/bryan/amici_model/coral_amici/src/interface/empty_track.png', key='-TRACK_IMG0-')],
            # [sg.Image('/src/interface/empty_track.png', key='-TRACK_IMG0-')],
            [sg.Image(os.path.join(self.interface_dir, 'empty_track.png'), key='-TRACK_IMG1-')],
            # [sg.Image('empty_track.png', key='-TRACK_IMG2-')],
            # [sg.Image('empty_track.png', key='-TRACK_IMG3-')]
        ]
        self.message_log = MessageLog(log_level=LogLevel.DEBUG)

        self.layout = [ self.control_panel,
                        self.timeline,
                        self.composer,
                        self.message_log.get_component()
                        ]

        
        self.event_callbacks = {            
            '-IMPORT-': self.import_midi,
            '-SAVE-': self.save,
            '-RECORD-': self.record_midi,
            '-PREVIOUS-': self.previous,
            '-PLAY-': self.play_pause,
            '-NEXT-': self.next,
            '-GENERATE-': self.generate_accompaniment,
            '-SLIDER-': self.update_timeline,
            '-MODEL-': self.select_model,
            None: print
        }

        sg.theme('DarkBlue15')
        self.window = sg.Window('AMICI Composer', self.layout)

    def get_plot_img(self, sampling_frequency=100, notes_to_plot=1500, filename='track.png'):
        if self.pm:
            notes = self.pm.get_piano_roll(fs=sampling_frequency)
            plt.matshow(notes[:,:notes_to_plot])
            plt.axis('off')
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    def test_import(self):
        self.midi_filepath = self.prompt('Enter the path to the MIDI file\n(e.g. \"/home/bryan/amici/POP909-Dataset/POP909/001/001.mid\"):')

        seq_duration = self.model_seq_secs if self.model_seq_secs else -1
        self.pm, self.selection_start, self.selection_end = mu.import_and_select(self.midi_filepath, seq_duration)
        self.display_notification(LogLevel.INFO, f'Loaded {self.midi_filepath}')

    def run_interface(self):
      try:
        while True:                            
            # Display and interact with the Window
            event, self.window_vals = self.window.read()

            print(f'{event}, {self.window_vals}')

            try:
                self.event_callbacks[event]()
            except KeyError:
                self.display_notification(LogLevel.WARNING, f'Unknown event: {event}')

            # self.display_notification(LogLevel.DEBUG, f'{event}')
            # self.display_notification(LogLevel.DEBUG, f'{values}')

            if event == sg.WIN_CLOSED or event == 'Exit':
                self.display_notification(LogLevel.INFO, 'Shutting down...')
                break

      except KeyboardInterrupt:
          self.display_notification(LogLevel.INFO, 'Shutting down...')
          self.window.close()

    def display_notification(self, level, text):
        self.message_log.log(level, text)
        self.window[self.message_log.key].update(self.message_log.messages)

    def update_timeline(self):
        self.selection_start = int(self.window_vals['-SLIDER-'])
        self.selection_end = min(self.selection_start + 15, self.song_duration)
        self.window['-TIMELINE_TEXT-'].update(f'Selection Start:End -> {self.selection_start}s : {self.selection_end}s')
        # self.window['-SLIDER_RIGHT-'].update(self.selection_end)


    def prompt(self, message, filepath=False):
        layout = [[sg.Text(f'{message}')],      
                 [sg.InputText(key='-IN-')],      
                 [sg.Submit(key='-PROMPT_SUBMIT-'), sg.Cancel(key='-PROMPT_CANCEL-')]]      
      
        window = sg.Window('User Input', layout)
        while True:
            event, values = window.read() 

            if event in ['-PROMPT_SUBMIT-', '-PROMPT_CANCEL-']:
                break
        # TODO: input validation here
        window.close()
        print('closing...')
        return values['-IN-']

    def import_midi(self):
        self.midi_filepath = None

        # try:
        self.midi_filepath = self.prompt('Enter the path to the MIDI file\n(e.g. \"/home/bryan/amici/POP909-Dataset/POP909/001/001.mid\"):')

        try:
            seq_duration = self.model_seq_secs if self.model_seq_secs else -1
            self.pm, (self.selection_start, self.selection_end) = mu.import_and_select(self.midi_filepath, seq_duration)
            self.song_duration = self.pm.get_end_time()
            self.display_notification(LogLevel.INFO, f'Loaded {self.midi_filepath}')
            roll_img = self.get_plot_img()
            self.window['-TRACK_IMG0-'].update('track.png')
        except Exception as e:
            print(e)
            self.display_notification(LogLevel.WARNING, f'Unable to load {self.midi_filepath}\nCheck path and try again.')
            return
    
    def record_midi(self):
        if self.recording:
            self.display_notification(LogLevel.INFO, 'Recording stopped.')
            self.recording = False
        else:
            self.recording = True
            self.display_notification(LogLevel.INFO, 'Recording started...')

    def previous(self):
        self.display_notification(LogLevel.DEBUG, 'Previous.')
        # decrement sequence pointers by seq_len if possible or go back to beginning
    
    def play_pause(self, start_time=0, stop_time=-1):
        if self.playing:
            self.player.stop()
            self.playing = False
            self.display_notification(LogLevel.INFO, 'Paused.')
        else:
            self.playing = True
            # self.play(values, start_time=self.selection_start, stop_time=self.selection_end)
            if self.midi_filepath:
                self.display_notification(LogLevel.INFO, 'Playing...')
                self.player.play(self.midi_filepath)
            else:
                self.display_notification(LogLevel.WARNING, 'No MIDI file selected/saved.')

    def next(self):
        self.display_notification(LogLevel.DEBUG, 'Next.')
        # increment sequence pointerssby seq_len if possible or go to end

    def load_model(self):        
        self.model = AccompanimentGenerator(self.model_path, self.midi_filepath, 'test_generation.mid', roll=False,
                 sequence_seconds=15, start_secs=0, skip_empty_intro=True,
                 vocab_size=128, samples_per_sec=1, only_keep_melody_track=True)

    def select_model(self):
        if self.model_name != self.window_vals['-MODEL-']:
            self.model_name = self.window_vals['-MODEL-']
            self.load_model()
    
    def add_track(self, instrument='piano'):
        self.pm.instruments.append(pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument)))
        # TODO: add new track visual
    
    def generate_accompaniment(self):
        # load accompaniment model
        # generation = model(self.selection)
        self.display_notification(LogLevel.INFO, 'Generating accompaniment...')
        self.model.generate()
    
    def save(self):
        # Save pm to .mid file
        # TODO: use pikle to save pm memory for quicker loading later
        self.display_notification(LogLevel.INFO, 'Saved.')

    def delete_track(self):
        pass

# Load File:
# path = input('Enter the path to the MIDI file\n(e.g. "/home/bryan/amici/POP909-Dataset/POP909/001/001.mid"):')
# midi_data = pretty_midi.PrettyMIDI(path)

if __name__ == '__main__':
    main_window = MainWindow()
    # main_window.test_import()
    main_window.run_interface()

    # load any large files into memory first, show splash screen
    # build main window
    # show main window
    # start interface loop