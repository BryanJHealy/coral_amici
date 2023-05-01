import pretty_midi

def import_midi(filepath):
    return pretty_midi.PrettyMIDI(filepath)

def import_and_select(filepath, selection_seconds=15):
    pm = import_midi(filepath)
    end_time = pm.get_end_time()
    if selection_seconds == -1:
        selection = (0, end_time)
    else:
        selection = (0, selection_seconds) if end_time >= selection_seconds else (0, end_time)
    return pm, selection

def plot_piano_roll(pm, start_pitch=40, end_pitch=118, fs=100):
    roll = pm.get_piano_roll(fs)  # [start_pitch:end_pitch]
    # pretty_midi.note_number_to_hz(start_pitch)
    return