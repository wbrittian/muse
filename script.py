from pathlib import Path
from pretty_midi import PrettyMIDI
import numpy as np

def quantize_notes(pm: PrettyMIDI, qdiv: int):
        count = 0
        beat_times = pm.get_beats() 
        end_time   = pm.get_end_time()

        beat_times = np.append(beat_times, end_time)

        def time_to_division(t):
            i = np.searchsorted(beat_times, t, side='right') - 1
            i = min(max(i, 0), len(beat_times) - 2)

            sec_per_beat = beat_times[i+1] - beat_times[i]
            frac = (t - beat_times[i]) / sec_per_beat
            return int(round(i * qdiv + frac * qdiv))

        notes = []
        for inst in pm.instruments:
            for n in inst.notes:
                start_div = time_to_division(n.start)
                end_div   = time_to_division(n.end)
                dur_divs  = max(1, end_div - start_div)

                start_beats = start_div / qdiv
                dur_beats = dur_divs / qdiv
                notes.append((start_beats, n.pitch, dur_beats))

                if abs((start_beats % 25)) >= 1e-8:
                    count += 1
                if abs((dur_beats) % 25) >= 1e-8:
                    count += 1
        return (sorted(notes, key=lambda x: (x[0], x[1])), count)

root = Path("data/raw_data/bimmuda_dataset")
full_count = 0
for midfile in root.rglob('*.mid'):
    if 'full' not in midfile.stem:
        pm = PrettyMIDI(str(midfile))


        beat_times = pm.get_beats()
        sec_per_div = np.diff(beat_times).mean() / 4

        notes = quantize_notes(pm, 12)
        full_count += notes[1]
        notes = notes[0]

        base_pitch = notes[0][1]
        tokens = []
        prev_end = 0
        seq_dur = 0
        for start, pitch, dur in notes:
            pitch_diff = pitch - base_pitch
            print(dur)
            raw_tokens = [
                f"<NOTE_{round(dur * 4) / 4}>",
                f"<PITCH_{pitch_diff:+d}>"
            ]

            offset = round((start - prev_end) * 4) / 4
            if offset != 0:
                raw_tokens.insert(0, f"<REST_{offset}>")

            tokens = tokens + raw_tokens
            prev_end = start + dur
            seq_dur += offset + dur
            
        diff = seq_dur - prev_end
        if diff > 0:
            tokens.append(f"<REST_{diff}>")

        # print(f"{tokens}: {str(midfile)}")
       
print(full_count)