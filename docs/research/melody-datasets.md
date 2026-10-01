# More melody data than BiMMuDa

**Question:** which datasets can add monophonic melodies to BiMMuDa's ~1,156,
do they help, and how do we handle the control features they lack?

**Answer:** HookTheory (Sheet Sage release) and POP909, both now loadable as
extra sources. Together they add ~24k melodies (about 20× BiMMuDa). Training on
them improves held-out BiMMuDa scores by a wide margin (see Results). Genre
and era for the new data are the new tokens `<GENRE_Unknown>` / `<ERA_Unknown>`.

Survey checked 2026-10-01; every URL below was opened or downloaded.

## Survey

| Dataset | Size | Licence | Available | Mono melody | Native tempo / TS / mode / genre / era | Pop fit | Cleaning |
|---|---|---|---|---|---|---|---|
| **HookTheory (Sheet Sage)** | 26,175 clips, 23,833 with melody, ~14k songs | CC BY-NC-SA 3.0 | direct, 20 MB JSON.gz | native | from beat alignment / yes / yes / – / – | very high | low |
| **POP909** | 909 songs | MIT (repo) | direct zip | `MELODY` track | yes / beat files / `key_audio.txt` / all pop / – | high (Mandopop) | low |
| Lakh MIDI (LMD) | 176k full, 45k matched | CC BY 4.0 | direct | needs melody-track detection | noisy MIDI meta / genre + year via MSD for matched | high but noisy | high |
| OpenEWLD | 568 lead sheets | public-domain scores | GitHub / Zenodo | yes | TS, key, genre, year (mostly 1890s–1940s) | low–medium | medium (MusicXML; TS/key bugs) |
| Nottingham (cleaned) | ~1,034 | GPL-3.0 | GitHub | yes | TS, key | low (folk) | low |
| IrishMAN / TheSession | 216k / tens of k | MIT / permissive | HF / GitHub | yes (ABC) | TS, key | low (Irish trad) | medium (ABC) |
| Essen (EsAC) | ~8.4k | unclear | GitHub mirror only | yes | TS, key | low | medium |
| MTC (Meertens) | 18.6k | CC BY-NC-SA 3.0 | form-gated | yes | TS, key | low (Dutch folk) | medium |
| Weimar Jazz DB | 456 solos | ODbL | direct | yes | tempo, TS, key, year, style | low (improv) | low |
| ComMU | 3,048 melodies | CC BY-NC-SA 4.0 | in repo | via track role | BPM, TS, key, bars | low (stock music) | low |
| TheoryTab (wayne391), HLSD | 11–18k | academic / none | Drive / none | yes | key, meter | superseded by Sheet Sage | – |
| Wikifonia | ~6.7k | taken down | no | – | – | – | – |

Sources:
- HookTheory: `https://github.com/chrisdonahue/sheetsage-data/raw/refs/heads/main/hooktheory/Hooktheory.json.gz`
- POP909: `https://github.com/music-x-lab/POP909-Dataset` (`POP909.zip`)
- LMD: `http://hog.ee.columbia.edu/craffel/lmd/`
- OpenEWLD: `https://github.com/00sapo/OpenEWLD`

### Why these two

- **HookTheory** is the closest match to BiMMuDa. It holds verse and chorus
  excerpts of Western pop and rock, already quantized in beats, with key, meter
  and an audio-aligned tempo. It is natively monophonic and needs almost no
  cleaning.
- **POP909** has a permissive licence and human-transcribed melodies of whole
  songs, with beat and key annotations.
- **Lakh** is the only larger pop source with genre and year labels. Picking the
  melody track out of each file is a project of its own, so it is the next step
  if more data is needed.
- The folk and jazz sets are a different style. At most they would be
  pretraining material.

### Licence

The two datasets differ:

- **POP909** is MIT. The underlying songs are copyrighted, which is the same
  position as BiMMuDa.
- **HookTheory is CC BY-NC-SA 3.0: non-commercial, share-alike.** That suits
  this capstone. A model trained on it should be treated as NC-SA too,
  including the committed `model/museformer.pt` if it is trained with
  HookTheory. If the project ever goes commercial, retrain without that source
  (`--sources bimmuda pop909`).

Neither dataset is committed. `scripts/download_data.sh` fetches both into the
gitignored `data/raw/`.

## Integration

`src/pysrc/data_client/sources.py` loads both datasets into the same record
shape as BiMMuDa: the 11 features, the notes, the source and the song id.
`DataClient(sources)` picks which sources to use; with `None` it uses every
source that has been downloaded.

### Missing labels

GENRE and ERA become `Unknown`, using two tokens appended to the vocabulary.
Ids 0–419 are unchanged, which was checked against the previous
`model/tokens.json`.

The alternatives were rejected:
- Mapping to `Other` would teach the model that `Other` means "modern pop".
- Dropping the prefix tokens would change the sequence layout.

At generation time the user still picks a real genre and era, which the model
learned from BiMMuDa.

### Per-source details

- **HookTheory**
  - Pitches are octave-relative, so each melody is placed with its median in
    C4–B4.
  - Tempo comes from the beat-to-audio alignment.
  - Only 4/4 and 3/4 clips without swing or tempo, key or meter changes are
    kept.
  - Modal scales map to major or minor by their third.
  - Clips whose song slug matches any BiMMuDa title are dropped (conservative),
    so held-out BiMMuDa songs cannot leak into training.
  - Result: 17,429 melodies from 10,560 songs.
- **POP909**
  - Notes are quantized to the annotated beat grid (`beat_midi.txt`) from the
    first 4/4 downbeat.
  - Songs with key changes are skipped.
  - Songs are cut into 8-bar phrases on bar lines, which matches BiMMuDa's
    typical section length (median 8 bars).
  - Result: 6,663 phrases from 757 songs.

### Tokenizer

- Notes are made strictly monophonic: the highest note wins on a shared onset,
  and an overlapping note is cut at the next onset.
- Notes longer than 4 beats are capped at 4 beats.
- Bug fixed: such notes used to be split into repeated NOTE/PITCH pairs and
  then followed by a spurious REST, because the end time of the shortened piece
  was used.
- Bug fixed: when a file ended exactly on a beat, the quantizer doubled the
  last note.
- Encoding and decoding are lossless on cleaned notes (`tests/test_tokenizer.py`,
  including real BiMMuDa files).
- Against the old tokenizer, 1,039 of 1,158 BiMMuDa sequences are identical.
  Every one that differs comes from one of the two fixes above.

## Results

### How the comparison was run

- **Held-out set:** 10% of BiMMuDa *songs* (seed 0), which is 116 melodies.
  All sections of a held-out song are excluded from training. The same split
  scores every run.
- **Training data:** the rest of BiMMuDa plus whichever extra sources the run
  uses.
- **Early stopping:** each run keeps its best epoch on that same held-out set
  (patience 8–15). This flatters every row by the same small amount.
- **Commands:** `scripts/experiment.py` trains; `scripts/sample_eval.py`
  scores generations.
- **Hardware:** the Mac (MPS) and the homelab RTX 2060 SUPER.

### Held-out loss and accuracy

`val_loss` is cross-entropy over every non-PAD target. `mel_loss` and `mel_acc`
cover only the targets after the control prefix: the melody tokens and EOS.

| Run | Train melodies | Model | Best epoch | val_loss | mel_loss | mel_acc |
|---|---|---|---|---|---|---|
| baseline (BiMMuDa only) | 1,042 | d128, 2 layers | 49 | 1.584 | 1.581 | 47.5% |
| + POP909 | 7,691 | d128, 2 layers | 35 | 1.500 | 1.475 | 50.6% |
| + POP909 + HookTheory | 24,755 | d128, 2 layers | 37 | 1.306 | 1.260 | 58.9% |
| + POP909 + HookTheory | 24,755 | d256, 4 layers, 8 heads | 15 | 1.269 | 1.219 | 60.3% |
| ↳ fine-tuned on BiMMuDa (lr 1e-4) | 1,042 | d256, 4 layers | 2 | **1.225** | **1.201** | **60.9%** |

Notes on the table:
- HookTheory supplies most of the gain. POP909 alone helps less, probably
  because it is Mandopop.
- The larger model helps once the data supports it, but it overfits after
  about 15 epochs.
- A short fine-tune on BiMMuDa adds a little more, so this is the shipped
  model (`model/`).
- `model/config.json` lists all three sources, so an in-app retrain trains on
  everything.

### Generation quality

There is one generation per held-out control prefix, scored against the real
held-out melodies. Settings: top-k 16, temperature 1.0, BARS guard on.

| | key fit | distinct pitches | span (semitones) | repetition | bars within ±1 |
|---|---|---|---|---|---|
| real held-out melodies | 0.986 | 6.97 | 11.4 | 0.346 | 91% |
| baseline | 0.985 | 6.88 | 11.1 | 0.179 | 95% |
| + all sources, small | 0.992 | 7.19 | 11.4 | 0.318 | 100% |
| shipped (medium, fine-tuned) | 0.994 | 6.61 | 11.1 | 0.363 | 100% |

How each column is measured:
- **Key fit** is the share of notes inside the best-fitting diatonic scale.
- **Repetition** is the share of 4-note interval patterns that already occurred
  earlier in the same melody.

What the numbers show:
- **Pitch variety.** The original complaint of about 6 distinct pitches and a
  narrow range came mostly from **top-k 8** sampling. At top-k 8 the baseline
  gives 6 distinct pitches and a 10-semitone span. Raising top-k to 16 (now
  the default) matches the real melodies for every model without losing key
  fit.
- **Repetition.** Real pop melodies reuse their motifs. The baseline
  under-repeats (0.18 against 0.35) and wanders. The models trained on more
  data reuse motifs at the real rate.
- **BARS.** Unconstrained, only 23% (baseline) and 66% (small, all sources) of
  samples land within ±1 bar of the request. The new **BARS guard** brings this
  to 97–100% and runs in both backends:
  - It masks EOS until the last bar has started.
  - It stops at the requested length.
  - It never stops between a NOTE and its PITCH.
  - It lives in the C++ `Museformer.generate` and in `sample_tokens`, with a
    parity test between the two.
- **Listening.** WAVs were rendered with a sine synth (`--wav`). No human has
  listened to them yet; the numbers above are proxies.

### Next steps

- Lakh MIDI with melody-track extraction, for more Western pop and real
  genre/era labels on the new data.
- Look up genre and era for HookTheory songs (MusicBrainz) to replace
  `Unknown`.
- Dropout 0.2 on the medium model was still improving when this was written
  (1.259 at epoch 26, against 1.269 for the original medium run). It is worth
  finishing and fine-tuning.
