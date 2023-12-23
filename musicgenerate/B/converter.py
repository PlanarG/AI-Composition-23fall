# Author: namespace-std 
# Convert midi from/to json text 

import json
import os
import math

# pip install mido
from mido import Message
import mido

def miditodata(input, output):
    file = mido.MidiFile(input)
    basetempo = file.ticks_per_beat
    octtempo = basetempo // 2 
    tempo = 1000000
    timesignature = {'type': 'time_signature', 'numerator': 4, 'denominator': 4, 'clocks_per_click': 24, 'notated_32nd_notes_per_beat': 8, 'time': 0}
    keysignature = {'type': 'key_signature', 'key': 'C', 'time': 0}
    # print(basetempo)

    # Same as in midi-standard (60 = C4) 
    pitch = []
    # Decide where should I break
    rhythm = []
    
    def decoder(trk):
        nonlocal tempo
        # print('track!')
        ti = 0
        beat = 0
        curpitch = 0
        for msg in trk:
            d = msg.dict()
            ti = ti + d['time']
            beatcnt = math.ceil(ti / octtempo)
            while(beat < beatcnt):
                beat = beat + 1
                pitch.append(curpitch)
                if(beat != beatcnt and pitch != 0):
                    rhythm.append(1)
                else:
                    rhythm.append(0)
            if(d['type'] == 'note_on'):
                # note on
                if(d['velocity'] >= 10):
                    curpitch = d['note']
                # note off
                else:
                    curpitch = 0
                # print(d)
            elif(d['type'] == 'set_tempo'):
                tempo = d['tempo']
            elif(d['type'] == 'key_signature'):
                keysignature['key'] = d['key']
            else:
                None
                # print(d)
            # print(d)
        return

    decoder(file.tracks[0])
    try:
        decoder(file.tracks[1])
    except:
        print('Midi don\'t have track #1')
    # print(tempo)

    data = {}
    data["octtempo"] = octtempo
    data["tempo"] = tempo
    data["timesignature"] = timesignature
    data["keysignature"] = keysignature
    data["pitch"] = pitch
    data["rhythm"] = rhythm

    json_str = json.dumps(data)
    ouf = open(output,mode='w')
    ouf.write(json_str)
    return

def datatomidi(input, output):
    inf = open(input,mode='r')
    data = json.loads(inf.read())
    file = mido.MidiFile()
    octtempo = data['octtempo']
    file.ticks_per_beat = octtempo * 2
    file.type = 1

    track0 = mido.MidiTrack()
    track1 = mido.MidiTrack()
    track0.append(Message('program_change',channel=0,program= 75,time=0))
    track0.append(mido.MetaMessage.from_dict({'type': 'track_name', 'name': '', 'time': 0}))
    track0.append(mido.MetaMessage.from_dict({'type': 'midi_port', 'port': 0, 'time': 0}))
    track0.append(mido.MetaMessage.from_dict(data['keysignature']))
    track0.append(mido.MetaMessage.from_dict(data['timesignature']))
    tempodata = {'type': 'set_tempo', 'tempo': 1000000, 'time': 0}
    tempodata['tempo'] = data['tempo']
    track0.append(mido.MetaMessage.from_dict(tempodata))
    track1.append(mido.MetaMessage.from_dict({'type': 'track_name', 'name': '', 'time': 0}))
    track1.append(mido.MetaMessage.from_dict({'type': 'midi_port', 'port': 0, 'time': 0}))
    
    # a = mido.Message.from_dict({'type': 'note_on', 'time': 1, 'note': 66, 'velocity': 80, 'channel': 0})
    curpitch = 0
    delta = -1
    note_on = {'type': 'note_on', 'note': 0, 'velocity': 100, 'channel': 0, 'time': 0}
    note_off = {'type': 'note_on', 'note': 0, 'velocity': 0, 'channel': 0, 'time': 0}
    for (pitch,cont) in zip(data['pitch'],data['rhythm']):
        # print(pitch,cont)
        if(pitch != curpitch):
            curpitch = pitch
            if(curpitch != 0):
                note_on['note'] = curpitch
                note_on['time'] = note_on['time'] + delta + 1
                # print(note_on)
                track1.append(mido.Message.from_dict(note_on))
                note_on['time'] = 1
                delta = -1
        delta = delta + octtempo
        if(cont == 0 and curpitch != 0):
            note_off['note'] = curpitch
            note_off['time'] = delta
            track1.append(mido.Message.from_dict(note_off))
            delta = -1
            curpitch = 0
        
    
    file.tracks.append(track0)
    file.tracks.append(track1)
    file.save(output)
    return

# miditodata("01.mid", "01.txt")
#datatomidi("1.txt", "1.mid")
datatomidi("2.txt", "2.mid")
datatomidi("3.txt", "3.mid")
datatomidi("4.txt", "4.mid")
datatomidi("5.txt", "5.mid")
datatomidi("6.txt", "6.mid")
datatomidi("7.txt", "7.mid")
datatomidi("8.txt", "8.mid")
datatomidi("9.txt", "9.mid")
datatomidi("10.txt", "10.mid")
# miditodata("re.mid", "de.txt")
