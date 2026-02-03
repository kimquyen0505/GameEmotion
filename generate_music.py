#!/usr/bin/env python3
"""
Generate game music for EmotionGame using pydub
Creates upbeat, kid-friendly background music and sound effects
"""

from pydub import AudioSegment
from pydub.generators import Sine, Square, Triangle
from pydub.utils import make_chunks
import os
import sys

# Ensure output directory exists
os.makedirs('assets', exist_ok=True)

def create_bgm():
    """Create upbeat background music loop (~8 seconds)"""
    # 120 BPM = 2 beats per second
    beat_duration = 500  # 500ms per beat
    
    # Create a cheerful melody using sine waves
    # Key of C Major for happy feel
    notes = {
        'C': 262,  # Hz
        'D': 294,
        'E': 330,
        'F': 349,
        'G': 392,
        'A': 440,
        'B': 494,
        'C_high': 524,
    }
    
    # Melody pattern: Happy, bouncy
    melody_pattern = [
        ('C', 250),
        ('E', 250),
        ('G', 250),
        ('C_high', 250),
        ('G', 250),
        ('E', 250),
        ('C', 250),
        ('D', 250),
        ('E', 250),
        ('F', 250),
        ('G', 500),
        ('G', 500),
    ]
    
    # Create bass line
    bass_pattern = [
        ('C', 500),
        ('C', 500),
        ('F', 500),
        ('F', 500),
        ('G', 500),
        ('G', 500),
        ('C', 500),
        ('C', 500),
    ]
    
    melody = AudioSegment.silent(0)
    bass = AudioSegment.silent(0)
    
    # Build melody
    for note_name, duration in melody_pattern:
        freq = notes[note_name]
        note = Sine(freq).to_audio_segment(duration=duration).fade_in(50).fade_out(50)
        melody += note
    
    # Build bass
    for note_name, duration in bass_pattern:
        freq = notes[note_name] // 2  # Lower octave for bass
        note = Square(freq).to_audio_segment(duration=duration).fade_in(50).fade_out(100)
        bass += note
    
    # Overlay bass under melody (quieter)
    bass = bass - 6  # Reduce volume
    bgm = melody.overlay(bass)
    
    # Add echo effect
    echo = bgm - 10
    delayed = AudioSegment.silent(200) + echo
    bgm = bgm.overlay(delayed)
    
    # Export
    bgm.export('assets/bgm.wav', format='wav')
    print("✓ Background music created: bgm.wav")

def create_beat():
    """Create a beat/tick sound"""
    beat = Square(440).to_audio_segment(duration=100).fade_in(10).fade_out(20)
    beat.export('assets/beat.wav', format='wav')
    print("✓ Beat sound created: beat.wav")

def create_correct_sound():
    """Create a winning/correct sound"""
    # Ascending melody
    sound = AudioSegment.silent(0)
    notes = [262, 330, 392, 524]  # C, E, G, C (high)
    
    for freq in notes:
        note = Sine(freq).to_audio_segment(duration=100).fade_in(10).fade_out(10)
        sound += note
    
    sound.export('assets/correct.wav', format='wav')
    print("✓ Correct sound created: correct.wav")

def create_wrong_sound():
    """Create a wrong/fail sound"""
    # Descending buzz
    sound = AudioSegment.silent(0)
    notes = [349, 294, 262]  # F, D, C (descending)
    
    for freq in notes:
        note = Square(freq).to_audio_segment(duration=80).fade_in(10).fade_out(10)
        sound += note
    
    sound.export('assets/wrong.wav', format='wav')
    print("✓ Wrong sound created: wrong.wav")

def create_success_sound():
    """Create a success/celebration sound"""
    # Triumphant chord
    freq1 = Sine(262).to_audio_segment(duration=200)  # C
    freq2 = Sine(330).to_audio_segment(duration=200)  # E
    freq3 = Sine(392).to_audio_segment(duration=200)  # G
    
    chord = freq1.overlay(freq2).overlay(freq3)
    chord = chord.fade_in(20).fade_out(50)
    
    success = chord
    success += AudioSegment.silent(50)
    success += chord.fade_in(20).fade_out(50)
    
    success.export('assets/success.wav', format='wav')
    print("✓ Success sound created: success.wav")

def create_cue_sound():
    """Create a cue/alert sound"""
    freq1 = Sine(523).to_audio_segment(duration=100)  # High C
    freq2 = Sine(659).to_audio_segment(duration=100)  # High E
    
    cue = freq1 + freq2
    cue = cue.fade_in(10).fade_out(20)
    
    cue.export('assets/cue.wav', format='wav')
    print("✓ Cue sound created: cue.wav")

if __name__ == '__main__':
    print("Generating game music and sounds...")
    try:
        create_bgm()
        create_beat()
        create_correct_sound()
        create_wrong_sound()
        create_success_sound()
        create_cue_sound()
        print("\n✨ All audio files generated successfully!")
    except ImportError:
        print("Error: pydub not installed. Install it with:")
        print("  pip install pydub")
        print("Also ensure ffmpeg is installed:")
        print("  brew install ffmpeg")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating audio: {e}")
        sys.exit(1)
