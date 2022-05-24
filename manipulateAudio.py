from pydub import AudioSegment
from pydub.generators import WhiteNoise
import os

directory = 'Data/Audio'
 
# iterate over files in
# that directory
for filename in os.listdir(directory):
    sound = AudioSegment.from_wav(directory+'/'+filename)
    # increase vol
    increasedVol = sound + 10
    increasedVol.export(directory+'/'+filename[:-4]+'1'+'.wav', format="wav")
    # decrease vol
    decreasedVol = sound - 10
    decreasedVol.export(directory+'/'+filename[:-4]+'2'+'.wav', format="wav")
    # add short silence before
    addSilence1 = AudioSegment.silent(duration=500) + sound
    addSilence1.export(directory+'/'+filename[:-4]+'3'+'.wav', format="wav")
    # add medium silence before
    addSilence2 = AudioSegment.silent(duration=1500) + sound
    addSilence2.export(directory+'/'+filename[:-4]+'4'+'.wav', format="wav")
    # add long silence before
    addSilence3 = AudioSegment.silent(duration=2500) + sound
    addSilence3.export(directory+'/'+filename[:-4]+'5'+'.wav', format="wav")
    # add noise
    noise = WhiteNoise().to_audio_segment(duration=10000) - 45
    noisy = sound.overlay(noise)
    noisy.export(directory+'/'+filename[:-4]+'6'+'.wav', format="wav")
    noisy = increasedVol.overlay(noise)
    noisy.export(directory+'/'+filename[:-4]+'7'+'.wav', format="wav")
    noisy = decreasedVol.overlay(noise)
    noisy.export(directory+'/'+filename[:-4]+'8'+'.wav', format="wav")
    noisy = addSilence1.overlay(noise)
    noisy.export(directory+'/'+filename[:-4]+'9'+'.wav', format="wav")
    noisy = addSilence2.overlay(noise)
    noisy.export(directory+'/'+filename[:-4]+'10'+'.wav', format="wav")
    noisy = addSilence3.overlay(noise)
    noisy.export(directory+'/'+filename[:-4]+'11'+'.wav', format="wav")
    
