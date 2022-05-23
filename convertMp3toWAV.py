from os import path
import sys
from pydub import AudioSegment

def convert(src: str, dst: str):
    # convert wav to mp3                                                            
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")

if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) == 3:
        print(sys.argv[1], sys.argv[2])
        convert(sys.argv[1], sys.argv[2])
    else:
        print()
        src = input("Enter mp3 name:\n")
        dst = input("Enter output file location:\n")
        convert(src, dst)