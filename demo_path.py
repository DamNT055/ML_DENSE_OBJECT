from pathlib import Path
import os 

def get_path():
    print("path file: ", Path(__file__))
    print("path cwd: ", Path.cwd())
    print("path home: ", Path.home())
    print("absolute :", Path(__file__).absolute())
    print("resolve: ", Path("..").resolve())
    
if __name__ == "__main__":
    get_path()