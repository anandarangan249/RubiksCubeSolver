import tkinter as tk
import pandas as pd
from CubeWireFrameUtils import *

if __name__ == '__main__':
    ## Initializations (Feel free to play around the numbers)
    root = tk.Tk()
    height = 600            # Desired height of the window
    width = 600             # Desired width of the window
    mouse_speed = 0.5       # Enter a positive float (desirably <= 1.0)
    
    ## Fetch data from object.txt file 
    data = pd.read_csv("object.txt", sep = " ", header=None)
    data = data.to_numpy()

    ## Implement the solution
    A = Solution()
    points, edges, faces = A.readInput(data)
    A.setupCanvas(root, width, height, points, edges, mouse_speed)
    root.title("Cube Wire Frame")
    root.mainloop()
