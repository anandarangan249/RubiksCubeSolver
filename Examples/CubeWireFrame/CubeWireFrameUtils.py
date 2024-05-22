import tkinter as tk
import numpy as np
from math import *

class Solution():
##  Read Input
    def readInput(self,data):
        """
        Extract the information of point coordinates, edges, and faces

        Args:
        -   data: A numpy array of shape (1+vertices_num+faces_num, 1) representing the information

        Returns:
        -   points: A list of shape (vertices_num,3) representing the vertices and their coordinates
        -   edges: A list of the form [..., (vi, vj), ...] representing a unique edge between vertex i and j
        -   faces: A list of the form [..., (vi, vj, vk), ...] representing a face containing vertices i, j and k
        """

        points = []
        edges = []
        faces = []
        
        # Get the number of vertices from the first line
        vertices_num, _ = data[0][0].split(',')

        # Get the point coordinates from lines 2 - vertices_num+1      
        for i in range(1,int(vertices_num)+1):
            _, x, y, z = data[i][0].split(',')
            points.append([float(x), float(y), float(z)])

        # Get the edge and face information from lines vertices_num+2 - vertices_num+faces_num+1
        for i in range(int(vertices_num)+1,len(data)):
            v1, v2, v3 = data[i][0].split(',')
            faces.append([int(v1)-1, int(v2)-1, int(v3)-1])
            if (int(v1)-1, int(v2)-1) not in edges:
                edges.append([int(v1)-1, int(v2)-1])
            if (int(v1)-1, int(v3)-1) not in edges:
                edges.append([int(v1)-1, int(v3)-1])
            if (int(v2)-1, int(v3)-1) not in edges:
                edges.append([int(v2)-1, int(v3)-1])

        return points, edges, faces
        
##   Functions related to UI
    def setupCanvas(self, root, width, height, points, edges, mouse_speed):
        """
        Setup Canvas for GUI

        Args:
        -   root: Represents the Tk instance
        -   width: Represents the size of the canvas in the X dimension.
        -   height: Represents the size of the canvas in the X dimension.
        -   points: A list of shape (vertices_num,3) representing the vertices and their coordinates
        -   edges: A list of the form [..., (vi, vj), ...] representing a unique edge between vertex i and j
        -   mouse_speed: Represents the rotation speed of the object w.r.t mouse speed (typically between 0.0 - 1.0)

        Returns:
        -   None
        """

        # Initializing variables
        self.root = root
        self.width = width
        self.height = height
        self.points = np.transpose(points)
        self.epsilon = 0.01*mouse_speed
        
        # Implementation
        self.createCanvas()
        self.displayObject(edges)
        self.bindMouse(edges)
        
    def bindMouse(self, edges):
        """
        Binds mouse events to corresponding function calls

        Args:
        -   edges: A list of the form [..., (vi, vj), ...] representing a unique edge between vertex i and j

        Returns:
        -   None
        """

        self.canvas.bind("<Button-1>", self.mouseClick)
        self.canvas.bind("<B1-Motion>", lambda event, arg = edges: self.mouseMovement(event, arg))
        
    def mouseClick(self, event):
        """
        Handler function for the event mouse Button-1 click

        Args:
        -   event: Event object containingthe x and y coordinate of the mouse pointer at the time of the event

        Returns:
        -   None
        """

        self.previous_x = event.x
        self.previous_y = event.y

    def mouseMovement(self, event, edges):
        """
        Handler function for the event mouse motion while Button-1 is pressed

        Args:
        -   event: Event object containingthe x and y coordinate of the mouse at the time of the event Button-1 click
        -   edges: A list of the form [..., (vi, vj), ...] representing a unique edge between vertex i and j

        Returns:
        -   None
        """

        # Determine change in mouse position
        dy = self.previous_y - event.y 
        dx = self.previous_x - event.x 

        # Apply Rotation to all the points
        self.points = self.rotateY(-dx*self.epsilon, self.points)
        self.points = self.rotateX(dy*self.epsilon, self.points)

        # Display the object and call mouseClick event
        self.displayObject(edges)
        self.mouseClick(event)
    
    def displayObject(self, edges):
        """
        Visualizes the 3D object

        Args:
        -   edges: A list of the form [..., (vi, vj), ...] representing a unique edge between vertex i and j

        Returns:
        -   None
        """

        # Initialization
        self.canvas.delete('all')
        w = self.canvas.winfo_width()/2         # X-coordinate of origin
        h = self.canvas.winfo_height()/2        # Y-coordinate of origin
        
        # Scale all the points by the farthest point so the worst case object fills approximately half the window
        farthest_point_dist = max(np.linalg.norm(self.points, axis=0))
        scale = 0.6*min(w,h)/farthest_point_dist

        # Display edges
        for i in range(len(edges)):
            # Scale point to fit the window and move origin from top left to the center of the window
            point1 = scale*self.points[:,edges[i][0]] + np.array([w, h, 0])
            point2 = scale*self.points[:,edges[i][1]] + np.array([w, h, 0])

            # Draw the ith edge 
            self.canvas.create_line(point1[0], point1[1], point2[0], point2[1], fill = 'blue', width=3)

            # Draw the vertices corresponding to the ith edge
            self.canvas.create_oval(point1[0], point1[1], point1[0], point1[1], outline='blue', width=10)
            self.canvas.create_oval(point2[0], point2[1], point2[0], point2[1], outline='blue', width=10)
        
        # Paint the closest point red for easy visualization
        # (Wireframe model is subject to optical illusions)
        closest_point = np.argmax(self.points[2,:])
        point5 = scale*self.points[:,closest_point] + np.array([w, h, 0])
        self.canvas.create_oval(point5[0], point5[1], point5[0], point5[1], outline='red', width=10)
        
    def createCanvas(self):
        """
        Create the canvas for GUI

        Args:
        -   None

        Returns:
        -   None
        """

        self.canvas = tk.Canvas(self.root, bg='white', width=self.width, height=self.height)
        self.canvas.pack()
    
##   Rotation Functions
    def rotateX(self, theta, matrix):
        """
        Rotates the matrix by theta degrees counter clockwise along the X axis

        Args:
        -   theta: The angle to rotate the object by
        -   matrix: A numpy array of shape (3, vertices_num) representing the vertices of the object

        Returns:
        -   A numpy array of shape (3, vertices_num) representing the vertices of the rotated object
        """

        rotation_matrix = np.array([[1, 0, 0], [0, cos(theta), -sin(theta)], [0, sin(theta), cos(theta)]])
        return rotation_matrix @ matrix
    
    def rotateY(self, theta, matrix):
        """
        Rotates the matrix by theta degrees counter clockwise along the Y axis

        Args:
        -   theta: The angle to rotate the object by
        -   matrix: A numpy array of shape (3, vertices_num) representing the vertices of the object

        Returns:
        -   A numpy array of shape (3, vertices_num) representing the vertices of the rotated object
        """

        rotation_matrix = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
        return rotation_matrix @ matrix
