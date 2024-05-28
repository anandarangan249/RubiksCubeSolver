import tkinter as tk
import numpy as np
from math import *

# Global Variables
# Colors
LIGHT_YELLOW    = '#FFFEE0'
BLACK           = '#000000'

SIDE_LENGTH = 1.0
EPSILON = 1e-4

class Solution():
##  Read Input
    def readInput(self,data):
        """
        Extract the information of point coordinates, edges, faces, and face-colors

        Args:
        -   data: A numpy array of shape (1+vertices_num+faces_num, 1) representing the information

        Returns:
        -   points: A list of shape (vertices_num,3) representing the vertices and their coordinates
        -   edges: A list of the form [..., (vi, vj), ...] representing a unique edge between vertex i and j
        -   faces: A list of the form [..., (vi, vj, vk, color), ...] representing a face containing vertices i, j and k
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
            v1, v2, v3, color = data[i][0].split(',')
            faces.append([int(v1)-1, int(v2)-1, int(v3)-1, color])
            if (int(v1)-1, int(v2)-1) not in edges:
                edges.append([int(v1)-1, int(v2)-1])
            if (int(v1)-1, int(v3)-1) not in edges:
                edges.append([int(v1)-1, int(v3)-1])
            if (int(v2)-1, int(v3)-1) not in edges:
                edges.append([int(v2)-1, int(v3)-1])

        return points, edges, faces

##   Functions related to UI
    def setupCanvas(self, root, width, height, points, edges, faces, mouse_speed):
        """
        Setup Canvas for GUI

        Args:
        -   root: Represents the Tk instance
        -   width: Represents the size of the canvas in the X dimension.
        -   height: Represents the size of the canvas in the X dimension.
        -   points: A list of shape (vertices_num,3) representing the vertices and their coordinates
        -   edges: A list of the form [..., (vi, vj), ...] representing a unique edge between vertex i and j
        -   faces: A list of the form [..., (vi, vj, vk, color), ...] representing a face containing vertices i, j and k
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
        self.bindMouse(faces)
        self.fillColor(faces)

    def bindMouse(self, faces):
        """
        Binds mouse events to corresponding function calls

        Args:
        -   faces: A list of the form [..., (vi, vj, vk), ...] representing a face containing vertices i, j and k
        
        Returns:
        -   None
        """

        self.canvas.bind("<Button-1>", self.mouseClick)
        self.canvas.bind("<B1-Motion>", lambda event, arg = faces: self.mouseMovement(event, arg))

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

    def mouseMovement(self, event, faces):
        """
        Handler function for the event mouse motion while Button-1 is pressed

        Args:
        -   event: Event object containingthe x and y coordinate of the mouse at the time of the event Button-1 click
        -   faces: A list of the form [..., (vi, vj, vk), ...] representing a face containing vertices i, j and k

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
        self.fillColor(faces)
        self.mouseClick(event)

    def fillColor(self, faces):
        """
        Visualizes the 3D object with filled faces

        Args:
        -   faces: A list of the form [..., (vi, vj, vk, color), ...] representing a face containing vertices i, j and k

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

        # Display the faces
        for i in range(len(faces)):
            # Extract the vertices
            point1 = self.points[:,faces[i][0]]
            point2 = self.points[:,faces[i][1]]
            point3 = self.points[:,faces[i][2]]

            # Scale point to fit the window and move origin from top left to the center of the window
            canvasPoint1 = scale*point1 + np.array([w, h, 0])
            canvasPoint2 = scale*point2 + np.array([w, h, 0])
            canvasPoint3 = scale*point3 + np.array([w, h, 0])

            # Determine how much the normal vector aligns with the Z axis 
            # Dot product of the normal vector and the Z axis
            factor = self.computeAngleWithZ(canvasPoint1, canvasPoint2, canvasPoint3)

            # Find the color corresponding to the alignment
            fill_color = faces[i][3]

            # Fill the face only if it is visible
            if factor >= 0:
                self.canvas.create_polygon(canvasPoint1[0], canvasPoint1[1], 
                                           canvasPoint2[0], canvasPoint2[1], 
                                           canvasPoint3[0], canvasPoint3[1], fill=fill_color)

                # Draw the edges
                if self.isLineOnEdgeOfCube(point1, point2):
                    self.canvas.create_line(canvasPoint1[0], canvasPoint1[1], canvasPoint2[0], canvasPoint2[1], fill = BLACK, width=3)
                if self.isLineOnEdgeOfCube(point2, point3):
                    self.canvas.create_line(canvasPoint2[0], canvasPoint2[1], canvasPoint3[0], canvasPoint3[1], fill = BLACK, width=3)
                if self.isLineOnEdgeOfCube(point3, point1):
                    self.canvas.create_line(canvasPoint3[0], canvasPoint3[1], canvasPoint1[0], canvasPoint1[1], fill = BLACK, width=3)

    def createCanvas(self):
        """
        Create the canvas for GUI

        Args:
        -   None

        Returns:
        -   None
        """

        self.canvas = tk.Canvas(self.root, bg=LIGHT_YELLOW, width=self.width, height=self.height)
        self.canvas.pack()

##   Math Functions
    def isLineOnEdgeOfCube(self, point1, point2):
        """
        Given two points, it finds if the line segment connecting them is an edge or a diagonal

        Assumption: uses the global const SIDE_LENGTH

        Args:
        -   point1, point2, - 2 numpy arrays of shape (3,1) corresponding to the endpoints of the line

        Returns:
        -   False if length of line is greater than 1.4 * SIDE_LENGTH; True otherwise
        """
        if np.linalg.norm(point1 - point2) - SIDE_LENGTH < EPSILON:
            return True
        return False

    def computeAngleWithZ(self, point1, point2, point3):
        """
        Computes how much the normal vector of the plane containing three points aligns with the Z axis

        Assumption1: The given 3D image is convex

        Assumption2: Origin is an interior point to the object
        A solution to this could be to use the centroid of the object as refernce to compute the
        outward normal vector

        Args:
        -   point1, point2, point3 - 3 numpy arrays of shape (3,1) corresponding to the three vertices
        of a face

        Returns:
        -   A number between -1.0 and 1.0 corresponding to the level of alignment with Z axis
        """

        # Form 2 vectors with same origin
        vector1 = point1 - point2
        vector2 = point3 - point2

        # Get the unit normal vector by cross product and normalization
        normal_vector = np.cross(vector1, vector2)
        normal_vector /= np.linalg.norm(normal_vector)

        w = self.canvas.winfo_width()/2         # X-coordinate of the origin
        h = self.canvas.winfo_height()/2        # Y-coordinate of the origin

        # Since origin is an interior point, if the normal vector aligns with 
        # the point1 (or point2 or point3) vector, then it is an outward facing vector 

        # Note: Z axis in Tkinter Canvas is away from the user, so the signs are inverted
        if normal_vector @ (point1-np.array([w,h,0])) < 0: # The vector is an outward normal
            return -normal_vector[2]

        # The vector is an inward normal
        return normal_vector[2]

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
