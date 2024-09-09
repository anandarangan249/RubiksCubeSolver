import tkinter as tk
import numpy as np
from math import *

# Global Variables
# Colors
LIGHT_YELLOW    = "#FFFEE0"
BLACK           = "#000000"
WHITE           = "#FFFFFF"
YELLOW          = "#FFFF00"
RED             = "#EE0000"
ORANGE          = "#FFA500"
BLUE            = "#0000FF"
GREEN           = "#00EE00"

ColorMap = {
    "+X": WHITE,
    "-X": YELLOW,
    "+Y": RED,
    "-Y": ORANGE,
    "+Z": GREEN,
    "-Z": BLUE,
    "O" : BLACK, 
}

SIDE_LENGTH = 1.0
EPSILON = 1e-4

class Solution():
## Constructor
    def __init__(self):

        # Tk related variables
        self.root = None
        self.width = 0.0
        self.height = 0.0
        self.epsilon = 0.0

        # Cube related variables
        self.points = np.array([])
        self.colors = []

        # Camera related parameters
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.zoom = 1.0

##  Read Input
    def intializeData(self,data):
        """
        Extract the information of point coordinates, edges, faces, and face-colors

        Args:
        -   data: A numpy array of shape (1+vertices_num+faces_num, 1) representing the information

        Returns:
        -   points: A list of shape (vertices_num,3) representing the vertices and their coordinates
        -   edges: A list of the form [..., (vi, vj), ...] representing a unique edge between vertex i and j
        -   faces: A list of the form [..., (vi, vj, vk), ...] representing a face containing vertices i, j and k
        """

        cube_points = []
        cube_edges = []
        cube_faces = []

        points = []
        edges = []
        faces = []
        colors = []

        # Get the number of vertices from the first line
        vertices_num, _ = data[0][0].split(",")

        # Get the point coordinates from lines 2 - vertices_num+1
        for i in range(1,int(vertices_num)+1):
            _, x, y, z = data[i][0].split(",")
            cube_points.append([float(x), float(y), float(z)])

        # Get the edge and face information from lines vertices_num+2 - vertices_num+faces_num+1
        for i in range(int(vertices_num)+1,len(data)):
            v1, v2, v3 = data[i][0].split(",")
            cube_faces.append([int(v1)-1, int(v2)-1, int(v3)-1])
            if (int(v1)-1, int(v2)-1) not in edges:
                cube_edges.append([int(v1)-1, int(v2)-1])
            if (int(v1)-1, int(v3)-1) not in edges:
                cube_edges.append([int(v1)-1, int(v3)-1])
            if (int(v2)-1, int(v3)-1) not in edges:
                cube_edges.append([int(v2)-1, int(v3)-1])

        # For 2x2 cube, offset the initial cube points by 0.5 * SIDE_LENGTH
        cube_points = (np.array(cube_points) + [-0.5, -0.5, -0.5]).tolist()

        points.extend(cube_points)
        edges.extend(cube_edges)
        faces.extend(cube_faces)

        # Duplicate the cube data to form a 2x2 cube
        for i in range(1,8):
            offset_vector = [(i // 4) % 2, (i // 2) % 2, i % 2]
            points.extend((np.array(cube_points) + offset_vector).tolist())
            edges.extend((np.array(cube_edges) + i*8).tolist())
            faces.extend((np.array(cube_faces) + i*8).tolist())
        
        # Determine face colors
        for face in faces:
            colors.append(self.getFaceColor(points, face[0], face[1], face[2]))

        return points, edges, faces, colors

##   Functions related to UI
    def setupCanvas(self, root, width, height, points, edges, faces, colors, mouse_speed):
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
        self.colors = colors

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

        # Save the camera rotation and apply while filling color
        self.rot_y += -dx*self.epsilon
        self.rot_x += dy*self.epsilon

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

        # Apply the camera transform to the points
        cameraPoints = self.rotateY(self.rot_y, self.points)
        cameraPoints = self.rotateX(self.rot_x, cameraPoints)

        # Initialization
        self.canvas.delete("all")
        w = self.canvas.winfo_width()/2         # X-coordinate of origin
        h = self.canvas.winfo_height()/2        # Y-coordinate of origin
        
        # Scale all the points by the farthest point so the worst case object fills approximately half the window
        farthest_point_dist = max(np.linalg.norm(cameraPoints, axis=0))
        scale = 0.6*min(w,h)/farthest_point_dist

        # Display the faces
        for i in range(len(faces)):
            # Extract the vertices
            point1 = cameraPoints[:,faces[i][0]]
            point2 = cameraPoints[:,faces[i][1]]
            point3 = cameraPoints[:,faces[i][2]]

            # Scale point to fit the window and move origin from top left to the center of the window
            canvasPoint1 = scale*point1 + np.array([w, h, 0])
            canvasPoint2 = scale*point2 + np.array([w, h, 0])
            canvasPoint3 = scale*point3 + np.array([w, h, 0])

            # Calculate the local interior point for the smaller cube
            cubeIdx = i // 12
            cubeCenter = np.mean(cameraPoints[:, cubeIdx * 8 : (cubeIdx + 1) * 8], axis = 1)
            canvasCubeCenter = scale * cubeCenter + np.array([w, h, 0])

            # Determine how much the normal vector aligns with the Z axis 
            # Dot product of the normal vector and the Z axis
            factor = self.computeAngleWithZ(canvasPoint1, canvasPoint2, canvasPoint3, canvasCubeCenter)

            # Determine if it is an outside face
            outside_face = self.isOutsideFace(point1, point2, point3)

            # Find the color corresponding to the alignment
            fill_color = self.colors[i]

            # Fill the face only if it is an outside face and visible
            if outside_face and factor >= 0:
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

## Cube Related Functions
    def getFaceColor(self, pointsList, v1, v2, v3):
        """
        Given three points of a face of the cube, return the color corresponding to the face.
        Return Black, if it is one of the inside facing edges

        Assumption: uses the global map colorMap

        Args:
        -   pointsList - list of size [number of points, 3] corresponding to each vertex of the cube
        -   v1, v2, v3 - 3 int corresponding to the edge for which the color is being determined

        Returns:
        -   String representing the color of the side
        """

        # Get the point coordinates from the vertices
        point1 = np.array(pointsList[v1])
        point2 = np.array(pointsList[v2])
        point3 = np.array(pointsList[v3])

        # Assumption centroid is an inside point
        overallCentroid = np.mean(pointsList, axis = 0)
        cubeIdx = v1 // 8
        cubeCentroid = np.mean(pointsList[cubeIdx * 8 : (cubeIdx + 1) * 8], axis = 0)

        # Get the normal vector
        normal_vector = self.computeOutwardNormal(point1, point2, point3, cubeCentroid)

        # An outside face is determined by its distance from centroid 
        # For 2x2, outside face is ATLEAST SIDE_LENGTH away from centroid
        faceCentroid = (point1 + point2 + point3) / 3
        if np.linalg.norm(faceCentroid - overallCentroid) < SIDE_LENGTH:
            return ColorMap["O"] 
        

        if (normal_vector[0] > EPSILON):
            return ColorMap["+X"]
        elif (normal_vector[0] < -EPSILON):
            return ColorMap["-X"]
        elif (normal_vector[1] > EPSILON):
            return ColorMap["+Y"]
        elif (normal_vector[1] < -EPSILON):
            return ColorMap["-Y"]
        elif (normal_vector[2] > EPSILON):
            return ColorMap["+Z"]
        elif (normal_vector[2] < -EPSILON):
            return ColorMap["-Z"]
        else:
            print("ERROR!!    getFaceColor    normal_vector matches no axis!!  normal_vector = [" + normal_vector[0] + ", " + 
                                                                                                    normal_vector[1] + ", " + 
                                                                                                    normal_vector[2])

##   Math Functions
    def isOutsideFace(self, point1, point2, point3):
        """
        Given three points, this function finds if the plane formed by them is on the 
        outside of the cube.

        Assumption: uses the global const SIDE_LENGTH

        Args:
        -   point1, point2, point3 - 3 numpy arrays of shape (3,1) corresponding to the points in the face

        Returns:
        -   For 2x2, True if origin lies on the plane; False otherwise
        """
        
        normal_vector = np.cross(point2 - point1, point3 - point1)
        origin = np.array([0.0, 0.0, 0.0])
        if abs(np.dot(normal_vector, origin - point1)) < EPSILON:
            return False
        return True

    def isLineOnEdgeOfCube(self, point1, point2):
        """
        Given two points, it finds if the line segment connecting them is an edge or a diagonal

        Assumption: uses the global const SIDE_LENGTH

        Args:
        -   point1, point2 - 2 numpy arrays of shape (3,1) corresponding to the endpoints of the line

        Returns:
        -   False if length of line is greater than 1.4 * SIDE_LENGTH; True otherwise
        """
        if np.linalg.norm(point1 - point2) - SIDE_LENGTH < EPSILON:
            return True
        return False
    
    def computeOutwardNormal(self, point1, point2, point3, interiorPoint):
        """
        Computes the outward normal of the plane containing the given 3 points

        Args:
        -   point1, point2, point3 - 3 numpy arrays of shape (3,1)
        -   interiorPoint - numpy array of shape (3,1)

        Returns:
        -   A numpy array of shape (3,1) corresponding to the normal of the plane
        """

        # Form 2 vectors with same origin
        vector1 = point1 - point2
        vector2 = point3 - point2

        # Get the unit normal vector by cross product and normalization
        normal_vector = np.cross(vector1, vector2)
        normal_vector /= np.linalg.norm(normal_vector)
        # Since origin is an interior point, if the normal vector aligns with 
        # the point1 (or point2 or point3) vector, then it is an outward facing vector 

        if normal_vector @ (point1-interiorPoint) < 0: # The vector is an inward normal normal
            normal_vector = -1.0 * normal_vector
        return normal_vector

    def computeAngleWithZ(self, point1, point2, point3, interiorPoint):
        """
        Computes how much the outward normal vector of the plane containing three points 
        aligns with the Z axis, given a local interior point

        Args:
        -   point1, point2, point3 - 3 numpy arrays of shape (3,1) corresponding to the three vertices
        of a face
        -   interiorPoint - numpy array of shape (3,1) representing a local interior point

        Returns:
        -   A number between -1.0 and 1.0 corresponding to the level of alignment with Z axis
        """

        # Get the outward normal vector
        normal_vector = self.computeOutwardNormal(point1, point2, point3, interiorPoint)

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
