# My Feed Forward Neural Network Library

## Motivation

Building Neural Network from scratch and Application to Computational Mechanics

## Details on Implemented Code
-  Programming Language : Python 3.0
-  Implementation : Objected –Oriented Programming in Python 
-  Unit-Testing
-  Verification of results with FFNET package
-  Imported Libraries:
    -  Numpy
    -  Matplotlib.pyplot
    -  time
    -  unittest
   
 ## Part - I 
 Development of ANN Library
 
 ## Part - II
 Identifying Parameters of the Bigoni Piccolroaz Criterion
 
## Bigoni–Piccolroaz yield surface
The Bigoni–Piccolroaz yield criterion [^1] is a seven-parameter surface defined by

![image](https://user-images.githubusercontent.com/49998891/124035968-98e48b00-d9fd-11eb-940c-879703615a1d.png)
 
where **F(p)** is the "meridian" function

![image](https://user-images.githubusercontent.com/49998891/124036111-c8939300-d9fd-11eb-8e4b-877ddcf22435.png)

![image](https://user-images.githubusercontent.com/49998891/124036138-d6491880-d9fd-11eb-976f-f9d0abf352cd.png)

describing the pressure-sensitivity and **g(&theta;)** is the "deviatoric" function 

![image](https://user-images.githubusercontent.com/49998891/124036343-2627df80-d9fe-11eb-8555-e7482534580b.png)

describing the Lode-dependence of yielding. The seven, non-negative material parameters:

 ![image](https://user-images.githubusercontent.com/49998891/124036368-2e801a80-d9fe-11eb-88f3-f27fd3d09250.png)

define the shape of the meridian and deviatoric sections.

This criterion represents a smooth and convex surface, which is closed both in hydrostatic tension and compression and has a drop-like shape, particularly suited to describe frictional and granular materials. This criterion has also been generalized to the case of surfaces with corners.

[^1]: https://en.wikipedia.org/wiki/Yield_surface
