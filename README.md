Vectors for Education
=====================
What is this?
-------------
This is a library/framework for building vector-based models to represent knowledge and to make predictions from that.
This also contains a small library for deploying this system to AWS.

This uses Theano and all its dependences (numpy, scipy)

How do I install/use this?
--------------------------

email me at yuerany@andrew.cmu.edu

Details
-------
This library is very much based on the Theano deep learning tutorials from LISA lab UMontreal. A little effort has been made to structure the code in a logical way; this is an ongoing effort. Several of the modules are still pretty tightly coupled mainly because I haven't really figured out how to isolate the training processes ("annealing", logging, cross-validation, etc.) from the logic of the network itself especially with all these entangled Theano variables.
