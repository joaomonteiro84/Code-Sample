# Project #3 – Multithreading
* Lab machine name used to test your program: apollo.cselabs.umn.edu
* Project group number: 23 
* Group member names and x500 addresses:
    * John Spruth   (sprut006)
    * Jacob Johnson (joh19042)
    * Joao Monteiro (monte092)
* Whether to complete the extra task: We completed the extra credit by adding a semaphore initialized with the bounded buffer size.
* Members’ individual contributions
    * Most of the project was completed by doing group programming. We had nightly group coding sessions over zoom from 7pm to around 9pm. 
    * Minor coding parts of the project was divided among us. We wrote below a list of the activities that was completed, and who was responsible for them.

    * Create a unbounded queue data structure - group programming
    * consumer and parse functions in consumer.c - group programming
    * producer function in producer.c - group programming
    * enhancement of error handling functions - group programming
    * implamentation of the bounded buffer - group programming
    * write README file - group programming

    * Code to read input and flags from the command line - check the input arguments and print error messages - Jacob Johnson
    * Code to write the log - Joao Monteiro
    * compute the asset change of the bank (occurs in the main.c) - John Spruth
      

* Any assumptions outside this document
    * Each line in the input file is no longer than 1024 characters

* How to compile your program:  
    make clean
    make

* How to run:
    ./bank #consumers inputFile [option] [#queueSize]  
    where option can be -b, -p, or -bp and #queueSize is only included if -b or -bp is used. b is for the bounded buffer and p is for printing the log out. 

* How to test:
    make test 

* How to test bounded buffer
   make testextracredit

* Screenshot of the terminal after running the tests
![screenshot](terminal.png)



