/*test machine: apollo.cselabs.umn.edu
* group number: G23
* name: Joao Monteiro, John Spruth, Jacob Johnson
* x500: monte092, sprut006, joh19042 */

#include "producer.h"

extern queue_t buffer;         // global unbounded queue buffer        
extern int numberConsumers;           //global int variable that will hold the number of consumer threads

/**
  * Producer thread will read from the file and write data to 
 * the end of the shared queue
 */
void *producer(void *arg){    
    

    if (printLog) {
        fprintf(fpLog, "producer\n");   //print in the log when producer is launched
    }
    
  
    char transactionLine[chunkSize];      //string that will hold transcation line


    FILE *fp = getFilePointer((char *) arg);
    
    //check if input file exists
    if (fp == NULL) {
        fprintf(stderr, "File %s not found: %d\n", (char *) arg, errno);
        exit(EXIT_FAILURE);
    } 
    
    int line = 0;                                                      //initiate line number
    while (getLineFromFile(fp, transactionLine, chunkSize) != -1) {

        //check if there was any error reading line
        if (errno == EINVAL || errno == ENOMEM) {
            perror("Failed to read line from input file.");
            exit(EXIT_FAILURE);
        }

        if (printLog) {
            fprintf(fpLog, "producer: line %d\n", line);               //print line producer is about to add to the queue
            fflush(fpLog);
            line++;
        } 
       //enqueue line in the buffer. 
       //notice the mutex lock is inside enqueue function  (in utils.c)
        if (enqueue(&buffer, transactionLine) == -1) { //if enqueue returns -1 then an error occured.
            exit(EXIT_FAILURE);
        }                    
    }
    
    //send an EOF signal for each consumer
    for (int i = 0; i < numberConsumers; i++) {
        if (printLog) {
            fprintf(fpLog, "producer: -1\n");  //print in the log EOF
            fflush(fpLog);
            line++;
        } 
        // enqueue EOF signal in the buffer
        //notice the mutex lock is inside enqueue function  (in utils.c)
        if (enqueue(&buffer, "-1") == -1) {//if enqueue returns -1 then an error occured.
            exit(EXIT_FAILURE);
        }
    }
    
    return NULL; 
}
