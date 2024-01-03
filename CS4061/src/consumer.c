/*test machine: apollo.cselabs.umn.edu
* group number: G23
* name: Joao Monteiro, John Spruth, Jacob Johnson
* x500: monte092, sprut006, joh19042 */

#include "consumer.h"
#include <ctype.h>

extern queue_t buffer;                    // global queue buffer   
extern sem_t bufferSizeSemaphore;
int lineNumber = 0;                           // gloal variable indicating the line number consumer is working on

/**
 * parse lines from the queue, calculate balance change
 * and update to global array
 */
int parse(char *line){
    
    int ret_code = 0;
    char *tokenized_line; 
    int customerID;
    double transactionValue;
    double runningSum = 0.0; 
    int error;   

    // get customer id
    tokenized_line = strtok (line, ",");
    errno = 0;
    sscanf(tokenized_line, "%d", &customerID);
    
    if (customerID < 0 ) {         //this means EOF special packet
        return -1;	
    }   
 
    tokenized_line = strtok (NULL, ",");
    while (tokenized_line != NULL) {                            // Go through the entire line
        sscanf(tokenized_line, "%lf", &transactionValue);       // Retrieve the current transaction value as an double
        runningSum += transactionValue;                         //added to the running sum
        tokenized_line = strtok (NULL, ",");                    // Change pointer right after the next comma so we can retrieve the next transaction value
    }

    // critical section begins
	if (error = pthread_mutex_lock(&balanceLock)) {
        fprintf(stderr, "Consumer thread failed to get mutex lock for accessing balance array.\n");
        exit(EXIT_FAILURE);
    }

    balance[customerID] += runningSum;                          //adding runningSum to the account of customerID

    //critical section ends
    if (error = pthread_mutex_unlock(&balanceLock)) {
        fprintf(stderr, "Consumer thread failed to unlock mutex lock for accessing balance array.\n");
        exit(EXIT_FAILURE);
    }  

    return 0;
}


// consumer function
void *consumer(void *arg){    

    if (printLog) {
        fprintf(fpLog,"consumer %d\n", *(int *) arg);  //print consumer when launched
        fflush(fpLog);
    }

    //TODO: keep reading from queue and process the data
    // feel free to change
    char tmpLine[chunkSize];
    char tmpLinePrintLog[chunkSize];
    int retCode = 0;
    int customerID;
    int error;

    
    while (1) { 
        
        //begin of critical section - initiate mutex lock 
        if (error = pthread_mutex_lock(&bufferLock)) {
            fprintf(stderr, "Consumer thread failed to get hold of mutex lock.\n");
            exit(EXIT_FAILURE);
        }

        memset(tmpLine, '\0', chunkSize);        // reset tmpLine

        
        while (buffer.size == 0) {  //while there are no items in the queue
            
            // block consumer thread if there are no items in the queue
            if (error = pthread_cond_wait(&condQueueSize, &bufferLock)) {      
                fprintf(stderr, "Consumer thread failed to block on the conditional variable.\n");
                exit(EXIT_FAILURE);
            }
        }            
        
        strcpy(tmpLine, buffer.head->transactions); //pull transaction in the queue into tmpLine
        
        // remove transaction (that was just copied into tmpLine) from the queue        
        dequeue(&buffer);


        if (printLog) {
            memset(tmpLinePrintLog, '\0', chunkSize);  // reset tmpLinePrintLog
            strcpy(tmpLinePrintLog, tmpLine);          //copy tmpLine into tmpLinePrintLog
            sscanf(strtok (tmpLinePrintLog, ","), "%d", &customerID);  //pulling customer id only to check if it is an EOF special packet

            if(customerID < 0) {   //EOF
                fprintf(fpLog,"consumer %d: line -1\n", *(int *) arg);      // print consumer thread number and -1 indicating EOF          
            } else {
                fprintf(fpLog,"consumer %d: line %d\n", *(int *) arg, lineNumber); // print consumer thread number and the line number consumer thread is working on         
                lineNumber++;                                                      //increment line number
            }            
            fflush(fpLog);            
        }
       
        //parse transacation. when parse function returns -1, it means thread got EOF special packet
        if (parse(tmpLine) == -1) {   
            //unlock mutex lock - scenario EOF
            if (error = pthread_mutex_unlock(&bufferLock)) {   
                fprintf(stderr, "Consumer thread failed to unlock mutex lock for buffer\n");
                exit(EXIT_FAILURE);
            }
            if (sem_post(&bufferSizeSemaphore)) {
                perror("Consumer thread failed to unlock semaphore");
                exit(EXIT_FAILURE);
            }

            break;       //break out of the outer while loop
        }         
        
        //critical section ends
        //unlock mutex lock - scenario thread processed transaction line 
        if (error = pthread_mutex_unlock(&bufferLock)) {
            fprintf(stderr, "Consumer thread failed to unlock mutex lock for buffer\n");
            exit(EXIT_FAILURE);
        }
        if (sem_post(&bufferSizeSemaphore)) {
            perror("Consumer thread failed to unlock semaphore");
            exit(EXIT_FAILURE);
        }
    }
    
    return NULL; 
}


