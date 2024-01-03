/*test machine: apollo.cselabs.umn.edu
* group number: G23
* name: Joao Monteiro, John Spruth, Jacob Johnson
* x500: monte092, sprut006, joh19042 */

#ifndef UTILS_H
#define UTILS_H

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <stdbool.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>

#define chunkSize 1024
#define acctsNum 1000

/* shared array definition */
double balance[acctsNum];

/* file I/O */
/**
 * Get a pointer to a opened file based on the file name
 * @param *inputFileName  the file path
 * @return a file pointer pointing to the file
 */
FILE * getFilePointer(char *inputFileName);

/**
 * Read an entire line from a file
 * @param  *fp    the file to be read
 * @param  *line  contain the line content
 * @param  len    the size of the line
 * @return the number of character reads (including the newline \n, but not including terminator)
           -1 when reaching the end of file or error occurs
 */
ssize_t getLineFromFile(FILE *fp, char *line, size_t len);

/**
 * Open a file, and write a line to the file
 * @param *filepath  the file path
 * @param *line      the line content
 */
int writeLineToFile(char *filepath, char *line);

/* directory */
void bookeepingCode();

/* other function declaration */
typedef struct node {
    char transactions[chunkSize];
    struct node *next;    
} node_t;


typedef struct {
    node_t *tail, *head;
    int size;
} queue_t;

/**
 * Initialize the fields of a queue data structure
 * @param *queue   the queue to initialize
 */
void queue_init(queue_t *queue);

/**
 * Returns the number of items in a queue
 * @param *queue   the queue to get the number of items of.
 */
int queue_size(queue_t *queue);

/**
 * Adds an item to the end of a queue
 * @param *queue   the queue to add the item to.
 * @param *newTransact  the transaction line to add to the queue.
 */
int enqueue(queue_t *queue, char *newTransact);

/**
 * Removes an item from the head of a queue.
 * @param *queue    the queue to remove the head item from.
 */
void dequeue(queue_t *queue);

/* External variables for mutual exclusion and sharing amongst threds */
extern pthread_mutex_t bufferLock;
extern pthread_cond_t condQueueSize;
extern sem_t bufferSizeSemaphore;
extern int printLog;
extern FILE *fpLog;


#endif
