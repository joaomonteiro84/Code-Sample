/*test machine: apollo.cselabs.umn.edu
* group number: G23
* name: Joao Monteiro, John Spruth, Jacob Johnson
* x500: monte092, sprut006, joh19042 */

#include "utils.h"
// pthread.h included in header.h

// Feel free to add any functions or global variables

/* File operations */
int writeLineToFile(char *filepath, char *line) {
    int fd;
    if((fd = open(filepath, O_CREAT | O_WRONLY | O_APPEND, 0777)) == -1){
        perror("Error:");
        return -1;
        // change function from void
    }
    int ret;
    if((ret  = write(fd, line, strlen(line))) == -1){
        perror("Error:");
        return -1;
    }

    return 0;

}

FILE * getFilePointer(char *inputFileName) {
    return fopen(inputFileName, "r");
    // need to check for null anywhere this function is called
}

ssize_t getLineFromFile(FILE *fp, char *line, size_t len) {
    memset(line, '\0', len);
    return getline(&line, &len, fp);
    // need to check for errors anywhere this function is called
}

int _removeOutputDir(){
    pid_t pid;
    if((pid = fork()) == -1){
        perror("Error:");
        return -1;
    }
    if(pid == 0){
        char *argv[] = {"rm", "-rf", "output", NULL};
        if (execvp(*argv, argv) == -1) {
            perror("Error:");
            return -1;
        }
        exit(EXIT_SUCCESS);
    } else{
        wait(NULL);
    }

    return 0;

}

int _createOutputDir(){
    if(mkdir("output", ACCESSPERMS) == -1){
        perror("Error:");
        return -1;
    }

    return 0;

}

void bookeepingCode(){
    _removeOutputDir();
    sleep(1);
    _createOutputDir();
}

/* other functions */


//initiate queue
void queue_init(queue_t *queue) {
    queue->head = NULL;
    queue->tail = NULL;
    queue->size = 0;
}

//pull queue size
int queue_size(queue_t *queue) {
    return queue->size;
}

//add newTransat to queue
int enqueue(queue_t *queue, char *newTransact) {     

    int error = 0;

    //block producer thread if queue size is equal to the bufferSize
    while (sem_wait(&bufferSizeSemaphore)) {
        if (errno != EINTR) {
            fprintf(stderr, "Produce thread failed to lock semaphore.\n");
            return -1;
        }
    }

    //get hold of buffer mutex lock
    if (error = pthread_mutex_lock(&bufferLock)) {
        fprintf(stderr, "Producer thread failed to lock buffer lock");
        return -1;
    }

    //create new node for the queue
    node_t *newNode;

    if ((newNode = (node_t*)malloc(sizeof(node_t))) == NULL) {
        perror("Producer thread failed to allocate space for queue node");
        return -1;
    }

    //copy line to new node
    strcpy(newNode->transactions, newTransact);
    newNode->next = NULL;
    
    //add new node to queue
    if (queue->size == 0) {
        queue->head = queue->tail = newNode;
    } else {
        queue->tail->next = newNode;
        queue->tail = newNode;
    }
    queue->size++;    //update queue size

    //send signal to other threads the queue size changed
    if (error = pthread_cond_signal(&condQueueSize)) {
        fprintf(stderr, "Producer thread failed to notify other threads of change in the queue\n");
        return -1;
    }

    //unlock buffer mutex lock
    if (error = pthread_mutex_unlock(&bufferLock)) {
        fprintf(stderr, "Producer thread failed to unlock buffer lock.\n");
        return -1;
    }

    return 0;
}

//remove head from queue
void dequeue(queue_t *queue) {
 
    //check if queue is empty
    if (queue->size == 0) {
        return;
    }

    node_t *newHead = queue->head->next;  //new head
 
    free(queue->head);
    queue->head = newHead;             // update queue head
    queue->size--;                     //update queue size
}