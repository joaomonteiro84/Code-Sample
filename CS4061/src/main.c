/*test machine: apollo.cselabs.umn.edu
* group number: G23
* name: Joao Monteiro, John Spruth, Jacob Johnson
* x500: monte092, sprut006, joh19042 */

#include "header.h"

/*Global variable*/
queue_t buffer;                                                // queue buffer
sem_t bufferSizeSemaphore;
pthread_mutex_t bufferLock = PTHREAD_MUTEX_INITIALIZER;        // mutex lock for enqueuing and dequeuing the buffer
pthread_mutex_t balanceLock = PTHREAD_MUTEX_INITIALIZER;       // mutex lock for updating balance array
pthread_cond_t condQueueSize = PTHREAD_COND_INITIALIZER;       // conditional variable for testing if buffer is empty
int numberConsumers;                                           // int variable that will hold the number of consumer threads
int printLog = 0;                                              /* int variable that will indicate if log should be printed. 
                                                                  0 means to NOT print log, 1 means to print log. So Default is no printing.*/
FILE *fpLog;                                                   // File pointer to log output
unsigned int bufferSize;                                       //bounded buffer size
int boundedBufferFlag = 0;                                     //int variable that will indicate whether user wants to use bounded buffer queue
                                                               //0 means unbounded buffer, 1 means bounded buffer. So default is unbounded buffer


/**
 * Write final balance to a single file.
 * The path name should be output/result.txt
 */
void writeBalanceToFiles(void) {    
    FILE * fpOutput;                                       // File pointer to output/results.txt
    fpOutput = fopen ("output/result.txt", "w");          // open file

    if (fpOutput == NULL) {
        perror("Error saving output/result.txt\n");
        exit(EXIT_FAILURE);
    }

    double totalBalanceChange = 0.0;                        // Initiate total balance change
    int nCharactersWritten;                                 // Number characters written. It will be used to check error in fprintf

    for (int j = 0; j < acctsNum; j++) {        
        totalBalanceChange += balance[j];
        nCharactersWritten = fprintf(fpOutput, "%d\t%lf\n", j, balance[j]);    // print balance for j-th customer

        if (nCharactersWritten < 0) {
            fprintf(stderr, "Error printing balance\n");
            exit(EXIT_FAILURE);
        }
	}
    
    nCharactersWritten = fprintf(fpOutput, "All: \t%lf\n", totalBalanceChange);  // print inf file the total assets changes
    if (nCharactersWritten < 0) {
        fprintf(stderr, "Error printing assets change\n");
        exit(EXIT_FAILURE);
    }

    fclose(fpOutput);                                     // Close the output file
}

int main(int argc, char *argv[]){

    /*Reading and checking input arguments */
    if (argc < 3 || argc > 5){                                              // Invalid number of arguments
        fprintf(stderr, "Invalid number of arguments. \n");                         // print error
        fprintf(stderr, "Usage: ./bank #consumers inputFile [option] [#queueSize]. \n");
        exit(EXIT_FAILURE);                                                     // exit with failure
    } else if (argc == 4) {                                                     // only aceptable if option is -p
        if (strcmp(argv[3], "-b") == 0 || strcmp(argv[3], "-bp") == 0) {
            fprintf(stderr, "Error: Missing #queueSize. \n");
            fprintf(stderr, "Usage: ./bank #consumers inputFile [option] [#queueSize]. \n");
            exit(EXIT_FAILURE); 
        } else if (strcmp(argv[3], "-p") == 0) {                                // confirm
            printLog = 1;                                                       // change int variable that indicates to print log
            
        } else {
            fprintf(stderr, "Invalid option. Possible options are -b, -p or -bp\n.");
            exit(EXIT_FAILURE);
        }
    } else if (argc == 5){                               // 5 arguments (optional flag)
        if (strcmp(argv[3], "-b") != 0 &&  strcmp(argv[3], "-p") != 0 && strcmp(argv[3], "-bp") != 0) { //invalid option argument
           fprintf(stderr, "Invalid option. Possible options are -b, -p or -bp\n.");
           exit(EXIT_FAILURE);
        } else {                                                                     //valid option argument
            if (strcmp(argv[3], "-b") == 0 || strcmp(argv[3], "-bp") == 0){          //using bounded buffer
                bufferSize = atoi(argv[4]);
                boundedBufferFlag = 1;                                              //indicate user wants bounded buffer queue
                  
                if (bufferSize <= 0) {                                        //checking if bounded buffer size is great than 0.
                    fprintf(stderr, "#queueSize must be greater than 0.\n");
                    exit(EXIT_FAILURE);
                }
            }
            if (strcmp(argv[3], "-p") == 0 || strcmp(argv[3], "-bp") == 0) {        // user wants to generate log
                printLog = 1;                                                       // change int variable that indicates to print log
            }
        }        
    } 

    numberConsumers = atoi(argv[1]);                                                // number of consumer threads
   
    if (numberConsumers <= 0) {                                                     // checking whether the number of consumer threads is invalid
        fprintf(stderr, "Error: Invalid argument for number of consumers threads.\n");
        exit(EXIT_FAILURE);
    }
    
    char * fileName = argv[2];                                                     // input file name    
   
    int nLines = 0;
    char ch;

    FILE *fp;
    
    if (boundedBufferFlag == 0) { //which means user wants unbounded buffer queue
      fp = fopen(fileName, "r");
      while ((ch = fgetc(fp)) != EOF) {
          if (ch == '\n') {
              nLines++;
          }          
      }
      bufferSize = nLines + 1;  //so BufferSize will alwyas be greater than the number of lines in the file provided by the user. thus providing an unbounded buffer
      fclose(fp);
    } 
    
    bookeepingCode();    
    
    queue_init(&buffer);                                                           // initialize queue
    
     //initiate semaphore
    if (sem_init(&bufferSizeSemaphore, 0, bufferSize)) {
        perror("Failed to initiate semaphore.");
        exit(EXIT_FAILURE);
    }                                


    int i;             
    pthread_t tidProducer[1];                                                       // producer  thread
    pthread_t *tidConsumer;
    
    // consumer threads - allocating space
    if ((tidConsumer = malloc(sizeof(pthread_t)*numberConsumers)) == NULL) {
        perror("Failed to allocate space for consumer threads");
        exit(EXIT_FAILURE);
    } 

    int *consumerID;
    
    // consumer threads id - allocating space
    if ((consumerID = malloc(sizeof(int)*numberConsumers))== NULL) {
        perror("Failed to allocate space for consumer thread ids");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < numberConsumers; i++) {                                         // define consumer thread ID - starting from 0
        consumerID[i] = i;
    }
    

    if (printLog) {       
       fpLog = fopen ("output/log.txt", "w");                                     // open output/log.txt to print log in it

       if (fpLog == NULL) {
        perror("Error opening output/log.txt");
        exit(EXIT_FAILURE);
        }
    }

   
    for (int k = 0; k < acctsNum; k++) {                                          // initiating balance array to 0.0 for each bank account
        balance[k] = 0.0;
    }
    
    int error;
    // create a producer thread. on success pthread_create() returns 0
    if (error = pthread_create(&(tidProducer[0]), NULL, producer, (void *) fileName)) {  
        fprintf(stderr, "Failed to create producer thread: %s\n", strerror(error));
        exit(EXIT_FAILURE);            
    }

    // create each consumer thread. on success pthread_create() returns 0 
    for (i = 0 ; i < numberConsumers; i++) {        
        if (error = pthread_create(&(tidConsumer[i]), NULL, consumer, &consumerID[i])) {
            fprintf(stderr, "Failed to create consumer thread %d: %s\n", i, strerror(error));
            exit(EXIT_FAILURE); 
        }
    }

    // wait for produce thread to complete execution. on success pthread_join returns 0    
    if (error = pthread_join(tidProducer[0], NULL)){
        fprintf(stderr, "Failed to join producer thread: %s.\n", strerror(error));
        exit(EXIT_FAILURE);
    }

    // wait for all consumer threads to complete execution. on success pthread_join returns 0
    for (i = 0; i < numberConsumers; i++) {
        if (error = pthread_join(tidConsumer[i], NULL) != 0){
            fprintf(stderr, "Failed to join consumer thread %d: %s\n", i, strerror(error));
            exit(EXIT_FAILURE);
        }
    }    

    //Write the final output
    writeBalanceToFiles();


    if (printLog) {
        fclose(fpLog);       //close log file
    }

   
    free(consumerID);       //de-allocate space reserved to consumerID
    free(tidConsumer);      //de-allocate space reserved to tidConsumer
  

    return 0; 
}
