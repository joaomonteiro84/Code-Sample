/*test machine: apollo.cselabs.umn.edu
* group number: G23
* name: Joao Monteiro, John Spruth, Jacob Johnson
* x500: monte092, sprut006, joh19042 */

#ifndef CONSUMER_H
#define CONSUMER_H

#include "utils.h"

void *consumer(void *arg);
int parse(char *line);

extern pthread_mutex_t balanceLock;
#endif