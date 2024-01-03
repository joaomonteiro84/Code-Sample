#include <stdio.h>
#include "libcardlist.h"
#include "libwar.h"

warstats stats = {{0,0}, 0, 0};	/* Global variable to track stats */
 
int main(int argc, char **argv){
  if(argc < 3){
    printf("usage: %s cardfile ngames [logfile]\n",argv[0]);
    return 1;
  }
  queue *master_deck = new_queue();       /* The master deck */
  read_deck(argv[1], master_deck);        /* read from the command line file */
  int ngames = atoi(argv[2]);             /* How many games to play */
  FILE *logfile = fopen("/dev/null","w"); /* default no logging */
  if(argc > 3){
    fclose(logfile);
    logfile = fopen(argv[3],"w");         /* A log file for tracking the games */
  }
  int i;
  for(i=0; i<ngames; i++){                /* Play games */
    fprintf(logfile,"==Game %d==\n",i);
    queue *deck = queue_copy(master_deck); /* Make a copy of the master deck */
    playwar(deck,&stats,logfile);          /* Play war with it, deck will be emptied */
    queue_rotate(master_deck);		   /* Set up for next game */
    queue_free(deck);
  }
  /* Clean up */
  queue_free(master_deck);
  fclose(logfile);
  
  printf("----------------------\n");
  printf("Final stats\n");
  printf("%6.2f P1 Win Ratio\n",((double) stats.winner[0])/ngames);
  printf("%6.2f Avg battles per game\n",((double) stats.battles)/ngames);
  printf("%6.2f Avg wars per game\n",((double) stats.wars)/ngames);
}
