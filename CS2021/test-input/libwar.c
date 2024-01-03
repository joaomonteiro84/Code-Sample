#include "libcardlist.h"
#include "libwar.h"

/* read contents of a file containing cards into a queue */
void read_deck(char *filename, queue *deck){
  FILE *f = fopen(filename,"r");
  int status;
  card c;
  char buf[1024];
  for(status = fscanf(f,"%s",buf);
      status != EOF;
      status = fscanf(f,"%s",buf)){
    c = str2card(buf);
    queue_add(deck,c);
  }
  fclose(f);
}

/* deal two cards from the deck into player hands */
void deal2(queue *deck, queue *hand1, queue *hand2){
  while(!queue_empty(deck)){
    queue_add(hand1, queue_front(deck));
    queue_remove(deck);
    if(!queue_empty(deck)){
      queue_add(hand2, queue_front(deck));
      queue_remove(deck);
    }
  }
}

/* Run a game of war */
void playwar(queue *deck, warstats *stats, FILE *logfile){
  queue *q1 = new_queue(), *q2 = new_queue();
  stack *s1 = new_stack(), *s2 = new_stack();
  int battles=0, wars=0;

  /* Deal cards */
  deal2(deck, q1, q2);

  while(!queue_empty(q1) && !queue_empty(q2)){
    /* Game continues */
    battles++;
    fprintf(logfile,"Queue 1: ");
    print_list(q1->front,logfile);
    fprintf(logfile,"Queue 2: ");
    print_list(q2->front,logfile);

    wars += start_battle(q1,q2,s1,s2,logfile);

    fprintf(logfile,"Stack 1: ");
    print_list(s1->top,logfile);
    fprintf(logfile,"Stack 2: ");
    print_list(s2->top,logfile);
    fprintf(logfile,"\n");

    end_battle(q1,q2,s1,s2,logfile);
  }

  /* One queue is empty, check for winner */
  int winner = !queue_empty(q1) ? 1 : 2;
  queue_free(q1); queue_free(q2);
  stack_free(s1); stack_free(s2);

  fprintf(logfile,"P%d wins %d battles %d wars\n",winner,battles,wars);
  stats->winner[winner-1]++;	/* Update global winner */
  stats->battles += battles;	/* Update global battles fought */
  stats->wars += wars;		/* Update global wars fought */
  return;
}

/* pull cards off hands (queues) into battle stacks until a winner is
   found; returns the number of battles "wars" that occurred (number
   of times equal cards were on the top of the battle stacks */
int start_battle(queue *q1, queue *q2, stack *s1, stack *s2, FILE *logfile){
  int warc = 0;
  stack_push(s1, queue_front(q1)); queue_remove(q1); /* move front card of q1 to the top of s1 */
  stack_push(s2, queue_front(q2)); queue_remove(q2); /* move front card of q2 to the top of s2 */

  while(stack_top(s1).face == stack_top(s2).face && !(queue_empty(q1) && queue_empty(q2))){
    /* A war has occurred */
    warc++;
    fprintf(logfile,"War %d\n",warc);
    int i;
    for(i=0; i<3; i++){
      if(!queue_empty(q1)){
	stack_push(s1,queue_front(q1)); queue_remove(q1);
      }
    }
    for(i=0; i<3; i++){
      if(!queue_empty(q2)){
	stack_push(s2,queue_front(q2)); queue_remove(q2);
      }
    }
  }
  return warc;
}  

/* then move both battle stacks into the winner's hand (queue);
   assumes that the top cards in the battle stack are not equal */
void end_battle(queue *q1, queue *q2, stack *s1, stack *s2, FILE *logfile){
  if(stack_top(s1).face > stack_top(s2).face){ /* Player 1 has better card */ 
    fprintf(logfile, "Player 1 wins\n");
    while(!stack_empty(s2)){
      queue_add(q1,stack_top(s2)); stack_pop(s2);
    }
    while(!stack_empty(s1)){
      queue_add(q1,stack_top(s1)); stack_pop(s1);
    }
  }
  else{				/* Player 2 has better card */
    fprintf(logfile, "Player 2 wins\n");
    while(!stack_empty(s1)){
      queue_add(q2,stack_top(s1)); stack_pop(s1);
    }
    while(!stack_empty(s2)){
      queue_add(q2,stack_top(s2)); stack_pop(s2);
    }
  }
}  
