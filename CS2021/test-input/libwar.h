#ifndef LIBWAR_H
#define LIBWAR_H

/* A type to track the overall statistics of war */
typedef struct {
  int winner[2];
  int battles;
  int wars;
} warstats;

void read_deck(char *filename, queue *deck);
void deal2(queue *deck, queue *hand1, queue *hand2);
int start_battle(queue *q1, queue *q2, stack *s1, stack *s2, FILE *logfile);
void end_battle(queue *q1, queue *q2, stack *s1, stack *s2, FILE *logfile);
void playwar(queue *deck, warstats *stats, FILE *logfile);

#endif
