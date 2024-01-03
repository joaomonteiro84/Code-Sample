#ifndef LIBCARDLIST_H
#define LIBCARDLIST_H

#include <stdlib.h>
#include <stdio.h>

typedef struct {
  int face;
  char suit;
} card;
typedef card node_data;

// Utility functions
node_data str2card(char *buf);
char *card2str(node_data c, char *buf);

/* List Functions */
typedef struct node {
  node_data data;
  struct node *next;
} node;
int length(node *l);
node *reverse(node *l);
void print_list(node *l, FILE *f);

/* Stack functions */
typedef struct stack {
  node *top;
} stack;
stack *new_stack();
void stack_free(stack *s);
stack *stack_empty(stack *s);
void stack_push(stack *s, node_data p);
void stack_pop(stack *s);
node_data stack_top(stack *s);

/* Queue functions */
typedef struct queue {
  node *front;
  node *rear;
} queue;
queue *new_queue();
void queue_free(queue *q);
int queue_empty(queue *q);
void queue_add(queue *q, node_data p);
void queue_remove(queue *q);
node_data queue_front(queue *q);
void queue_rotate(queue *q);

node *list_copy(node *l);
queue *queue_copy(queue *qq);

#endif
