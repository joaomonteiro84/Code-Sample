/* This file contains functions to operate on a lists, stacks, and
   queues of cards */

#include <stdio.h>
#include <stdlib.h>

/* Report an error and exit the program with a failure */
void cardlist_error(char *msg){
  fprintf(stderr,"libcardlist: %s\n",msg);
  exit(EXIT_FAILURE);
}

/* Basic type for a card */
typedef struct {
  int face;			/* 2-14 */
  char suit;			/* C H S D */
} card;

/* Convert a string like 14D into a card */
card str2card(char *buf){
  card c;
  sscanf(buf,"%d%c",&c.face,&c.suit);
  return c;
}

/* Given a card c, put a string like 14D in buf representing it.  Good
   for printing  */
char *card2str(card c, char *buf){
  sprintf(buf, "%d%c", c.face, c.suit);
  return buf;
}

/* Lists are of cards */
typedef card node_data;

/* List Functions */

/* Basic type for a list: data and next */
typedef struct node {
  node_data data;
  struct node *next;
} node;

/* Returns how many nodes are in a list */
int length(node *l){
  int n = 0;
  while(l != NULL){
    n++;
    l = l->next;
  }
  return n;
}

   

/* Reverses a list, creates a fresh and distinct copy of the list */
node *reverse(node *l){
  node *r = NULL;
  while(l != NULL){
    node *new = malloc(sizeof(node));
    new->data = l->data;
    new->next = r;
    r = new;
    l = l->next;
  }
  return r;
}

/* Print a list of cards to a file pointer */
void print_list(node *l, FILE *f){
  char buf[1024];		/* Use this for string conversion */
  while(l != NULL){		/* Til end of list */
    fprintf(f,"%3s ", card2str(l->data,buf)); /* Convert to string and print */
    l = l->next;			     /* Advance to next */
  }
  fprintf(f,"\n");
}


/* Stack functions */

/* Basic type for a stack */
typedef struct stack {
  node *top;
} stack;

/* Make a new stack: allocate memory and set its top pointer to
   initially be NULL for an empty stack */
stack *new_stack(){
  stack *s = malloc(sizeof(stack));
  s->top = NULL;
  return s;
}

/* Return 1 if the stack is empty and 0 otherwise  */
int stack_empty(stack *s){
  return s->top == NULL;
}

/* Push something on the top of the stack */
void stack_push(stack *s, node_data p){
  node *new = malloc(sizeof(node)); /* New node for the new data */
  new->data = p;		    /* New node gets the new data */
  new->next = s->top;		    /* new will be on top, point it at current top */
  s->top = new;			    /* new is on top now */
}

/* Remove the top element of the stack */
void stack_pop(stack *s){
  if(!stack_empty(s)){		/* If the stack is not empty */
    node *remove = s->top;	/* Track what's being removed */
    s->top = s->top->next;	/* Advance the top down one */
    free(remove);		/* Get rid of the old top node */
  }
}

/* Retrive data from the top of the stack */
node_data stack_top(stack *s){
  if(!stack_empty(s)){		/* If the stack is not empty */
    return (s->top->data);	/* Return the data */
  }
  else{				/* Otherwise there is an error */
    cardlist_error("stack_top called on empty stack");
  }
}

void stack_free(stack *s){
  while(!stack_empty(s)){
    stack_pop(s);
  }
  free(s);
}


/* Queue functions */

/* Basic type for the queue data structure */
typedef struct queue {
  node *front;			/* Front of the line */
  node *rear;			/* Back of the line */
} queue;

/* Make a new queue which is initially empty */
queue *new_queue(){
  queue *q = malloc(sizeof(queue));
  q->front = NULL;
  q->rear = NULL;
  return q;
}

/* Returns 1 if the queue is empty and 0 otherwise */
int queue_empty(queue *q){
  return q->front == NULL;
}

/* Add something to the rear of the queue */
void queue_add(queue *q, node_data p){
  node *new = malloc(sizeof(node)); /* Adding a new node */
  new->data = p;		    /* Set new node's data */
  new->next = NULL;		    /* It will be the end of the line */
  if(queue_empty(q)){		/* First node to be added */
    q->front = new;		/* Front and back are new node */
    q->rear = new;
  }
  else {			/* Not first node */
    q->rear->next = new;	/* Current rear is second to last */
    q->rear = new;		/* new guy is last */
  }
}

/* Remove first element of the queue */
void queue_remove(queue *q){
  if(!queue_empty(q)){		/* If the queue is not empty */
    node *remove = q->front;	/* Track who is being removed */
    q->front = q->front->next;	/* Second in line is now at front */
    free(remove);		/* Remove the old front */
  }
}

/* Get the data for the front of the queue */
node_data queue_front(queue *q){
  if(!queue_empty(q)){		/* If queue is not empty */
    return (q->front->data);	/* Get data for front node */
  }
  else{				/* Otherwise this is an error */
    cardlist_error("queue_front called on empty queue");
  }
}

void queue_free(queue *q){
  while(!queue_empty(q)){
    queue_remove(q);
  }
  free(q);
}

/* Make a copy of a list using a queue */
node *list_copy(node *l){
  queue *q = new_queue();
  while(l != NULL){
    queue_add(q,l->data);
    l = l->next;
  }
  node *new = q->front;
  free(q);
  return new;
}

/* Make an identical copy of a queue */
queue *queue_copy(queue *qq){
  queue *q = new_queue();
  node *l = qq->front;
  while(l != NULL){
    queue_add(q,l->data);
    l = l->next;
  }
  return q;
}

/* Move a card from the front of the queue to the back */
void queue_rotate(queue *q){
  card c = queue_front(q);
  queue_remove(q);
  queue_add(q,c);
}


