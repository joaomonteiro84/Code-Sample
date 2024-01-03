#include <stdlib.h>
// #include "list.h"

typedef struct node_struct {
  char data[256];
  struct node_struct *next;
} node_t;

typedef struct {
  node_t *head;
  int size;
} list_t;

node_t nodes[15] = {
  {.data="Within C++, there is a much smaller and cleaner language struggling to get out. (It is C.)",
   .next=&nodes[1]},
  {.data="So don't use C.",
   .next=&nodes[2]},
  {.data="Java prevents you from shooting yourself in the foot by cutting off all your fingers.",
   .next=&nodes[3]},
  {.data="But at least you don't need Skynet to compile it which is more than I can say for C++.",
   .next=&nodes[4]},
  // ANSWER
  {.data="C++ makes it harder, but when you do, it blows away your whole leg.",  
   .next=&nodes[5]},
  {.data="When Dennis Ritchi discovered this, he immediately bought stock in steel-footed boots.",
   .next=&nodes[6]},
  {.data="This is why most programmers are such poor dancers.",
   .next=&nodes[7]},
  {.data="I made up the term \"object-oriented\", and I can tell you I did not have C++ in mind.",
   .next=&nodes[8]},
  {.data="C++ is a language strongly optimized for liars and people who go by guesswork and ignorance.",
   .next=&nodes[9]},
  {.data="C++ maintains the original spirit of C, that the programmer not the language is in charge.",
   .next=&nodes[10]},
  {.data="C++ is an octopus made by nailing extra legs onto a dog.",
   .next=&nodes[11]},
  {.data="The only way to do good, efficient, and system-level and portable C++ ends up to limit\nyourself to all the things that are basically available in C.",
   .next=&nodes[12]},
  {.data="When you’re programming C++ no one can ever agree on which ten percent of the language is\nsafe to use.",
   .next=&nodes[13]},
  {.data="C++ is just an abomination. Everything is wrong with it in every way.",
   .next=&nodes[14]},
  {.data="The major cause of complaints is C++ undoubted success. As someone remarked:\nThere are only two kinds of programming languages: those people always bitch about and those nobody uses.",
   .next=NULL},
};

list_t choices_actual = {
  .head = &nodes[0],
  .size = 15,
};

list_t *choices = &choices_actual;

int max_size(){
  return choices->size;
}

// Get a pointer to the data at the specified index. Returns NULL if
// the index is out of bounds. Uses an iterative approach.
char *list_get(list_t *list, int index){
  if(index < 0 || index >= list->size){
    // printf("index %d out of bounds for list size %d\n",
    //        index, list->size);
    return NULL;
  }
  int i = 0;
  node_t *ptr = list->head;
  while(ptr != NULL && i!=index){
    i++;
    ptr = ptr->next;
  }
  return ptr->data;
}

char *get_it(int i){
  char *ans = list_get(choices, i);
  if(ans == NULL){
    ans = "Out of bounds";
  }
  return ans;
}
