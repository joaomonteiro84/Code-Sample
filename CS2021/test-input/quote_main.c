#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Defined in list_data.c
int max_size();
char *get_it(int i);

char correct[128] = "C++ makes it harder, but when you do, it blows away your whole leg.";

int main(int argc, char *argv[]){
  printf("Complete this sentence by C++ creator Bjarne Stroustrup:\n");
  printf("C makes it easy to shoot yourself in the foot; ...\n");
  printf("\n");
  printf("enter a number from 0 to %d on command line\n",max_size());
  if(argc < 2){
    printf("no command line arg given\n");
    return 0;
  }
  int index = atoi(argv[1]);
  char *selected = get_it(index);
  printf("%d: %s\n",index,selected);
  int diff = strcmp(selected,correct);
  printf("\n");
  if(diff == 0){
    printf("Correct!\n");
  }
  else{
    printf("Have a nice tall glass of ... NOPE.\n");
  }
  return 0;
}

