#include <stdio.h>

void print_greeting();
extern int ngreet;
int main(){
  for(int i=0; i<ngreet; i++){
    print_greeting();
  }
  return 0;
}
