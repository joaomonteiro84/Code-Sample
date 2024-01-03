// globals.c: simple program which has several global variables in
// it. Used for testing programs which manipulate ELF symbol tables.

#include <stdio.h>

char string1[8]  = "Hello";
char string2[64] = "Goodbye cruel world";
char string3[16] = "All your bass";
int int1 = 0xAABBCCDD;
int int2 = 0xFFEEDDCC;
double a_doub = 1.234567;

int main(int argc, char *argv[]){
  printf("string1: %s\n",string1);
  printf("string2: %s\n",string2);
  printf("string3: %s\n",string3);
  printf("int1: %x\n",int1);
  printf("int2: %x\n",int2);
  printf("a_doub: %f\n",a_doub);
  return 0;
}
