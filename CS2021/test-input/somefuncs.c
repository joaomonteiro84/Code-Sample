int return_five(){
  return 5;
}

int global_int = 2;

int my_pow(int base, int exp){
  int pow=1;
  for(int i=0; i<exp; i++){
    pow *= base;
  }
  return pow;
}

char an_array[] = "abcde";

void triple(int *xp){
  *xp = *xp * 3;
  return;
}

