void quintuple(int *xp){
  int x = *xp;
  x *= 2;
  x += x;
  x += *xp;
  *xp = x;
  return;
}

int sum10(){
  int sum =0;
  for(int i=1; i<=10; i++){
    sum += i;
  }
  return sum;
}

int modulo(int x, int y){
  int rem = x % y;
  return rem;
}
