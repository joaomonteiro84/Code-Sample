#ifndef COINS_H
#define COINS_H

// Type for collections of coins
typedef struct {
  char quarters;
  char dimes;
  char nickels;
  char pennies;
} coins_t;
// coint_t has the following memory layout and argument passing
// properties:
//
// |          | Pointer | Packed | Packed |
// |          |  Memory | Struct | Struct |
// | Field    |  Offset | Arg#   | Bits   |
// |----------+---------+--------+--------|
// | quarters |      +0 | #1     | 0-31   |
// | dimes    |      +1 | #1     | 32-63  |
// | nickels  |      +2 | #2     | 0-31   |
// | pennies  |      +3 | #2     | 32-63  |

int set_coins(int cents, coins_t *coins);
int total_coins(coins_t coins);

#endif
